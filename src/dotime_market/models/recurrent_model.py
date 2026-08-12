"""Gate-A backbone: a Δt-gated GatedDeltaProduct recurrent causal PFN.

Why this architecture (evidence trail in ``docs/results_summary.md``,
2026-07-31): the one-shot attention readout learns an amortized,
context-independent, duration-flat intervention response no matter the prior
(rungs 1–3 all falsified), while *composition* of one-step predictions
accrues correctly (+0.683 → +0.774 on the Gate-B benchmark with the same
weights). A recurrent state-space model makes that composition native:

- **Per-step intervention injection**: the do-tokens enter at every step of
  the window, so the response accrues through the state recurrence instead
  of being predicted in one shot.
- **Δt-exponentiated, input-dependent gates**: each channel's decay is
  ``exp(-softplus(a(u_t)) * Δt)`` — a learned linear ODE between
  observations (continuous time by construction; exact for the linear-drift
  priors). A gate near 1 is a permanent/integrator channel, a smaller gate a
  transient one, selectable in context — the transient-vs-permanent split
  the prior ladder tried and failed to teach the one-shot readout.
- **DeltaProduct state mixing** (products of Householder-style rank-1 delta
  updates per token, after GatedDeltaProduct / TempoPFN arXiv:2510.25502):
  non-diagonal state transitions for tracking the hidden confounder through
  regime changes, at parallelizable-recurrence cost.

The model consumes the SAME batch dict as ``ContinuousDoOverTimePFN`` and
returns the same quantile-head output, so the training loop
(``scripts/15_train_recurrent.py``), the do-ablation protocol (zeroing the
five intervention keys), and the baseline wrapper all carry over unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dotime.models.quantile_head import QuantileHead

__all__ = ["RecurrentDoTPFN"]


class _GatedDeltaProductBlock(nn.Module):
    """One recurrent block: Δt-gated decay + ``n_householder`` delta updates.

    State ``S`` has shape ``(B, n_heads, d_k, d_v)``. Per step::

        g_t = exp(-softplus(a(u_t)) * Δt_t)          (per-head decay)
        S   = g_t * S
        for each of n_householder sub-updates:
            k = l2norm(W_k u_t);  v = W_v u_t;  β = sigmoid(W_β u_t)
            S = S - β k (kᵀ S) + β k vᵀ              (delta rule)
        o_t = qᵀ S with q = W_q u_t

    Args:
        d_model: residual-stream width.
        n_heads: independent state heads.
        d_k, d_v: per-head key/value dims.
        n_householder: delta sub-updates per token (DeltaProduct order).
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_k: int = 32,
                 d_v: int = 32, n_householder: int = 2):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.n_householder = n_householder
        self.norm = nn.LayerNorm(d_model)
        self.a_proj = nn.Linear(d_model, n_heads)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.k_proj = nn.Linear(d_model, n_heads * d_k * n_householder)
        self.v_proj = nn.Linear(d_model, n_heads * d_v * n_householder)
        self.b_proj = nn.Linear(d_model, n_heads * n_householder)
        self.o_proj = nn.Linear(n_heads * d_v, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        """Run the recurrence over a sequence.

        Args:
            u: ``(B, T, d_model)`` residual stream.
            dts: ``(B, T)`` time gap preceding each step (Δt for step 0 is 1).

        Returns:
            ``(B, T, d_model)`` residual stream after this block.
        """
        bsz, t_len, _ = u.shape
        h = self.norm(u)
        # Precompute all projections; only the state update is sequential.
        decay = F.softplus(self.a_proj(h))                       # (B,T,H)
        gates = torch.exp(-decay * dts.unsqueeze(-1))            # (B,T,H)
        q = self.q_proj(h).view(bsz, t_len, self.n_heads, self.d_k)
        k = F.normalize(
            self.k_proj(h).view(bsz, t_len, self.n_heads, self.n_householder, self.d_k),
            dim=-1)
        v = self.v_proj(h).view(bsz, t_len, self.n_heads, self.n_householder, self.d_v)
        beta = torch.sigmoid(
            self.b_proj(h).view(bsz, t_len, self.n_heads, self.n_householder))

        state = u.new_zeros(bsz, self.n_heads, self.d_k, self.d_v)
        outs = []
        for t in range(t_len):
            state = gates[:, t].unsqueeze(-1).unsqueeze(-1) * state
            for j in range(self.n_householder):
                kj = k[:, t, :, j]                               # (B,H,dk)
                vj = v[:, t, :, j]                               # (B,H,dv)
                bj = beta[:, t, :, j].unsqueeze(-1)              # (B,H,1)
                k_s = torch.einsum("bhk,bhkv->bhv", kj, state)   # kᵀS
                state = state + torch.einsum(
                    "bhk,bhv->bhkv", kj, bj * (vj - k_s))
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
        o = torch.stack(outs, dim=1).reshape(bsz, t_len, -1)
        u = u + self.o_proj(o)
        return u + self.mlp(self.mlp_norm(u))


class RecurrentDoTPFN(nn.Module):
    """Recurrent causal PFN over the continuous-trainer batch contract.

    Args:
        n_max: padded variable count (matches the dataloader).
        embed_size: residual width.
        n_layers: recurrent blocks.
        n_heads, d_k, d_v, n_householder: block hyperparameters.
        tau_levels: quantile levels for the head (trainer default if None).

    Raises:
        KeyError: in ``forward`` if a required batch field is missing.
    """

    def __init__(self, n_max: int = 16, embed_size: int = 256,
                 n_layers: int = 4, n_heads: int = 4, d_k: int = 32,
                 d_v: int = 32, n_householder: int = 2,
                 tau_levels: Optional[List[float]] = None):
        super().__init__()
        self.n_max = n_max
        # Per-step features: values + observed-mask (2*n_max), Δt, onset flag,
        # active flag, intervention value, kind one-hot (3), target one-hot
        # (n_max), window start/end (norm), query flag.
        d_in = 2 * n_max + 1 + 1 + 1 + 1 + 3 + n_max + 2 + 1
        self.in_proj = nn.Linear(d_in, embed_size)
        self.blocks = nn.ModuleList([
            _GatedDeltaProductBlock(embed_size, n_heads, d_k, d_v, n_householder)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(embed_size)
        self.quantile_head = QuantileHead(embed_size, tau_levels=tau_levels)
        self.head = self.quantile_head  # trainer/wrapper contract

    def _features(self, batch: Dict[str, torch.Tensor]) -> tuple:
        x = batch.get("X_obs_norm", batch["X_obs"])              # (B,T,n_max)
        bsz, t_len, _ = x.shape
        times = batch["times"]                                   # (B,T)
        dts = torch.ones_like(times)
        dts[:, 1:] = batch["dts"]
        onset = batch["int_onset_idx"].view(-1, 1)               # (B,1)
        step_idx = torch.arange(t_len, device=x.device).view(1, -1)
        observed = (step_idx < onset).float()                    # (B,T)
        # Causal masking upstream zeroes post-onset values; the explicit
        # observed flag lets the state distinguish "masked" from "value 0".
        active = ((times >= batch["t_int_start"].view(-1, 1))
                  & (times < batch["t_int_end"].view(-1, 1))).float()
        kind = F.one_hot(batch["intervention_type"].long(), 3).float()
        target = F.one_hot(batch["intervention_target"].long().clamp(max=self.n_max - 1),
                           self.n_max).float()
        value = batch["intervention_value"].view(-1, 1)
        w_start = batch["intervention_time_start"].view(-1, 1)
        w_end = batch["intervention_time_end"].view(-1, 1)
        # Query step: nearest normalized observation time to query_time.
        span = (times[:, -1] - times[:, 0]).clamp(min=1e-6).view(-1, 1)
        times_norm = (times - times[:, :1]) / span
        q_idx = (times_norm - batch["query_time"].view(-1, 1)).abs().argmin(dim=1)
        query_flag = F.one_hot(q_idx, t_len).float()             # (B,T)

        a3 = active.unsqueeze(-1)                                # (B,T,1)
        const = torch.cat([value, kind, target, w_start, w_end], dim=-1)
        feats = torch.cat([
            x * observed.unsqueeze(-1),
            observed.unsqueeze(-1).expand(-1, -1, self.n_max)
            * batch["variable_mask"].unsqueeze(1),
            dts.unsqueeze(-1),
            observed.unsqueeze(-1),
            a3,
            a3 * const.unsqueeze(1).expand(-1, t_len, -1),
            query_flag.unsqueeze(-1),
        ], dim=-1)
        return feats, dts, q_idx

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the quantile matrix at the query step, shape ``(B, n_tau)``."""
        feats, dts, q_idx = self._features(batch)
        u = self.in_proj(feats)
        for blk in self.blocks:
            u = blk(u, dts)
        h = self.out_norm(u[torch.arange(u.shape[0], device=u.device), q_idx])
        return self.quantile_head(h)
