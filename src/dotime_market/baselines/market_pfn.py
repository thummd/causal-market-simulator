"""Trained continuous-time DoT-PFN checkpoints as dotime baselines.

Two registrations over one wrapper:

- ``"market-dotpfn"``  — the causal PFN: full intervention tokens.
- ``"ablated-dotpfn"`` — the predictive twin: the five do-information mixer
  inputs are zeroed at predict time, matching the ``ablate_intervention``
  training flag in :mod:`dotime_market.models.train`. Wrap the checkpoint
  that was *trained* ablated; ablating a causally-trained checkpoint at
  test time is a different (worse) ablation.

Episode -> batch conversion mirrors ``ContinuousExtendedPrior.generate_sample``
field-for-field on a batch of size 1 per query (suites are regular ``dt = 1``,
as ``build_market_suite`` enforces), and normalization reuses the public
``dotime.normalization.normalize_batch`` so train/eval statistics agree.
Predictions come back de-normalized to the raw scale, as the Baseline
protocol requires.
"""

from __future__ import annotations

import torch

from dotime import InterventionType
from dotime.baselines import register
from dotime.benchmarks import Episode
from dotime.normalization import normalize_batch

_KIND_INDEX = {
    InterventionType.HARD: 0,
    InterventionType.SOFT: 1,
    InterventionType.TIME_VARYING: 2,
}

__all__ = ["MarketDoTPFNBaseline", "AblatedDoTPFNBaseline"]

_ABLATE_KEYS = (
    "intervention_target",
    "intervention_type",
    "intervention_value",
    "intervention_time_start",
    "intervention_time_end",
)


class MarketDoTPFNBaseline:
    """Wrap a ``continuous_do_over_time_pfn_best.pt`` checkpoint."""

    name = "market-dotpfn"
    _ablate = False

    def __init__(self, checkpoint: str, device: str = "cpu"):
        from dotime_market.models.continuous_model import ContinuousDoOverTimePFN

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        if cfg.get("arch") == "recurrent":
            # Gate-A backbone: same batch contract and head API, so the
            # conversion/predict paths below need no changes.
            from dotime_market.models.recurrent_model import RecurrentDoTPFN

            self.model = RecurrentDoTPFN(
                n_max=cfg["n_max"], embed_size=cfg["embed_size"],
                n_layers=cfg["n_layers"], tau_levels=cfg.get("tau_levels"),
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(device).eval()
            self.device = device
            self.n_max = int(cfg["n_max"])
            return
        self.model = ContinuousDoOverTimePFN(
            n_max=cfg["n_max"],
            embed_size=cfg["embed_size"],
            n_encoder_layers=cfg["n_encoder_layers"],
            n_cross_attn_heads=cfg["n_cross_attn_heads"],
            encoder_backend=cfg.get("encoder_backend", "transformer"),
            context_window=cfg.get("context_window", 128),
            n_mixer_layers=cfg.get("n_mixer_layers", 1),
            num_time_frequencies=cfg.get("num_time_frequencies", 64),
            time_min_freq=cfg.get("time_min_freq", 0.01),
            time_max_freq=cfg.get("time_max_freq", 10.0),
            tau_levels=cfg.get("tau_levels"),
            head_type=cfg.get("head_type", "quantile"),
            n_buckets=cfg.get("n_buckets", 1000),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.n_max = int(cfg["n_max"])

    # ------------------------------------------------------------ conversion

    def _episode_to_batch(self, episode: Episode, q: int) -> dict:
        """One query -> one batch-size-1 model batch (regular dt=1 schedule)."""
        T, n_vars = episode.x_obs.shape
        onset = min(episode.intervention.times)
        end = max(episode.intervention.times) + 1
        span = float(max(T - 1, 1))

        x_obs = episode.x_obs.clone()
        x_obs[onset:] = 0.0  # causal masking, as in generate_sample
        x_pad = torch.zeros(T, self.n_max)
        x_pad[:, : min(n_vars, self.n_max)] = x_obs[:, : self.n_max]

        variable_mask = torch.zeros(self.n_max)
        variable_mask[:n_vars] = 1.0
        for v in range(n_vars):
            if bool((episode.x_obs[:, v] == 0).all()):
                variable_mask[v] = 0.0  # hidden (zeroed) columns are padding

        value = float(episode.intervention.values)
        batch = {
            "X_obs": x_pad.unsqueeze(0),
            "variable_mask": variable_mask.unsqueeze(0),
            "times": torch.arange(T, dtype=torch.float32).unsqueeze(0),
            "dts": torch.ones(T - 1, dtype=torch.float32).unsqueeze(0),
            "int_onset_idx": torch.tensor([onset], dtype=torch.long),
            "t_int_start": torch.tensor([float(onset)], dtype=torch.float32),
            "t_int_end": torch.tensor([float(end)], dtype=torch.float32),
            "intervention_target": torch.tensor(
                [episode.intervention.targets[0]], dtype=torch.long
            ),
            # Mixer kind index matches _INT_KIND_ORDER = (HARD, SOFT, TIME_VARYING).
            "intervention_type": torch.tensor(
                [_KIND_INDEX[episode.intervention.intervention_type]], dtype=torch.long
            ),
            "intervention_value": torch.tensor([value], dtype=torch.float32),
            "intervention_time_start": torch.tensor([onset / span], dtype=torch.float32),
            "intervention_time_end": torch.tensor([end / span], dtype=torch.float32),
            "query_target": torch.tensor([int(episode.query_target[q])], dtype=torch.long),
            "query_time": torch.tensor([float(episode.query_time[q])], dtype=torch.float32),
            "Y_true": torch.tensor([0.0]),  # placeholder; only *_norm stats are used
        }
        return batch

    # -------------------------------------------------------------- predict

    @property
    def tau_levels(self) -> list:
        return list(self.model.quantile_head.tau_levels)

    @torch.no_grad()
    def predict_quantiles(self, episode: Episode) -> torch.Tensor:
        """Raw-scale quantile predictions, shape ``(n_queries, n_tau)``.

        The quantile head's forward output IS the per-tau quantile matrix;
        each row is de-normalized with the same per-episode stats as the
        mean prediction. Used by the counterfactual-calibration study
        (coverage / interval width) — the axis where a distributional head
        differs from point-estimate impact models.
        """
        rows = []
        for q in range(episode.query_target.numel()):
            batch = self._episode_to_batch(episode, q)
            batch = normalize_batch(batch, target_key="Y_true")
            if self._ablate:
                for key in _ABLATE_KEYS:
                    batch[key] = torch.zeros_like(batch[key])
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            quantiles = self.model(batch).reshape(-1)  # (n_tau,)
            qt = int(batch["query_target"][0])
            mean = float(batch["_norm_means"][0, qt])
            std = float(batch["_norm_stds"][0, qt])
            rows.append(quantiles.cpu() * std + mean)
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def predict(self, episode: Episode) -> torch.Tensor:
        preds = []
        for q in range(episode.query_target.numel()):
            batch = self._episode_to_batch(episode, q)
            batch = normalize_batch(batch, target_key="Y_true")
            if self._ablate:
                for key in _ABLATE_KEYS:
                    batch[key] = torch.zeros_like(batch[key])
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            output = self.model(batch)
            pred_norm = self.model.head.predict_mean(output).reshape(-1)[0]
            qt = int(batch["query_target"][0])
            mean = batch["_norm_means"][0, qt]
            std = batch["_norm_stds"][0, qt]
            preds.append(float(pred_norm) * float(std) + float(mean))
        return torch.tensor(preds, dtype=torch.float32)


class AblatedDoTPFNBaseline(MarketDoTPFNBaseline):
    """Predictive twin: do-information zeroed at predict time (use with a
    checkpoint trained under ``ablate_intervention=True``)."""

    name = "ablated-dotpfn"
    _ablate = True


register("market-dotpfn")(MarketDoTPFNBaseline)
register("ablated-dotpfn")(AblatedDoTPFNBaseline)
