"""Continuous-time on-the-fly temporal intervention dataloader.

Analogue of :class:`dotime.data.TemporalInterventionDataLoader (released layout)`
that pulls samples from :class:`ContinuousExtendedPrior` instead of
``ExtendedCausalTimePrior``.

Batch contract
--------------
Every field returned by the discrete loader is still present with the
same semantics, plus two new tensors:

- ``times``: ``(B, T)`` float tensor of absolute observation times.
- ``dts``: ``(B, T - 1)`` float tensor of inter-observation gaps.

``t_int_start``, ``t_int_end``, ``t_query`` are in the same absolute
units as ``times``, so downstream encoders can compute ``times -
t_int_start`` directly.

Normalisation
-------------
The usual :func:`dotime.normalization.normalize_batch` is applied
so that ``X_obs_norm`` / ``Y_true_norm`` are available alongside the
raw ``X_obs`` / ``Y_true`` tensors.  It does not touch ``times`` or
``dts`` (they don't share the per-variable scale of the trajectory
tensors), so no extra changes are needed there.
"""

from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Dict, Iterator

import torch

from dotime.normalization import normalize_batch
from dotime.continuous.extended_prior import ContinuousExtendedPrior
from dotime.continuous.random_sampler import RandomContinuousExtendedPrior


class ContinuousTemporalInterventionDataLoader:
    """Infinite dataloader for continuous-time causal PFN training.

    Parameters
    ----------
    num_steps : int
        Number of batches per iteration over the loader.
    batch_size : int
        Batch size.
    tscm_structure : str
        Named :class:`TSCMStructure` value (``back_door``, ``front_door``, ...).
    schedule : {"regular", "jittered", "exponential"}
        Observation schedule family; see :mod:`time_schedule`.
    pair_mode : {"counterfactual", "interventional"}
        Paired-sample semantics.  The workshop paper defaults to
        ``counterfactual`` (shared noise).
    t_range : tuple of int
        Uniform prior on the number of observations per trajectory.
    dt, jitter, exp_rate : forwarded to :class:`ContinuousExtendedPrior`.
    n_max : int
        Variable-axis padding (should match the model's ``n_max``).
    seed : int
        Base seed for the prior's RNG.
    normalize : bool
        Apply per-variable z-score normalisation to ``X_obs`` and
        ``Y_true`` (producing ``X_obs_norm`` / ``Y_true_norm``).
    device : str
        Device to move the returned batch tensors to.
    prefetch : int
        Background-prefetch queue depth.  ``0`` disables prefetch.
    target_key : str
        Which field of the raw batch to use as the regression target
        (typically ``"Y_true"`` or ``"Y_causal_effect"``).
    n_queries : int
        Number of (variable, time) query points per trajectory.
    query_mode : {"single", "all_pairs"}
        See :meth:`ContinuousExtendedPrior.generate_sample`.
    theta_range, sigma_range, weight_scale, intervention_value_scale, intervention_window_frac :
        Forwarded to the prior.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        tscm_structure: str = "back_door",
        schedule: str = "regular",
        pair_mode: str = "counterfactual",
        t_range: tuple = (50, 200),
        dt: float = 1.0,
        jitter: float = 0.3,
        exp_rate: float = 1.0,
        n_max: int = 41,
        seed: int = 42,
        normalize: bool = True,
        device: str = "cpu",
        prefetch: int = 0,
        target_key: str = "Y_true",
        n_queries: int = 1,
        query_mode: str = "single",
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        intervention_value_scale: float = 2.0,
        intervention_window_frac: tuple = (0.1, 0.3),
        prior_mode: str = "tscm",
        n_min_prior: int = 3,
        n_max_prior: int = 10,
        edge_prob: float = 0.3,
        hidden_prob: float = 0.0,
        regime_prob: float = 0.0,
        regime_count_range: tuple = (2, 3),
        sticky_alpha: float = 9.0,
        other_alpha: float = 0.5,
        mechanism_kind: str = "linear",
        p_neural: float = 0.0,
        neural_hidden_dim: int = 8,
        neural_out_scale_range: tuple = (0.5, 2.0),
        num_substeps: int = 1,
        p_no_context: float = 0.0,
        vectorize: bool = False,
        market_coupling_gamma=(0.0, 2.0),
        market_core_share=(0.3, 0.9),
        market_impact_lambda=(0.25, 1.0),
        market_coupling_gamma_power: float = 1.0,
        market_jump_rate=(0.0, 0.0),
        market_jump_scale_mult: float = 3.0,
        market_theta_price_range=None,
        market_impact_lambda_log_range=None,
        market_perm_prob: float = 0.0,
        market_hawkes_rate=(0.0, 0.0),
        intervention_kind_probs: tuple = (1.0, 0.0, 0.0),
        intervention_source: str = "prior",
        soft_shift_scale: float = 1.0,
        time_varying_profile: str = "random",
    ) -> None:
        """Construct the loader.

        ``prior_mode`` switches between a fixed named TSCM structure
        (default, original behaviour) and a random-graph prior that
        samples a fresh DAG per trajectory:

        - ``"tscm"``   : use :class:`ContinuousExtendedPrior` with
          ``tscm_structure``.
        - ``"random"`` : use :class:`RandomContinuousExtendedPrior` with
          ``n_min_prior`` / ``n_max_prior`` / ``edge_prob``.
        - ``"market"`` : use :class:`dotime_market.prior.MarketDoTime` with
          the ``market_*`` knobs (each a scalar to pin or a ``(lo, hi)``
          range to sample per episode). [dotime-market extension — the only
          change to this vendored file.]

        All other arguments are interpreted the same way in all modes.
        """
        if prior_mode not in ("tscm", "random", "market"):
            raise ValueError(f"invalid prior_mode: {prior_mode!r}")

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.prefetch = prefetch
        self.target_key = target_key
        self.n_queries = n_queries
        self.query_mode = query_mode
        self.prior_mode = prior_mode

        common_kwargs = dict(
            n_max=n_max,
            t_range=t_range,
            schedule=schedule,
            dt=dt,
            jitter=jitter,
            exp_rate=exp_rate,
            pair_mode=pair_mode,
            intervention_value_scale=intervention_value_scale,
            intervention_window_frac=intervention_window_frac,
            # Regression fix (2026-07-31): these four were accepted by
            # train_continuous and logged, but never forwarded — every prior
            # ran with hard-only interventions regardless of the CLI flag.
            intervention_kind_probs=intervention_kind_probs,
            intervention_source=intervention_source,
            soft_shift_scale=soft_shift_scale,
            time_varying_profile=time_varying_profile,
            theta_range=theta_range,
            sigma_range=sigma_range,
            weight_scale=weight_scale,
            num_substeps=num_substeps,
            p_no_context=p_no_context,
            vectorize=vectorize,
            seed=seed,
        )

        # Released dotime (<= 0.1.2) predates the vectorize kwarg of the
        # development tree; drop it when the installed parent cannot accept
        # it so the same code runs against both.
        import inspect as _inspect

        from dotime.continuous.extended_prior import ContinuousExtendedPrior as _CEP

        if "vectorize" not in _inspect.signature(_CEP.__init__).parameters:
            common_kwargs.pop("vectorize", None)

        if prior_mode == "market":
            from dotime_market.prior.market_scm import MarketDoTime

            self.prior = MarketDoTime(
                coupling_gamma=market_coupling_gamma,
                core_share=market_core_share,
                impact_lambda=market_impact_lambda,
                coupling_gamma_power=market_coupling_gamma_power,
                jump_rate=market_jump_rate,
                jump_scale_mult=market_jump_scale_mult,
                theta_price_range=market_theta_price_range,
                impact_lambda_log_range=market_impact_lambda_log_range,
                perm_prob=market_perm_prob,
                hawkes_rate=market_hawkes_rate,
                **common_kwargs,
            )
        elif prior_mode == "random":
            self.prior = RandomContinuousExtendedPrior(
                n_min=n_min_prior,
                n_max_prior=n_max_prior,
                edge_prob=edge_prob,
                hidden_prob=hidden_prob,
                regime_prob=regime_prob,
                regime_count_range=regime_count_range,
                sticky_alpha=sticky_alpha,
                other_alpha=other_alpha,
                mechanism_kind=mechanism_kind,
                p_neural=p_neural,
                neural_hidden_dim=neural_hidden_dim,
                neural_out_scale_range=neural_out_scale_range,
                **common_kwargs,
            )
        else:
            self.prior = ContinuousExtendedPrior(
                tscm_structure=tscm_structure,
                **common_kwargs,
            )

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.prefetch > 0:
            yield from self._iter_prefetch()
        else:
            for _ in range(self.num_steps):
                yield self._generate_batch()

    def _iter_prefetch(self) -> Iterator[Dict[str, torch.Tensor]]:
        queue: Queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def _fill():
            for _ in range(self.num_steps):
                queue.put(self._generate_batch())
            queue.put(sentinel)

        thread = Thread(target=_fill, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item
        thread.join(timeout=5)

    def _generate_batch(self) -> Dict[str, torch.Tensor]:
        batch = self.prior.generate_batch(
            batch_size=self.batch_size,
            n_queries=self.n_queries,
            query_mode=self.query_mode,
        )
        if self.normalize:
            batch = normalize_batch(batch, target_key=self.target_key)
        if self.device != "cpu":
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        return batch
