"""MarketDoTime samples -> in-memory dotime BenchmarkSuite.

Bridges Layer 2 (the market prior) to Layer 1 (the stock evaluation harness):
``dotime.evaluation.evaluate(baseline, suite)`` runs unchanged on the suites
built here, exactly as it does on released suites — no file round-trip.

Conventions
-----------
- Canonical variable order (from ``ContinuousExtendedPrior``): A (flow) = 0,
  U (hidden core, zeroed) = 1, Y (price) = 2. The hidden column stays in the
  tensors (baselines see zeros — U is *unobserved*, that is the point).
- Episodes are built on a regular ``dt = 1`` schedule, so time indices and
  absolute times coincide and ``InterventionSpec.times`` (discrete indices)
  is exact.
- Queries are placed on the *outcome* variable at post-onset times
  (``y_true`` re-read from the sample's full ``X_int``), because the
  market-impact estimand is Y under ``do(A)``. Training keeps
  ``MarketDoTime``'s own mixed query sampling; this builder is for
  evaluation suites.
- ``structure`` is labelled ``"market_core_reaction"`` — deliberately not a
  stock label: structure-gated baselines (BackDoorOLS, IV2SLS) fall back to
  their naive behaviour, which is correct here because the confounder is
  hidden and no adjustment set exists in the observables.
- Per-episode ground truth the model never sees (gamma, core_share, lambda,
  analytic naive/do slopes and confounding bias) goes into
  ``Episode.metadata`` for oracle baselines and diagnostics.
"""

from __future__ import annotations

import numpy as np
import torch

from dotime import InterventionSpec, InterventionType
from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

from dotime_market.prior.market_scm import MarketDoTime, N_VARS

__all__ = ["build_market_suite", "market_episode"]

_FLOW_CANON, _CORE_CANON, _PRICE_CANON = 0, 1, 2
_STRUCTURE = "market_core_reaction"


def market_episode(
    sample: dict,
    prior: MarketDoTime,
    scm_id: int,
    rng: np.random.RandomState,
    n_queries: int = 1,
) -> Episode:
    """Convert one ``generate_sample()`` dict into a dotime :class:`Episode`."""
    x_obs = sample["X_obs_full"][:, :N_VARS].clone()
    x_int = sample["X_int"][:, :N_VARS].clone()
    T = x_obs.shape[0]
    times = sample["times"]
    span = float((times[-1] - times[0]).item())

    onset = int(sample["int_onset_idx"])
    t_int_end = float(sample["t_int_end"])
    end = int((times < t_int_end).sum())  # first index at/after the window end

    intervention = InterventionSpec(
        targets=[_FLOW_CANON],
        times=list(range(onset, max(end, onset + 1))),
        intervention_type=InterventionType.HARD,
        values=float(sample["intervention_value"]),
    )

    # Outcome queries at distinct post-onset observation times.
    hi = max(onset + 1, T - 1)
    q_idx = rng.randint(onset, hi + 1, size=n_queries)
    query_target = torch.full((n_queries,), _PRICE_CANON, dtype=torch.long)
    query_time = ((times[q_idx] - times[0]) / max(span, 1e-6)).to(torch.float32)
    y_true = x_int[q_idx, _PRICE_CANON].to(torch.float32)

    conf, coup = prior.last_confounder, prior.last_coupling
    metadata = {
        "query_idx": [int(i) for i in q_idx],
        "y_obs": [float(v) for v in x_obs[q_idx, _PRICE_CANON]],
        "core_share": conf.core_share,
        "coupling_gamma": coup.gamma,
        "impact_lambda": coup.impact_lambda,
        "theta_price": coup.theta_price,
        "sigma_price": coup.sigma_price,
        "theta_flow": conf.theta_flow,
        "sigma_flow": conf.sigma_flow,
        "theta_core": conf.theta_core,
        "sigma_core": conf.sigma_core,
        "core_weight": conf.core_weight,
        "naive_slope_ana": coup.naive_slope(conf),
        "do_slope_ana": coup.do_slope(),
        "confounding_bias_ana": coup.confounding_bias(conf),
    }
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=intervention,
        y_true=y_true,
        query_target=query_target,
        query_time=query_time,
        structure=_STRUCTURE,
        scm_id=scm_id,
        metadata=metadata,
    )


def build_market_suite(
    prior: MarketDoTime,
    n_episodes: int,
    T: int | None = None,
    n_queries: int = 1,
    seed: int = 0,
    name: str = "market-synth",
) -> BenchmarkSuite:
    """Draw ``n_episodes`` from ``prior`` and package them as a suite.

    The prior must use a regular ``dt = 1`` schedule (the ``MarketDoTime``
    default) so discrete intervention indices are exact.
    """
    if prior.schedule != "regular" or prior.dt != 1.0:
        raise ValueError(
            f"build_market_suite requires a regular dt=1 schedule, got "
            f"schedule={prior.schedule!r}, dt={prior.dt}"
        )
    rng = np.random.RandomState(seed)
    episodes = [
        market_episode(
            prior.generate_sample(T=T),
            prior,
            scm_id=i,
            rng=rng,
            n_queries=n_queries,
        )
        for i in range(n_episodes)
    ]
    meta = SuiteMetadata(
        name=name,
        version="0.0.1",
        zenodo_record_id="",
        doi="",
        description="In-memory synthetic market-impact suite (core/reaction confounding).",
        n_episodes=n_episodes,
        structures=(_STRUCTURE,),
    )
    return BenchmarkSuite(meta, episodes)
