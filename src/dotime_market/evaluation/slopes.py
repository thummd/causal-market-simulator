"""Naive vs interventional slope measurement — the Phase-0/Phase-1 gap protocol.

Factored out of ``scripts/00_stock_gap_check.py`` (same defaults, so numbers
stay comparable) for reuse by the bias-vs-gamma sweep
(``scripts/01_bias_vs_gamma.py``) and the market-prior CI gates.

- :func:`naive_ols_slope` — contemporaneous OLS slope of ``Y_t`` on ``A_t``
  over a long observational trajectory: the naive predictive/conditional
  estimate. Converges to ``Cov(Y, A) / Var(A)`` at stationarity
  (analytically :meth:`InformationalCoupling.naive_slope`).
- :func:`interventional_slope` — empirical ``dE[Y | do(A = a)] / da`` from
  hard ``do(A = +v)`` vs ``do(A = -v)`` clamp runs, averaging ``Y`` past a
  settling offset. Converges to ``impact_lambda / theta_price`` for the
  market prior (analytically :meth:`InformationalCoupling.do_slope`).
"""

from __future__ import annotations

import torch

from dotime.continuous import (
    ContinuousIntervention,
    InterventionKind,
    regular_schedule,
)

__all__ = ["naive_ols_slope", "interventional_slope"]


def naive_ols_slope(
    x: torch.Tensor,
    a_idx: int,
    y_idx: int,
    burn: int = 50,
) -> float:
    """OLS slope of ``x[:, y_idx]`` on ``x[:, a_idx]``, discarding ``burn``
    initial observations (transient toward stationarity)."""
    a = x[burn:, a_idx]
    y = x[burn:, y_idx]
    a = a - a.mean()
    y = y - y.mean()
    return float((a * y).sum() / (a * a).sum().clamp_min(1e-8))


def interventional_slope(
    scm,
    a_idx: int,
    y_idx: int,
    generator: torch.Generator,
    t_len: int = 300,
    dt: float = 1.0,
    window: tuple = (150.0, 300.0),
    settle: float = 20.0,
    value: float = 2.0,
    n_reps: int = 6,
    num_substeps: int = 2,
) -> float:
    """Empirical interventional slope from paired hard-clamp arms.

    Runs ``n_reps`` independent-noise simulations per arm with
    ``do(A = +value)`` / ``do(A = -value)`` active on ``window`` (absolute
    time), averages ``Y`` over observations at least ``settle`` time units
    past onset, and returns the centred difference
    ``(mean_+ - mean_-) / (2 * value)``.
    """
    times, dts = regular_schedule(T=t_len, dt=dt)
    post = times >= (window[0] + settle)
    means = {}
    for sign in (+1.0, -1.0):
        vals = []
        for _ in range(n_reps):
            iv = ContinuousIntervention(
                target=a_idx,
                t_start=window[0],
                t_end=window[1],
                kind=InterventionKind.HARD,
                value=sign * value,
            )
            _, traj = scm.simulate(
                times, dts, intervention=iv, generator=generator, num_substeps=num_substeps
            )
            vals.append(float(traj[post, y_idx].mean()))
        means[sign] = sum(vals) / len(vals)
    return (means[1.0] - means[-1.0]) / (2.0 * value)
