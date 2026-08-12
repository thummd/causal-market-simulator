"""Phase 1 unit tests: CoreReactionConfounder calibration and simulation.

Two layers of checks:

1. Analytic — the variance-share calibration must round-trip exactly and the
   ``core_share = 0`` case must reduce to a stock parentless OU mechanism.
2. Empirical — a long trajectory through dotime's own Euler–Maruyama
   integrator must reproduce the analytic stationary moments (Var(U), Var(A),
   Cov(A, U)). Tolerances allow for O(theta * dt) EM inflation plus
   finite-sample noise; the effective step is kept small via num_substeps.
"""

import math

import pytest
import torch

from dotime.continuous import ContinuousSCM, regular_schedule
from dotime_market.prior.confounding import (
    CoreReactionConfounder,
    sample_core_reaction_confounder,
)

# ---------------------------------------------------------------- analytic


@pytest.mark.parametrize("share", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_core_share_round_trip(share):
    c = CoreReactionConfounder(
        core_share=share, theta_core=0.8, sigma_core=0.5, theta_flow=1.3, sigma_flow=0.3
    )
    achieved = c.core_driven_var() / c.stationary_flow_var()
    assert math.isclose(achieved, share, abs_tol=1e-12)


def test_core_weight_monotone_in_share():
    shares = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    weights = [CoreReactionConfounder(core_share=s).core_weight for s in shares]
    assert weights[0] == 0.0
    assert all(w1 < w2 for w1, w2 in zip(weights, weights[1:]))


def test_zero_share_reduces_to_stock_root_ou():
    c = CoreReactionConfounder(core_share=0.0, theta_flow=1.7, sigma_flow=0.25)
    core, flow = c.mechanisms(core_idx=0, flow_idx=1)
    assert flow.parents == ()
    assert flow.parent_weights.numel() == 0
    assert flow.theta == 1.7 and flow.sigma == 0.25
    assert core.parents == ()
    assert c.stationary_flow_core_cov() == 0.0


def test_mechanisms_wire_calibrated_weight():
    c = CoreReactionConfounder(core_share=0.6)
    core, flow = c.mechanisms(core_idx=2, flow_idx=5)
    assert flow.parents == (2,)
    assert flow.parent_weights.shape == (1,)
    assert math.isclose(float(flow.parent_weights[0]), c.core_weight, rel_tol=1e-6)


# -------------------------------------------------------------- validation


@pytest.mark.parametrize("share", [-0.1, 1.0, 1.5])
def test_invalid_core_share_raises(share):
    with pytest.raises(ValueError, match="core_share"):
        CoreReactionConfounder(core_share=share)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta_core": 0.0},
        {"sigma_core": -1.0},
        {"theta_flow": 0.0},
        {"sigma_flow": 0.0},
    ],
)
def test_invalid_process_params_raise(kwargs):
    with pytest.raises(ValueError):
        CoreReactionConfounder(core_share=0.5, **kwargs)


def test_mechanisms_require_core_before_flow():
    c = CoreReactionConfounder(core_share=0.5)
    with pytest.raises(ValueError, match="topological"):
        c.mechanisms(core_idx=1, flow_idx=1)
    with pytest.raises(ValueError, match="topological"):
        c.mechanisms(core_idx=3, flow_idx=1)


# ----------------------------------------------------------------- sampler


def test_sampler_reproducible_and_in_range():
    lo_hi = dict(
        core_share_range=(0.3, 0.9), theta_range=(0.5, 2.0), sigma_range=(0.2, 0.6)
    )
    c1 = sample_core_reaction_confounder(**lo_hi, generator=torch.Generator().manual_seed(11))
    c2 = sample_core_reaction_confounder(**lo_hi, generator=torch.Generator().manual_seed(11))
    assert c1 == c2
    assert 0.3 <= c1.core_share <= 0.9
    for theta in (c1.theta_core, c1.theta_flow):
        assert 0.5 <= theta <= 2.0
    for sigma in (c1.sigma_core, c1.sigma_flow):
        assert 0.2 <= sigma <= 0.6


def test_sampler_degenerate_range_pins_share():
    c = sample_core_reaction_confounder(
        core_share_range=(0.7, 0.7), generator=torch.Generator().manual_seed(0)
    )
    assert c.core_share == pytest.approx(0.7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"core_share_range": (-0.1, 0.5)},
        {"core_share_range": (0.5, 1.0)},
        {"core_share_range": (0.8, 0.3)},
        {"theta_range": (0.0, 1.0)},
        {"sigma_range": (0.5, 0.2)},
    ],
)
def test_sampler_invalid_ranges_raise(kwargs):
    with pytest.raises(ValueError):
        sample_core_reaction_confounder(**kwargs)


# -------------------------------------------------------------- simulation


def test_simulated_stationary_moments_match_analytic():
    c = CoreReactionConfounder(
        core_share=0.7, theta_core=0.6, sigma_core=0.4, theta_flow=0.9, sigma_flow=0.3
    )
    core, flow = c.mechanisms(core_idx=0, flow_idx=1)
    scm = ContinuousSCM([core, flow])

    times, dts = regular_schedule(T=6000, dt=0.5)
    gen = torch.Generator().manual_seed(7)
    # num_substeps=4 -> effective EM step 0.125; theta*dt <= 0.1125, so EM
    # inflates the OU variance by <~ 6% — well inside the tolerances below.
    _, x = scm.simulate(times, dts, generator=gen, num_substeps=4)

    burn = 500
    u = x[burn:, 0]
    a = x[burn:, 1]
    var_u = float(u.var(unbiased=True))
    var_a = float(a.var(unbiased=True))
    cov_au = float(((a - a.mean()) * (u - u.mean())).mean())

    assert var_u == pytest.approx(c.stationary_core_var(), rel=0.15)
    assert var_a == pytest.approx(c.stationary_flow_var(), rel=0.15)
    assert cov_au == pytest.approx(c.stationary_flow_core_cov(), rel=0.20)


def test_simulated_share_monotone_in_core_share():
    """The knob must actually move the simulated core-driven share of Var(A).

    corr(A, U)^2 is the observable proxy; analytically it equals
    core_share * theta_flow / (theta_flow + theta_core), so with equal thetas
    it is core_share / 2 and strictly increasing in core_share.
    """
    corrs = []
    for share in (0.1, 0.5, 0.9):
        c = CoreReactionConfounder(core_share=share, theta_core=1.0, theta_flow=1.0)
        core, flow = c.mechanisms(core_idx=0, flow_idx=1)
        scm = ContinuousSCM([core, flow])
        times, dts = regular_schedule(T=4000, dt=0.5)
        gen = torch.Generator().manual_seed(3)
        _, x = scm.simulate(times, dts, generator=gen, num_substeps=4)
        u, a = x[500:, 0], x[500:, 1]
        corr = float(((a - a.mean()) * (u - u.mean())).mean() / (a.std() * u.std()))
        corrs.append(corr**2)
        assert corr**2 == pytest.approx(share / 2.0, abs=0.06)
    assert corrs[0] < corrs[1] < corrs[2]
