"""Phase 1 unit tests: InformationalCoupling analytics vs dotime's integrator.

Checks the closed-form slopes (module docstring of coupling.py) against
simulation, and the two exact reductions: gamma = 0 drops the U -> Y edge
entirely; confounding bias is zero at gamma = 0 and at core_share = 0.
"""

import dataclasses
import math

import pytest
import torch

from dotime.continuous import ContinuousSCM, regular_schedule
from dotime_market.evaluation.slopes import interventional_slope, naive_ols_slope
from dotime_market.prior.confounding import CoreReactionConfounder
from dotime_market.prior.coupling import (
    InformationalCoupling,
    sample_informational_coupling,
)

CONF = CoreReactionConfounder(
    core_share=0.7, theta_core=0.6, sigma_core=0.4, theta_flow=0.9, sigma_flow=0.3
)
COUP = InformationalCoupling(gamma=1.5, impact_lambda=0.5, theta_price=0.8, sigma_price=0.3)


def _market_scm(conf: CoreReactionConfounder, coup: InformationalCoupling) -> ContinuousSCM:
    core, flow = conf.mechanisms(0, 1)
    price = coup.mechanism(0, 1, 2)
    return ContinuousSCM([core, flow, price])


# ---------------------------------------------------------------- analytic


def test_gamma_zero_drops_informational_edge():
    coup = dataclasses.replace(COUP, gamma=0.0)
    mech = coup.mechanism(0, 1, 2)
    assert mech.parents == (1,)  # flow only, no core parent
    assert math.isclose(float(mech.parent_weights[0]), 0.5, rel_tol=1e-6)
    assert coup.confounding_bias(CONF) == 0.0


def test_lambda_and_gamma_zero_reduce_to_stock_root():
    coup = dataclasses.replace(COUP, gamma=0.0, impact_lambda=0.0)
    mech = coup.mechanism(0, 1, 2)
    assert mech.parents == ()
    assert mech.parent_weights.numel() == 0


def test_confounding_bias_zero_without_core_share():
    conf = dataclasses.replace(CONF, core_share=0.0)
    assert COUP.confounding_bias(conf) == 0.0
    # And the naive slope collapses to the pure filtering value.
    gamma0 = dataclasses.replace(COUP, gamma=0.0)
    assert COUP.naive_slope(conf) == pytest.approx(gamma0.naive_slope(conf), rel=1e-12)


def test_confounding_bias_linear_in_gamma():
    b1 = dataclasses.replace(COUP, gamma=0.7).confounding_bias(CONF)
    b2 = dataclasses.replace(COUP, gamma=1.4).confounding_bias(CONF)
    assert b1 > 0.0
    assert b2 == pytest.approx(2.0 * b1, rel=1e-9)


def test_bias_is_naive_slope_excess_over_gamma_zero():
    gamma0 = dataclasses.replace(COUP, gamma=0.0)
    excess = COUP.naive_slope(CONF) - gamma0.naive_slope(CONF)
    assert COUP.confounding_bias(CONF) == pytest.approx(excess, rel=1e-9)


def test_do_slope_steady_state():
    assert COUP.do_slope() == pytest.approx(0.5 / 0.8)


# -------------------------------------------------------------- validation


# theta_price == 0.0 is legal since the rung-2 unit-root carve-out
# (test_perm_impact.py); only NEGATIVE theta is invalid now.
@pytest.mark.parametrize("kwargs", [{"theta_price": -0.5}, {"sigma_price": -0.1}])
def test_invalid_price_params_raise(kwargs):
    with pytest.raises(ValueError):
        InformationalCoupling(gamma=1.0, impact_lambda=0.5, **kwargs)


def test_mechanism_requires_topological_order():
    with pytest.raises(ValueError, match="topological"):
        COUP.mechanism(0, 2, 1)


# ----------------------------------------------------------------- sampler


def test_sampler_reproducible_and_pinnable():
    c1 = sample_informational_coupling(generator=torch.Generator().manual_seed(4))
    c2 = sample_informational_coupling(generator=torch.Generator().manual_seed(4))
    assert c1 == c2
    pinned = sample_informational_coupling(
        gamma_range=(1.25, 1.25), generator=torch.Generator().manual_seed(0)
    )
    assert pinned.gamma == pytest.approx(1.25)


def test_sampler_invalid_ranges_raise():
    with pytest.raises(ValueError):
        sample_informational_coupling(gamma_range=(2.0, 1.0))
    with pytest.raises(ValueError):
        sample_informational_coupling(theta_range=(0.0, 1.0))


# -------------------------------------------------------------- simulation


def test_simulated_naive_slope_matches_analytic():
    scm = _market_scm(CONF, COUP)
    times, dts = regular_schedule(T=6000, dt=0.5)
    gen = torch.Generator().manual_seed(13)
    _, x = scm.simulate(times, dts, generator=gen, num_substeps=4)
    emp = naive_ols_slope(x, a_idx=1, y_idx=2, burn=500)
    assert emp == pytest.approx(COUP.naive_slope(CONF), rel=0.2)


def test_simulated_do_slope_matches_lambda_over_theta():
    scm = _market_scm(CONF, COUP)
    gen = torch.Generator().manual_seed(21)
    emp = interventional_slope(scm, a_idx=1, y_idx=2, generator=gen, n_reps=6)
    assert emp == pytest.approx(COUP.do_slope(), rel=0.2)
    # The interventional slope must NOT contain the confounding bias.
    assert abs(emp - COUP.naive_slope(CONF)) > COUP.confounding_bias(CONF) / 2
