"""Phase 1: MarketDoTime produces valid model-ready episodes, the hidden core
is masked, knobs pin per-episode parameters, and — the CI gate from the
architecture doc — the market prior opens a controllable confounding gap that
grows with coupling_gamma.
"""

import pytest
import torch

from dotime.continuous import regular_schedule
from dotime_market.evaluation.slopes import naive_ols_slope
from dotime_market.prior.market_scm import (
    CORE_TOPO,
    FLOW_TOPO,
    PRICE_TOPO,
    MarketContinuousSCMSampler,
    MarketDoTime,
)

# ------------------------------------------------------------- episode shape


def test_generate_sample_is_model_ready():
    prior = MarketDoTime(coupling_gamma=1.0, core_share=0.7, seed=0)
    s = prior.generate_sample(T=80)

    T, n_max = 80, prior.n_max
    assert s["X_obs"].shape == (T, n_max)
    assert s["X_int"].shape == (T, n_max)
    assert s["times"].shape == (T,)
    assert s["dts"].shape == (T - 1,)
    assert int(s["num_vars"]) == 3

    # Canonical order: A -> 0, U (middle) -> 1, Y -> 2.
    assert int(s["intervention_target"]) == 0
    # The hidden core is masked out of the tokens and the variable mask.
    mask = s["variable_mask"]
    assert mask[0] == 1.0 and mask[1] == 0.0 and mask[2] == 1.0
    assert mask[3:].sum() == 0.0
    assert torch.all(s["X_int"][:, 1] == 0.0)
    assert torch.all(s["X_obs"][:, 1] == 0.0)
    # Observable variables are simulated (nonzero somewhere).
    assert s["X_int"][:, 0].abs().sum() > 0
    assert s["X_int"][:, 2].abs().sum() > 0


def test_reproducible_across_instances():
    s1 = MarketDoTime(seed=7).generate_sample(T=60)
    s2 = MarketDoTime(seed=7).generate_sample(T=60)
    for key in s1:
        assert torch.equal(s1[key], s2[key]), key


def test_scalar_knobs_pin_episode_parameters():
    prior = MarketDoTime(coupling_gamma=1.25, core_share=0.7, impact_lambda=0.5, seed=0)
    prior.generate_sample(T=60)
    assert prior.last_confounder.core_share == pytest.approx(0.7)
    assert prior.last_coupling.gamma == pytest.approx(1.25)
    assert prior.last_coupling.impact_lambda == pytest.approx(0.5)
    # Range knobs actually vary per episode.
    prior2 = MarketDoTime(coupling_gamma=(0.0, 2.0), seed=0)
    prior2.generate_sample(T=60)
    g1 = prior2.last_coupling.gamma
    prior2.generate_sample(T=60)
    assert prior2.last_coupling.gamma != g1


def test_coupling_gamma_power_skews_high():
    """power=3 must shift gamma draws toward the top of the range
    (analytic mean of lo + (hi-lo)*U^(1/3) is lo + 0.75*(hi-lo));
    power=1 stays uniform; draws remain inside the range and reproducible."""
    import numpy as np

    def draws(power, seed=3, n=200):
        s = MarketContinuousSCMSampler(
            coupling_gamma=(0.0, 2.0), coupling_gamma_power=power, seed=seed
        )
        return np.array([s.sample()[2].gamma for _ in range(n)])

    g_uniform, g_skewed = draws(1.0), draws(3.0)
    assert g_skewed.min() >= 0.0 and g_skewed.max() <= 2.0
    assert abs(g_uniform.mean() - 1.0) < 0.15
    assert abs(g_skewed.mean() - 1.5) < 0.15
    assert np.array_equal(draws(3.0), draws(3.0))  # same seed, same draws

    with pytest.raises(ValueError, match="coupling_gamma_power"):
        MarketContinuousSCMSampler(coupling_gamma_power=0.0)


def test_sampler_role_accessors():
    sampler = MarketContinuousSCMSampler(seed=0)
    assert sampler.get_intervention_target() == FLOW_TOPO
    assert sampler.get_outcome_var() == PRICE_TOPO
    assert sampler.get_hidden_vars() == [CORE_TOPO]
    scm, conf, coup = sampler.sample()
    assert len(scm.mechanisms) == 3


# ------------------------------------------------- the gap gate (CI version)


def _mean_empirical_bias(gamma: float, n_scms: int = 10, t_obs: int = 500) -> tuple:
    """Mean (empirical bias, analytic bias) over sampled market SCMs.

    Empirical bias = naive OLS slope on an observational path minus the
    *analytic* gamma=0 naive slope of the same episode (isolating the
    confounding contribution from the filtering baseline).

    num_substeps=4 keeps Euler-Maruyama error small even at theta=2 (the
    hyperprior's upper end); at coarser steps the discretization offset is
    comparable to the gamma=0 noise floor asserted below.
    """
    import dataclasses

    sampler = MarketContinuousSCMSampler(core_share=0.7, coupling_gamma=gamma, seed=123)
    gen = torch.Generator().manual_seed(7)
    emp, ana = [], []
    for _ in range(n_scms):
        scm, conf, coup = sampler.sample()
        times, dts = regular_schedule(T=t_obs, dt=1.0)
        _, x = scm.simulate(times, dts, generator=gen, num_substeps=4)
        naive_emp = naive_ols_slope(x, a_idx=FLOW_TOPO, y_idx=PRICE_TOPO)
        gamma0 = dataclasses.replace(coup, gamma=0.0)
        emp.append(naive_emp - gamma0.naive_slope(conf))
        ana.append(coup.confounding_bias(conf))
    t = torch.tensor
    return float(t(emp).mean()), float(t(ana).mean())


THRESHOLD = 0.05  # same scale as the Phase-0 stock gate


def test_market_confounding_gap_gate():
    """The PoC precondition on the market prior (architecture doc, Phase 1):
    at gamma=1, core_share=0.7 the confounding bias must be clearly
    measurable — and it must agree with the analytic prediction."""
    emp, ana = _mean_empirical_bias(gamma=1.0)
    assert emp > THRESHOLD, f"market-prior confounding bias {emp:.4f} <= {THRESHOLD}"
    assert ana > THRESHOLD
    assert emp == pytest.approx(ana, rel=0.5)


def test_gap_is_controlled_by_gamma():
    """The knob must move the gap: zero at gamma=0, increasing in gamma."""
    emp0, ana0 = _mean_empirical_bias(gamma=0.0, n_scms=8)
    emp2, ana2 = _mean_empirical_bias(gamma=2.0, n_scms=8)
    assert ana0 == 0.0
    assert abs(emp0) < 0.1  # only estimation noise at gamma=0
    assert emp2 > emp0 + 0.1
    assert ana2 > 0.2
