"""Rung-2 prior fix: unit-root (permanent-impact) carve-out.

Three guarantees, mirroring the jump-layer test discipline:

1. Reduction: ``perm_prob=0`` consumes no randomness — seeded episode
   streams are bit-identical to the legacy prior (the CI gate for prior
   correctness).
2. Carve-out semantics: ``perm_prob=1`` pins ``theta_price == 0`` and
   only ``theta_price`` — every other parameter draw is untouched.
3. Accrual: under a sustained input, the integrator price's mean response
   grows ~linearly with duration (no saturation), unlike the OU price —
   the mechanism-level property Gate B demands.
"""

import pytest
import torch

from dotime_market.prior.coupling import (
    InformationalCoupling,
    IntegratorMechanism,
    sample_informational_coupling,
)
from dotime_market.prior.market_scm import MarketContinuousSCMSampler, MarketDoTime


def test_perm_prob_zero_is_bit_exact():
    kw = dict(coupling_gamma=1.0, core_share=0.7, seed=99)
    a = MarketDoTime(**kw)
    b = MarketDoTime(perm_prob=0.0, **kw)
    sa, sb = a.generate_sample(), b.generate_sample()
    assert torch.equal(sa["X_obs_full"], sb["X_obs_full"])


def test_perm_prob_one_pins_theta_price_only():
    # Slot alignment holds only for the FIRST episode: the enabled path
    # consumes one extra Bernoulli per episode, shifting later streams
    # (by design — only perm_prob=0 promises bit-exactness, tested above).
    _, conf_b, coup_b = MarketContinuousSCMSampler(seed=5).sample()
    perm = MarketContinuousSCMSampler(perm_prob=1.0, seed=5)
    _, conf_p, coup_p = perm.sample()
    assert coup_p.theta_price == 0.0
    assert coup_p.gamma == coup_b.gamma
    assert coup_p.impact_lambda == coup_b.impact_lambda
    assert coup_p.sigma_price == coup_b.sigma_price
    assert conf_p.theta_core == conf_b.theta_core
    assert conf_p.theta_flow == conf_b.theta_flow
    # Later episodes keep the carve-out pinned.
    for _ in range(4):
        assert perm.sample()[2].theta_price == 0.0


def test_perm_prob_intermediate_mixes():
    sampler = MarketContinuousSCMSampler(perm_prob=0.5, seed=7)
    thetas = [sampler.sample()[2].theta_price for _ in range(40)]
    n_perm = sum(t == 0.0 for t in thetas)
    # Loose two-sided bound: P(outside) < 1e-4 under Binomial(40, 0.5).
    assert 8 <= n_perm <= 32


def test_integrator_mechanism_validation():
    IntegratorMechanism(theta=0.0, sigma=0.3)  # the point of the class
    with pytest.raises(ValueError):
        IntegratorMechanism(theta=-0.1, sigma=0.3)
    with pytest.raises(ValueError):
        InformationalCoupling(gamma=0.0, impact_lambda=0.5, theta_price=-1.0)
    with pytest.raises(ValueError):
        sample_informational_coupling(perm_prob=1.5)


def test_do_slope_undefined_at_unit_root():
    coup = InformationalCoupling(gamma=0.0, impact_lambda=0.5, theta_price=0.0)
    with pytest.raises(ValueError, match="unit-root"):
        coup.do_slope()


def test_integrator_impact_accrues_linearly():
    """EM-roll the price mechanism under a clamped parent: the integrator's
    mean response after 2T must be ~2x its response after T (linear accrual),
    while the OU response ratio is visibly sub-linear (saturation)."""
    lam, a, dt, n = 0.5, 1.0, 0.05, 400
    zero_noise = torch.tensor(0.0)
    dt_t = torch.tensor(dt)
    parents = torch.tensor([a])

    def roll(mech):
        x = torch.tensor(0.0)
        path = []
        for _ in range(n):
            x = mech.step(x, parents, dt_t, zero_noise)
            path.append(float(x))
        return path

    integ = IntegratorMechanism(
        theta=0.0, sigma=0.3, parent_weights=torch.tensor([lam]), parents=(1,)
    )
    p = roll(integ)
    ratio_integ = p[-1] / p[n // 2 - 1]
    assert ratio_integ == pytest.approx(2.0, rel=0.02)

    ou = InformationalCoupling(gamma=0.0, impact_lambda=lam, theta_price=1.0).mechanism(0, 1, 2)
    q = roll(ou)
    ratio_ou = q[-1] / q[n // 2 - 1]
    assert ratio_ou < 1.1  # saturated well before T: the Gate-B failure mode


def test_lambda_log_range_none_is_bit_exact():
    import torch as _t
    kw = dict(coupling_gamma=1.0, core_share=0.7, seed=99)
    a = MarketDoTime(**kw)
    b = MarketDoTime(impact_lambda_log_range=None, **kw)
    sa, sb = a.generate_sample(), b.generate_sample()
    assert _t.equal(sa["X_obs_full"], sb["X_obs_full"])


def test_lambda_log_range_changes_lambda_only():
    base = MarketContinuousSCMSampler(seed=5)
    wide = MarketContinuousSCMSampler(impact_lambda_log_range=(0.02, 1.0), seed=5)
    _, cb, kb = base.sample()
    _, cw, kw_ = wide.sample()
    assert 0.02 <= kw_.impact_lambda <= 1.0
    assert kw_.gamma == kb.gamma and kw_.theta_price == kb.theta_price
    assert cw.theta_core == cb.theta_core
