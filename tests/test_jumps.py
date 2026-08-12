"""Reduction and counterfactual-exactness tests for the jump-diffusion layer."""

import torch

from dotime_market.prior.jumps import JumpDiffusionSCM
from dotime_market.prior.market_scm import MarketContinuousSCMSampler, MarketDoTime

from dotime.continuous.continuous_scm import ContinuousSCM


def test_zero_rate_reduces_to_stock_scm():
    """jump_rate=(0,0) must construct plain ContinuousSCM instances."""
    sampler = MarketContinuousSCMSampler(jump_rate=(0.0, 0.0), seed=0)
    for _ in range(5):
        scm, _, _ = sampler.sample()
        assert type(scm) is ContinuousSCM


def test_positive_rate_constructs_jump_scm():
    sampler = MarketContinuousSCMSampler(jump_rate=(0.2, 0.2), seed=0)
    scm, conf, _ = sampler.sample()
    assert isinstance(scm, JumpDiffusionSCM)
    assert scm.jump_rate > 0
    assert abs(scm.jump_scale - 3.0 * conf.sigma_core) < 1e-9


def test_zero_rate_prior_matches_stock_prior_exactly():
    """Feature off == stock market prior: identical samples, same seed."""
    kw = dict(coupling_gamma=1.0, core_share=0.7, seed=123)
    a = MarketDoTime(**kw)
    b = MarketDoTime(jump_rate=(0.0, 0.0), **kw)
    sa, sb = a.generate_sample(), b.generate_sample()
    assert torch.equal(sa["X_obs_full"], sb["X_obs_full"])
    assert torch.equal(sa["X_int"], sb["X_int"])


def test_counterfactual_pair_shares_jumps():
    """Pre-onset paths of a counterfactual pair must stay identical with jumps on."""
    prior = MarketDoTime(coupling_gamma=1.0, core_share=0.7,
                         jump_rate=(0.5, 0.5), jump_scale_mult=3.0, seed=7)
    for _ in range(5):
        sample = prior.generate_sample()
        onset = int(sample["int_onset_idx"])
        if onset < 3:
            continue
        pre_obs = sample["X_obs_full"][:onset]
        pre_int_full = sample["X_int"]
        assert torch.allclose(pre_obs[:, 2], pre_int_full[:onset, 2], atol=1e-5), \
            "pre-onset price paths must be identical (shared noise incl. jumps)"


def test_jumps_fatten_core_tails():
    """High-rate jumps must visibly fatten the core increment distribution."""

    def max_core_step(jump_rate, n=15):
        sampler = MarketContinuousSCMSampler(jump_rate=jump_rate,
                                             jump_scale_mult=5.0, seed=11)
        m = 0.0
        for _ in range(n):
            scm, _, _ = sampler.sample()
            _, traj = scm.simulate(torch.arange(0, 60, dtype=torch.float32),
                                   torch.ones(59))
            core = traj[:, 0]  # topological order: core first
            m = max(m, float(core[1:].sub(core[:-1]).abs().max()))
        return m

    assert max_core_step((1.0, 1.0)) > 1.5 * max_core_step((0.0, 0.0))


def test_theta_price_range_none_is_bit_exact():
    """theta_price_range=None must reproduce the legacy prior exactly."""
    kw = dict(coupling_gamma=1.0, core_share=0.7, seed=99)
    a = MarketDoTime(**kw)
    b = MarketDoTime(theta_price_range=None, **kw)
    sa, sb = a.generate_sample(), b.generate_sample()
    assert torch.equal(sa["X_obs_full"], sb["X_obs_full"])


def test_theta_price_range_slows_price_only():
    """A slow-theta override changes theta_Y but no other parameter."""
    from dotime_market.prior.market_scm import MarketContinuousSCMSampler
    base = MarketContinuousSCMSampler(seed=5)
    slow = MarketContinuousSCMSampler(theta_price_range=(0.05, 0.2), seed=5)
    _, cb, kb = base.sample()
    _, cs, ks = slow.sample()
    assert ks.theta_price <= 0.2
    assert kb.gamma == ks.gamma and kb.impact_lambda == ks.impact_lambda
    assert cb.theta_core == cs.theta_core and cb.theta_flow == cs.theta_flow
