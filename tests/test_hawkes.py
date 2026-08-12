"""Rung-3 Hawkes flow-timing layer: reduction, exactness, and clustering.

Same discipline as the jump layer (tests/test_jumps.py): the feature-off
path must be bit-exact with the stock prior, counterfactual pairs must share
the arrival randomness, and the feature-on path must actually produce the
clustered same-sign flow structure it exists to model.
"""

import pytest
import torch

from dotime.continuous import ContinuousSCM

from dotime_market.prior.hawkes import HawkesFlowSCM
from dotime_market.prior.market_scm import MarketContinuousSCMSampler, MarketDoTime


def test_hawkes_off_is_bit_exact():
    kw = dict(coupling_gamma=1.0, core_share=0.7, seed=99)
    a = MarketDoTime(**kw)
    b = MarketDoTime(hawkes_rate=(0.0, 0.0), **kw)
    sa, sb = a.generate_sample(), b.generate_sample()
    assert torch.equal(sa["X_obs_full"], sb["X_obs_full"])
    assert torch.equal(sa["X_int"], sb["X_int"])


def test_hawkes_on_constructs_hawkes_scm():
    on = MarketContinuousSCMSampler(hawkes_rate=(0.2, 0.2), seed=1)
    off = MarketContinuousSCMSampler(seed=1)
    assert isinstance(on.sample()[0], HawkesFlowSCM)
    assert type(off.sample()[0]) is ContinuousSCM


def test_hawkes_validation():
    sampler = MarketContinuousSCMSampler(seed=0)
    _, conf, coup = sampler.sample()
    core, flow = conf.mechanisms(0, 1)
    price = coup.mechanism(0, 1, 2)
    with pytest.raises(ValueError, match="branching"):
        HawkesFlowSCM([core, flow, price], flow_var=1, mu=0.2,
                      alpha=1.5, beta=1.0, jump_scale=0.5)
    with pytest.raises(ValueError, match="sign_persist"):
        HawkesFlowSCM([core, flow, price], flow_var=1, mu=0.2,
                      alpha=0.5, beta=1.0, jump_scale=0.5, sign_persist=0.3)


def test_counterfactual_pair_shares_arrivals():
    """Pre-onset X_int must equal X_obs bit-for-bit: the interventional arm
    replays the identical pre-drawn arrival sequence (shared randomness)."""
    prior = MarketDoTime(coupling_gamma=1.0, core_share=0.7,
                         hawkes_rate=(0.3, 0.3), seed=7)
    for _ in range(3):
        sample = prior.generate_sample()
        onset = int(sample["int_onset_idx"])
        if onset < 2:
            continue
        pre_obs = sample["X_obs_full"][:onset]
        pre_int = sample["X_int"][:onset]
        assert torch.equal(pre_obs, pre_int)


def test_hawkes_flow_has_jump_outliers_and_sign_runs():
    """Feature-on flow shows (i) heavy-tailed increments (arrival jumps) and
    (ii) longer same-sign runs than the plain OU flow (sign persistence +
    self-excitation) — the count-driven structure rung 3 exists to model."""
    def flow_stats(hawkes_rate, seed=3):
        sampler = MarketContinuousSCMSampler(hawkes_rate=hawkes_rate, seed=seed)
        scm, _, _ = sampler.sample()
        times = torch.arange(400, dtype=torch.float32)
        dts = torch.ones(399)
        gen = torch.Generator().manual_seed(11)
        noise = scm._draw_noise(399 * 2, generator=gen)
        _, x = scm.simulate(times, dts, intervention=None, noise=noise,
                            num_substeps=2)
        flow = x[:, 1]
        inc = flow[1:] - flow[:-1]
        kurt = float(((inc - inc.mean()) ** 4).mean() / inc.var() ** 2)
        s = torch.sign(flow)
        runs = (s[1:] != s[:-1]).sum()
        mean_run = len(s) / max(int(runs) + 1, 1)
        return kurt, mean_run

    kurt_on, run_on = flow_stats((0.4, 0.4))
    kurt_off, run_off = flow_stats((0.0, 0.0))
    assert kurt_on > kurt_off
    assert run_on > run_off
