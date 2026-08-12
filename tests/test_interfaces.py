"""The pluggable-prior interfaces are satisfied by the shipped mechanisms —
the structural guarantee the upstreaming plan rests on."""

import numpy as np
import pytest
import torch

from dotime.continuous import ContinuousTSCMSampler
from dotime.tscm_sampler import TSCMStructure
from dotime_market.prior import (
    CoreReactionConfounder,
    EpisodeSCMSampler,
    InformationalCoupling,
    MarketContinuousSCMSampler,
    MechanismFactory,
    UniformWindowTiming,
)


def test_shipped_mechanisms_satisfy_mechanism_factory():
    conf = CoreReactionConfounder(core_share=0.5)
    coup = InformationalCoupling(gamma=1.0, impact_lambda=0.5)
    assert isinstance(conf, MechanismFactory)
    assert isinstance(coup, MechanismFactory)
    # And the unified call really returns mechanism tuples.
    assert len(conf.mechanisms(0, 1)) == 2
    assert len(coup.mechanisms(0, 1, 2)) == 1


def test_samplers_satisfy_episode_scm_sampler():
    market = MarketContinuousSCMSampler(seed=0)
    stock = ContinuousTSCMSampler(structure=TSCMStructure.BACK_DOOR)
    assert isinstance(market, EpisodeSCMSampler)
    # The stock dotime sampler satisfies the same duck-type structurally —
    # that is the whole point of the upstream proposal.
    assert isinstance(stock, EpisodeSCMSampler)


def test_uniform_window_timing_mirrors_stock_behaviour():
    timing = UniformWindowTiming(window_frac=(0.1, 0.3))
    times = torch.arange(100, dtype=torch.float32)
    rng = np.random.RandomState(7)
    starts, lengths = [], []
    for _ in range(200):
        t_start, t_end = timing.sample_window(times, None, rng)
        assert 0.0 <= t_start < t_end <= 99.0 + 1e-6
        starts.append(t_start)
        lengths.append(t_end - t_start)
    lengths = np.array(lengths)
    assert lengths.min() >= 0.1 * 99.0 - 1e-6
    assert lengths.max() <= 0.3 * 99.0 + 1e-6
    assert min(starts) >= 0.3 * 99.0 - 1e-6  # stock earliest-start rule

    # Reproducible under a seeded rng.
    w1 = UniformWindowTiming().sample_window(times, None, np.random.RandomState(3))
    w2 = UniformWindowTiming().sample_window(times, None, np.random.RandomState(3))
    assert w1 == w2

    with pytest.raises(ValueError, match="window_frac"):
        UniformWindowTiming(window_frac=(0.0, 0.3))
