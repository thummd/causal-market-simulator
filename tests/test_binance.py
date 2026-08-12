"""Phase 4: aggTrades binning, burst detection, and burst episodes — all
offline (synthetic trade frames); no network in CI."""

import numpy as np
import pandas as pd
import pytest
import torch

from dotime import InterventionType
from dotime_market.data.binance import bin_agg_trades
from dotime_market.data.liquidation import burst_episode, detect_flow_bursts


def _trades(times_s, qtys, buyer_maker, prices=None):
    return pd.DataFrame(
        {
            "price": prices if prices is not None else np.full(len(times_s), 100.0),
            "quantity": qtys,
            "transact_time": np.asarray(times_s) * 1000,
            "is_buyer_maker": buyer_maker,
        }
    )


# ------------------------------------------------------------------ binning


def test_bin_agg_trades_signs_and_bars():
    # Bar 0: aggressive buy 2.0 and sell 0.5 -> flow +1.5, volume 2.5.
    # Bar 2 (bar 1 empty): aggressive sell 1.0 -> flow -1.0; price forward-fills.
    df = _trades(
        times_s=[0, 30, 130],
        qtys=[2.0, 0.5, 1.0],
        buyer_maker=[False, True, True],
        prices=[100.0, 101.0, 99.0],
    )
    bars = bin_agg_trades(df, bar_seconds=60)
    assert len(bars) == 3
    np.testing.assert_allclose(bars["flow"], [1.5, 0.0, -1.0])
    np.testing.assert_allclose(bars["volume"], [2.5, 0.0, 1.0])
    # price_bps: bar0 last=101; bar1 forward-fills 101; bar2 = 99.
    assert bars["price_bps"][0] == pytest.approx(0.0)  # vs first bar's own last
    assert bars["price_bps"][1] == pytest.approx(bars["price_bps"][0])
    assert bars["price_bps"][2] < bars["price_bps"][1]


# ---------------------------------------------------------------- detection


def _bars_with_burst(n=300, burst_at=200, burst_size=50.0, seed=0):
    rng = np.random.RandomState(seed)
    flow = rng.normal(0, 1.0, size=n)
    flow[burst_at] += burst_size
    price = np.cumsum(rng.normal(0, 1.0, size=n))
    return pd.DataFrame(
        {"time": np.arange(n) * 60, "flow": flow,
         "price_bps": price - price[0], "volume": np.abs(flow)}
    )


def test_detect_flow_bursts_finds_the_burst():
    bars = _bars_with_burst()
    events = detect_flow_bursts(bars, z_thresh=5.0, pre_bars=90, post_bars=30)
    assert events == [200]


def test_detect_flow_bursts_respects_context_and_gap():
    bars = _bars_with_burst(burst_at=50)  # not enough pre-context
    assert detect_flow_bursts(bars, pre_bars=90, post_bars=30) == []
    # Two bursts closer than min_gap: exactly one survives the cluster
    # (the one with the higher z — at 150, whose lookback is clean; the raw
    # -larger burst at 170 has its z deflated by 150 contaminating its
    # lookback window).
    bars2 = _bars_with_burst(burst_at=150, burst_size=30.0)
    bars2.loc[170, "flow"] += 60.0
    events = detect_flow_bursts(bars2, z_thresh=5.0, min_gap_bars=60)
    assert events == [150]


# ------------------------------------------------------------------ episode


def test_burst_episode_fields_and_standardization():
    bars = _bars_with_burst()
    rng = np.random.RandomState(1)
    ep = burst_episode(bars, 200, rng=rng, pre_bars=90, post_bars=30, window_bars=3)
    assert ep.x_obs.shape == (120, 2)
    assert torch.equal(ep.x_obs, ep.x_int)  # factual task: no counterfactual
    assert ep.structure == "binance_flow_burst"
    assert ep.intervention.intervention_type == InterventionType.SOFT
    assert ep.intervention.times == [90, 91, 92]
    # Pre-onset flow std ~1 after standardization (scale uses population
    # std, torch reports sample std: ratio sqrt(90/89)).
    assert float(ep.x_obs[:90, 0].std()) == pytest.approx(1.0, rel=0.01)
    assert float(ep.intervention.values) == pytest.approx(
        ep.metadata["excess_flow"], rel=1e-6
    )
    assert ep.metadata["excess_flow"] > 10.0  # 50-sigma burst over 3 bars
    for q in range(ep.query_target.numel()):
        q_idx = ep.metadata["query_idx"][q]
        assert q_idx >= 90
        assert float(ep.y_true[q]) == pytest.approx(float(ep.x_int[q_idx, 1]))
