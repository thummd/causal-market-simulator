"""Databento tier: binning reduction, unsigned trades, shared episode encoder,
auction/scheduled onsets — all offline on synthetic frames (no DBN files, no
network in CI)."""

import numpy as np
import pandas as pd
import pytest
import torch

from dotime import InterventionType
from dotime_market.data.auction import auction_episode, first_published_imbalance, onset_index
from dotime_market.data.binance import bin_agg_trades, bin_signed_trades
from dotime_market.data.databento import bin_trades, et_epoch
from dotime_market.data.liquidation import burst_episode, make_flow_episode


def _db_trades(times_s, qtys, signs, prices=None):
    return pd.DataFrame({
        "price": prices if prices is not None else np.full(len(times_s), 100.0),
        "quantity": np.asarray(qtys, dtype=float),
        "transact_time": np.asarray(times_s) * 1000,
        "sign": np.asarray(signs, dtype=float),
    })


# ------------------------------------------------------------------ binning


def test_bin_trades_reduces_to_bin_agg_trades_when_all_signed():
    rng = np.random.RandomState(0)
    n = 500
    t = np.sort(rng.randint(0, 3600, size=n))
    qty = rng.uniform(0.1, 5.0, size=n)
    buyer_maker = rng.rand(n) < 0.5
    price = 100 + np.cumsum(rng.normal(0, 0.01, size=n))
    binance = bin_agg_trades(pd.DataFrame({
        "price": price, "quantity": qty, "transact_time": t * 1000, "is_buyer_maker": buyer_maker,
    }), bar_seconds=60)
    db = bin_trades(_db_trades(t, qty, np.where(buyer_maker, -1.0, 1.0), price), bar_seconds=60)
    pd.testing.assert_frame_equal(binance, db)


def test_unsigned_trades_count_in_volume_not_flow():
    df = _db_trades([0, 10, 20], [1.0, 2.0, 4.0], [1.0, 0.0, -1.0])
    bars = bin_trades(df, bar_seconds=60)
    assert len(bars) == 1
    assert bars["flow"][0] == pytest.approx(-3.0)
    assert bars["volume"][0] == pytest.approx(7.0)
    assert bars["n_trades"][0] == 3


def test_bin_trades_with_bbo_uses_mid_at_bar_end():
    df = _db_trades([0, 70, 130], [1, 1, 1], [1, 1, 1], prices=[100.0, 110.0, 120.0])
    # Mids sampled every 10 s; bar ends are 60, 120, 180 -> last mid before
    # each end is at t=50, 110, 170.
    bbo = pd.DataFrame({"time": np.arange(0, 180, 10), "mid": 100 + np.arange(0, 180, 10) / 10})
    bars = bin_trades(df, bar_seconds=60, bbo=bbo)
    mids = np.array([105.0, 111.0, 117.0])
    np.testing.assert_allclose(bars["price_bps"], 1e4 * np.log(mids / mids[0]))
    # Flow is untouched by the price source.
    np.testing.assert_allclose(bars["flow"], [1.0, 1.0, 1.0])


def test_bin_signed_trades_rejects_empty():
    with pytest.raises(ValueError):
        bin_signed_trades(np.array([]), np.array([]), np.array([]), np.array([]))


# ------------------------------------------------------------------ episodes


def _bars(n=200, seed=0):
    rng = np.random.RandomState(seed)
    flow = rng.normal(0, 1.0, size=n)
    price = np.cumsum(rng.normal(0, 1.0, size=n))
    return pd.DataFrame({"time": 1000 + np.arange(n) * 60, "flow": flow,
                         "price_bps": price - price[0], "volume": np.abs(flow),
                         "n_trades": np.ones(n)})


def test_burst_episode_delegates_to_make_flow_episode():
    bars = _bars()
    idx = 120
    a = burst_episode(bars, idx, np.random.RandomState(1), n_queries=3)
    flow = bars["flow"].to_numpy()
    excess_raw = flow[idx:idx + 3].mean() - flow[idx - 90:idx].mean()
    b = make_flow_episode(bars, idx, excess_raw, np.random.RandomState(1), n_queries=3)
    assert a.intervention.values == pytest.approx(b.intervention.values)
    assert a.structure == b.structure == "binance_flow_burst"
    assert np.allclose(a.x_obs.numpy(), b.x_obs.numpy())
    assert np.allclose(a.y_true.numpy(), b.y_true.numpy())
    assert a.metadata["query_idx"] == b.metadata["query_idx"]


def test_make_flow_episode_rejects_window_outside_frame():
    with pytest.raises(ValueError):
        make_flow_episode(_bars(), 10, 1.0, np.random.RandomState(0))


def test_auction_episode_dose_is_imbalance_per_window_bar():
    bars = _bars()
    idx = 120
    ep = auction_episode(bars, idx, signed_imbalance=-3000.0, rng=np.random.RandomState(0),
                         pre_bars=90, post_bars=31, window_bars=3,
                         extra_metadata={"symbol": "AAPL"})
    flow_scale = bars["flow"].to_numpy()[idx - 90:idx].std()
    assert ep.intervention.values == pytest.approx(-1000.0 / flow_scale)
    assert ep.intervention.intervention_type == InterventionType.SOFT
    assert ep.intervention.times == [90, 91, 92]
    assert ep.structure == "xnas_closing_auction"
    assert ep.metadata["symbol"] == "AAPL"
    assert ep.metadata["signed_imbalance"] == -3000.0
    assert ep.x_obs.shape == (121, 2)


# ------------------------------------------------------------------ onsets


def test_onset_index_locates_bar_containing_epoch():
    # Bars anchored at t0=1000 with 20 s width: bar i covers [1000+20i, 1020+20i).
    bars = pd.DataFrame({"time": 1020 + 20 * np.arange(50)})
    assert onset_index(bars, 1000, 20) == 0
    assert onset_index(bars, 1019, 20) == 0
    assert onset_index(bars, 1020, 20) == 1
    assert onset_index(bars, 1000 + 20 * 37 + 5, 20) == 37


def test_et_epoch_is_dst_aware():
    # 15:50 ET is 19:50 UTC in June (EDT) and 20:50 UTC in January (EST).
    assert et_epoch("2026-06-24", 15, 50) == int(pd.Timestamp("2026-06-24 19:50", tz="UTC").timestamp())
    assert et_epoch("2026-01-28", 14, 0) == int(pd.Timestamp("2026-01-28 19:00", tz="UTC").timestamp())


def test_first_published_imbalance_picks_earliest_row_after_1550():
    noii = pd.DataFrame({
        "symbol": ["AAPL"] * 3 + ["MSFT"],
        "day": ["2026-06-24"] * 4,
        "secs": [15 * 3600 + 49 * 60, 15 * 3600 + 50 * 60, 15 * 3600 + 50 * 60 + 10, 15 * 3600 + 55 * 60],
        "signed_imbalance": [1.0, 2.0, 3.0, 4.0],
    })
    row = first_published_imbalance(noii, "AAPL", "2026-06-24")
    assert row["signed_imbalance"] == 2.0
    assert first_published_imbalance(noii, "NVDA", "2026-06-24") is None


# ------------------------------------------------------------ cross encoding


def test_bin_signed_trades_records_anchor_price():
    bars = bin_signed_trades(np.array([0, 30, 130]), np.array([1.0, -1.0, 1.0]),
                             np.array([1.0, 1.0, 1.0]), np.array([100.0, 101.0, 99.0]))
    assert bars.attrs["anchor_price"] == 101.0  # last price of bar 0


def test_cross_bar_index_finds_largest_late_print():
    from dotime_market.data.auction import cross_bar_index
    day = "2026-06-24"
    t0 = et_epoch(day, 15, 30)
    times = np.array([t0, t0 + 600, t0 + 1799, t0 + 1800.38, t0 + 1814])  # cross at 16:00:00.38
    df = _db_trades(times, [10, 20, 30, 5_000_000, 41], [1, -1, 0, 0, 0],
                    prices=[100.0, 100.5, 100.4, 100.2, 100.9])
    bars = bin_trades(df, bar_seconds=20)
    idx, price, size = cross_bar_index(df, bars, day, 20)
    assert size == 5_000_000 and price == 100.2
    assert idx == int((t0 + 1800.38 - t0) // 20)


def test_cross_episode_places_cross_price_and_final_imbalance():
    from dotime_market.data.auction import cross_episode
    bars = _bars(n=120)
    bars.attrs["anchor_price"] = 200.0
    ep = cross_episode(bars, cross_idx=110, cross_price=202.0, final_imbalance=-4000.0,
                       rng=np.random.RandomState(0), pre_bars=90, n_queries=2)
    assert ep.structure == "xnas_closing_cross"
    assert ep.x_obs.shape == (91, 2)  # 90 context bars + the cross bar
    assert ep.intervention.times == [90]
    flow_scale = bars["flow"].to_numpy()[20:110].std()
    assert ep.intervention.values == pytest.approx(-4000.0 / flow_scale)
    # Both queries land on the cross bar and y_true is the cross price in bps.
    assert ep.metadata["query_idx"] == [90, 90]
    seg_price0 = bars["price_bps"].to_numpy()[20]
    expected = (1e4 * np.log(202.0 / 200.0) - seg_price0) / ep.metadata["price_scale"]
    assert ep.y_true[0].item() == pytest.approx(expected, rel=1e-5)
    # The cross bar's flow channel carries the signed final imbalance (standardized).
    assert ep.x_obs[90, 0].item() == pytest.approx(-4000.0 / flow_scale, rel=1e-5)


def test_cross_episode_requires_anchor():
    from dotime_market.data.auction import cross_episode
    bars = _bars(n=120)
    with pytest.raises(ValueError):
        cross_episode(bars, 110, 1.0, 1.0, np.random.RandomState(0))


def test_final_published_imbalance_picks_last_row():
    from dotime_market.data.auction import final_published_imbalance
    noii = pd.DataFrame({"symbol": ["AAPL"] * 3, "day": ["2026-06-24"] * 3,
                         "secs": [15 * 3600 + 50 * 60, 15 * 3600 + 55 * 60, 15 * 3600 + 59 * 60 + 59],
                         "signed_imbalance": [1e6, 2e5, -3e3]})
    assert final_published_imbalance(noii, "AAPL", "2026-06-24")["signed_imbalance"] == -3e3


def test_announced_channel_is_visible_before_onset():
    from dotime_market.data.auction import announced_channel_episode
    bars = _bars(n=200)
    ep = announced_channel_episode(bars, 120, signed_imbalance=3000.0, rng=np.random.RandomState(0),
                                   pre_bars=90, post_bars=31, window_bars=3, lead_bars=3)
    assert ep.x_obs.shape == (118, 4) and ep.query_target.tolist() == [3]  # 90 + (31 - 3) bars
    onset = ep.intervention.times[0]
    assert onset == 90  # onset sits lead_bars after the publication inside the frame
    w = ep.x_obs[:, 2]
    assert torch.all(w[:87] == 0) and torch.all(w[87:] == w[87]) and w[87] != 0
    assert w[87].item() == pytest.approx(ep.intervention.values)
    # Same last bar as the plain announced encoding (post window shortened by lead_bars).
    assert ep.metadata["lead_bars"] == 3
    with pytest.raises(ValueError):
        announced_channel_episode(bars, 120, 1.0, np.random.RandomState(0), lead_bars=0)
