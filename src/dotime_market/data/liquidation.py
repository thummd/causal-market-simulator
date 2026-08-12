"""Liquidation-style flow-burst episodes from Binance aggTrades (Phase 4).

The architecture planned episodes around *labeled* liquidations
(``liquidationSnapshot``), but that dataset has been removed from the public
dumps (verified 2026-07-13 — see :mod:`.binance`). This module implements
the secondary route the architecture also anticipated: **detected** extreme
one-sided flow bursts, which include liquidation cascades. The exogeneity
claim is correspondingly weaker — bursts are *plausibly* forced/exogenous
(margin calls, stop cascades) but not certified — and the paper should say
so.

Unlike the synthetic and ABIDES suites there is NO counterfactual here:
``y_true`` is the *realized* post-burst price. Methods therefore compete on
predicting the factual outcome given the burst size; the meaningful causal
readout is again the PAIRED delta between the causal PFN and its do-ablated
twin on identical episodes (does knowing the burst's signed size improve the
prediction?).

Episode encoding matches the ABIDES adapter: variable 0 = signed flow per
bar, variable 1 = price in log-bps, both standardized by pre-onset stds;
intervention = SOFT (additive forced flow) over the burst bars with value =
mean signed excess flow per bar in rescaled units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dotime import InterventionSpec, InterventionType
from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

from .binance import bin_agg_trades, download_agg_trades

__all__ = ["detect_flow_bursts", "burst_episode", "build_binance_suite"]

_STRUCTURE = "binance_flow_burst"


def detect_flow_bursts(
    bars: pd.DataFrame,
    z_thresh: float = 5.0,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    min_gap_bars: int = 60,
) -> list:
    """Indices of extreme one-sided flow bursts with clean context.

    A bar qualifies when ``|flow|`` exceeds ``z_thresh`` standard deviations
    of the *preceding* ``pre_bars`` window (so the trigger uses only past
    information), the full ``[idx - pre_bars, idx + post_bars]`` range fits
    in the day, and events are at least ``min_gap_bars`` apart (keep the
    largest in each cluster).
    """
    flow = bars["flow"].to_numpy()
    n = len(flow)
    candidates = []
    for i in range(pre_bars, n - post_bars - window_bars):
        pre = flow[i - pre_bars : i]
        sd = pre.std()
        if sd <= 0:
            continue
        z = abs(flow[i] - pre.mean()) / sd
        if z >= z_thresh:
            candidates.append((z, i))

    # Greedy non-overlap: strongest bursts first, enforce the gap.
    events: list = []
    for _, i in sorted(candidates, reverse=True):
        if all(abs(i - j) >= min_gap_bars for j in events):
            events.append(i)
    return sorted(events)


def burst_episode(
    bars: pd.DataFrame,
    event_idx: int,
    rng: np.random.RandomState,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    n_queries: int = 1,
    standardize: bool = True,
) -> Episode:
    """One detected burst -> a dotime Episode (factual outcome as y_true)."""
    lo, hi = event_idx - pre_bars, event_idx + post_bars
    seg = bars.iloc[lo:hi]
    T = len(seg)
    onset = pre_bars
    end = onset + window_bars

    flow = seg["flow"].to_numpy().copy()
    price = seg["price_bps"].to_numpy()
    price = price - price[0]  # re-anchor to the episode start
    x = torch.tensor(np.column_stack([flow, price]), dtype=torch.float32)

    pre_flow = flow[:onset]
    excess = float(flow[onset:end].mean() - pre_flow.mean())

    flow_scale = price_scale = 1.0
    if standardize:
        flow_scale = float(max(pre_flow.std(), 1e-9))
        price_scale = float(max(price[:onset].std(), 1e-9))
        x = x / torch.tensor([flow_scale, price_scale], dtype=torch.float32)
        excess = excess / flow_scale

    intervention = InterventionSpec(
        targets=[0],
        times=list(range(onset, end)),
        intervention_type=InterventionType.SOFT,
        values=float(excess),
    )
    hi_q = max(onset + 1, T - 1)
    q_idx = rng.randint(onset, hi_q + 1, size=n_queries)
    return Episode(
        x_obs=x,          # realized trajectory: no counterfactual exists on
        x_int=x.clone(),  # real data — factual prediction is the task
        intervention=intervention,
        y_true=x[q_idx, 1].to(torch.float32),
        query_target=torch.full((n_queries,), 1, dtype=torch.long),
        query_time=torch.tensor(q_idx / max(T - 1, 1), dtype=torch.float32),
        structure=_STRUCTURE,
        scm_id=int(bars["time"].iloc[event_idx]),
        metadata={
            "query_idx": [int(i) for i in q_idx],
            "event_bar_time": int(bars["time"].iloc[event_idx]),
            "excess_flow": excess,
            "flow_scale": flow_scale,
            "price_scale": price_scale,
        },
    )


def build_binance_suite(
    symbol: str,
    dates: list,
    bar_seconds: int = 60,
    cache_dir: str = "data/raw/binance",
    z_thresh: float = 5.0,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    n_queries: int = 1,
    seed: int = 0,
    name: str = "binance-flow-bursts",
) -> BenchmarkSuite:
    """Detected-burst episodes over one symbol and a list of dates."""
    rng = np.random.RandomState(seed)
    episodes = []
    for date in dates:
        bars = bin_agg_trades(
            download_agg_trades(symbol, date, cache_dir=cache_dir),
            bar_seconds=bar_seconds,
        )
        for idx in detect_flow_bursts(
            bars, z_thresh=z_thresh, pre_bars=pre_bars,
            post_bars=post_bars, window_bars=window_bars,
        ):
            episodes.append(
                burst_episode(
                    bars, idx, rng=rng, pre_bars=pre_bars, post_bars=post_bars,
                    window_bars=window_bars, n_queries=n_queries,
                )
            )
    meta = SuiteMetadata(
        name=name, version="0.0.1", zenodo_record_id="", doi="",
        description=f"{symbol} extreme one-sided flow bursts (liquidation proxies).",
        n_episodes=len(episodes), structures=(_STRUCTURE,),
    )
    return BenchmarkSuite(meta, episodes)
