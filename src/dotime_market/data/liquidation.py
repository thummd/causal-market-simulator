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

from typing import Optional

import numpy as np
import pandas as pd
import torch

from dotime import InterventionSpec, InterventionType
from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

from .binance import bin_agg_trades, download_agg_trades

__all__ = ["detect_flow_bursts", "make_flow_episode", "burst_episode", "widen_to_canon",
           "build_binance_suite"]

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


def make_flow_episode(
    bars: pd.DataFrame,
    event_idx: int,
    excess_raw: float,
    rng: np.random.RandomState,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    n_queries: int = 1,
    standardize: bool = True,
    structure: str = _STRUCTURE,
    extra_metadata: Optional[dict] = None,
) -> Episode:
    """Encode a flow event at ``event_idx`` as a factual-outcome Episode.

    This is the single episode encoder shared by every real-data tier
    (detected bursts, announced closing-auction imbalances, scheduled FOMC
    windows); only the *dose* ``excess_raw`` differs between them, so the
    PFN always sees the same injection format it was validated on.

    Args:
        bars: Frame from :func:`~dotime_market.data.binance.bin_signed_trades`.
        event_idx: Bar index of the intervention onset.
        excess_raw: Intervention dose in raw flow units per bar (before
            standardization by the pre-onset flow std).
        rng: Random state for the query-time draw.
        pre_bars: Context bars before the onset.
        post_bars: Bars after the onset kept in the episode.
        window_bars: Length of the soft-intervention window.
        n_queries: Number of post-onset price queries.
        standardize: Divide flow and price by their pre-onset stds.
        structure: Structure label stored on the Episode.
        extra_metadata: Extra keys merged into ``Episode.metadata``.

    Returns:
        A dotime Episode whose ``x_int`` equals ``x_obs`` (no counterfactual
        exists on real data) and whose ``y_true`` is the realized price.

    Raises:
        ValueError: If the ``[event_idx - pre_bars, event_idx + post_bars)``
            range does not fit inside ``bars``.
    """
    lo, hi = event_idx - pre_bars, event_idx + post_bars
    if lo < 0 or hi > len(bars):
        raise ValueError("event window does not fit inside the bar frame")
    seg = bars.iloc[lo:hi]
    T = len(seg)
    onset = pre_bars
    end = onset + window_bars

    flow = seg["flow"].to_numpy().copy()
    price = seg["price_bps"].to_numpy()
    price = price - price[0]  # re-anchor to the episode start
    x = torch.tensor(np.column_stack([flow, price]), dtype=torch.float32)

    pre_flow = flow[:onset]
    excess = float(excess_raw)

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
    # Queries are drawn on [onset, T-1]; with post_bars == 1 the only legal
    # query is the onset bar itself, so the upper bound must never exceed T-1.
    hi_q = max(onset, T - 1)
    q_idx = rng.randint(onset, hi_q + 1, size=n_queries)
    metadata = {
        "query_idx": [int(i) for i in q_idx],
        "event_bar_time": int(bars["time"].iloc[event_idx]),
        "excess_flow": excess,
        "flow_scale": flow_scale,
        "price_scale": price_scale,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return Episode(
        x_obs=x,          # realized trajectory: no counterfactual exists on
        x_int=x.clone(),  # real data — factual prediction is the task
        intervention=intervention,
        y_true=x[q_idx, 1].to(torch.float32),
        query_target=torch.full((n_queries,), 1, dtype=torch.long),
        query_time=torch.tensor(q_idx / max(T - 1, 1), dtype=torch.float32),
        structure=structure,
        scm_id=int(bars["time"].iloc[event_idx]),
        metadata=metadata,
    )


def widen_to_canon(episode: Episode, n_vars: int) -> Episode:
    """Re-lay a two-column ``(flow, price)`` episode into a wider canonical frame.

    Priors with extra observed nodes (proxy, announcement channel) train with
    the price at canonical index ``n_vars - 1`` and the extra channels in
    between. A real-data episode meant for such a checkpoint must present
    the same layout: flow at 0, zero (masked-as-padding) middle columns, price
    last, with the query target moved accordingly. Two-column episodes are
    returned unchanged when ``n_vars == 2``.

    Args:
        episode: Episode with ``x_obs[:, 0]`` = flow and ``x_obs[:, 1]`` = price.
        n_vars: Target width (>= 2).

    Returns:
        A new Episode (``x_int`` widened identically, ``y_true`` unchanged).

    Raises:
        ValueError: If the episode does not have exactly two columns or ``n_vars < 2``.
    """
    if episode.x_obs.shape[1] != 2 or n_vars < 2:
        raise ValueError("widen_to_canon expects a (flow, price) episode and n_vars >= 2")
    if n_vars == 2:
        return episode

    def widen(x):
        T = x.shape[0]
        out = torch.zeros(T, n_vars, dtype=x.dtype)
        out[:, 0] = x[:, 0]
        out[:, n_vars - 1] = x[:, 1]
        return out

    return Episode(
        x_obs=widen(episode.x_obs), x_int=widen(episode.x_int),
        intervention=episode.intervention, y_true=episode.y_true,
        query_target=torch.full_like(episode.query_target, n_vars - 1),
        query_time=episode.query_time, structure=episode.structure, scm_id=episode.scm_id,
        metadata=episode.metadata,
    )


def burst_episode(
    bars: pd.DataFrame,
    event_idx: int,
    rng: np.random.RandomState,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    n_queries: int = 1,
    standardize: bool = True,
    structure: str = _STRUCTURE,
    extra_metadata: Optional[dict] = None,
) -> Episode:
    """One detected burst -> a dotime Episode (factual outcome as y_true).

    The dose is the realized mean excess flow over the burst window relative
    to the pre-onset mean; encoding is delegated to :func:`make_flow_episode`.
    """
    onset, end = event_idx, event_idx + window_bars
    flow = bars["flow"].to_numpy()
    excess_raw = float(flow[onset:end].mean() - flow[event_idx - pre_bars:onset].mean())
    return make_flow_episode(
        bars, event_idx, excess_raw, rng, pre_bars=pre_bars, post_bars=post_bars,
        window_bars=window_bars, n_queries=n_queries, standardize=standardize,
        structure=structure, extra_metadata=extra_metadata,
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
