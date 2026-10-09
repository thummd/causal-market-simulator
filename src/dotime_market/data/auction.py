"""Closing-auction imbalance episodes: an announced, mechanically-timed intervention.

Nasdaq publishes the closing-cross Net Order Imbalance Indicator (NOII) from
15:50 ET; the imbalance is a known-size, known-side quantity of shares that
the 16:00 cross will execute against whatever offsetting liquidity arrives.
Compared with detected flow bursts, the *timing* is exogenous (fixed by the
exchange) and the *dose* is observed before the outcome, which makes this
the equity analogue of the liquidation route in the paper plan. Timing
exogeneity does not make the imbalance itself informationally clean: index
rebalances and fund flows drive it, so results are still "predictive bias
reduction consistent with causal transfer", not proof of it.

Two encodings, both through :func:`~.liquidation.make_flow_episode`:

* ``announced`` — dose = first-published (15:50) signed imbalance spread over
  ``window_bars`` bars at 15:50; 20-second bars give a 90-bar context (15:20
  to 15:50) and a 31-bar post window reaching the cross, the burst tier's
  shape. Measured 2026-09-16 to be a null: the announced quantity is
  absorbed before the cross (median final/first = 0.05), so it is not the
  flow that hits the book.
* ``cross`` (law-matched) — dose = the *final* signed imbalance, i.e. the
  net quantity that sweeps the continuous book at the 16:00 cross; onset =
  the cross bar, whose price is set to the cross print and whose flow
  channel carries the signed residual (the paired volume is two-sided and
  nets to zero). Context = 90 bars before the cross, one post bar, so the
  single query is the cross itself. This is the mechanical impact the
  prior's intervention describes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

from .databento import RAW_DIR, bin_trades, et_epoch, load_bbo, load_closing_imbalance, load_trades
from .liquidation import make_flow_episode

__all__ = ["first_published_imbalance", "final_published_imbalance", "onset_index",
           "cross_bar_index", "auction_episode", "announced_channel_episode", "cross_episode",
           "announced_path_episode", "build_auction_suite"]

_STRUCTURE = "xnas_closing_auction"
_STRUCTURE_CROSS = "xnas_closing_cross"
_NOII_START_SECS = 15 * 3600 + 50 * 60


def first_published_imbalance(noii: pd.DataFrame, symbol: str, day: str,
                              at_or_after_secs: float = _NOII_START_SECS) -> pd.Series | None:
    """First closing-cross NOII row for ``(symbol, day)`` at or after 15:50 ET.

    Args:
        noii: Frame from :func:`~.databento.load_closing_imbalance`.
        symbol: Raw symbol.
        day: ``YYYY-MM-DD``.
        at_or_after_secs: Earliest ET seconds-of-day to accept.

    Returns:
        The row (as a Series) or ``None`` if no publication exists.
    """
    rows = noii[(noii["symbol"] == symbol) & (noii["day"] == day) & (noii["secs"] >= at_or_after_secs)]
    if rows.empty:
        return None
    return rows.iloc[0]


def final_published_imbalance(noii: pd.DataFrame, symbol: str, day: str) -> pd.Series | None:
    """Last closing-cross NOII row for ``(symbol, day)`` before the cross (15:59:59).

    Args:
        noii: Frame from :func:`~.databento.load_closing_imbalance`.
        symbol: Raw symbol.
        day: ``YYYY-MM-DD``.

    Returns:
        The row or ``None``.
    """
    rows = noii[(noii["symbol"] == symbol) & (noii["day"] == day) & (noii["secs"] >= _NOII_START_SECS)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def cross_bar_index(trades: pd.DataFrame, bars: pd.DataFrame, day: str,
                    bar_seconds: int) -> tuple[int, float, float] | None:
    """Locate the closing-cross print and the bar that contains it.

    The cross is the largest trade at or after 15:59:59 ET (millions of
    shares, unsigned); using the print itself rather than the 16:00:00
    wall clock avoids a bar boundary falling between 16:00:00.0 and the
    print's ~0.4 s latency.

    Args:
        trades: Symbol-day trade frame (``price, quantity, transact_time`` ms).
        bars: Bars built from it.
        day: ``YYYY-MM-DD``.
        bar_seconds: Bar width.

    Returns:
        ``(bar_index, cross_price, cross_size)`` or ``None`` if no print exists.
    """
    t = trades["transact_time"].to_numpy() / 1000.0
    late = np.where(t >= et_epoch(day, 15, 59, 59))[0]
    if len(late) == 0:
        return None
    j = late[np.argmax(trades["quantity"].to_numpy()[late])]
    idx = onset_index(bars, int(t[j]), bar_seconds)
    if idx >= len(bars):
        return None
    return idx, float(trades["price"].iloc[j]), float(trades["quantity"].iloc[j])


def onset_index(bars: pd.DataFrame, onset_epoch: int, bar_seconds: int) -> int:
    """Index of the bar containing ``onset_epoch``.

    Bars from :func:`~.binance.bin_signed_trades` are anchored at the first
    trade and carry bar-END times, so bar ``i`` covers
    ``[t0 + i*bs, t0 + (i+1)*bs)``.

    Args:
        bars: Bar frame.
        onset_epoch: Epoch seconds of the onset.
        bar_seconds: Bar width used to build ``bars``.

    Returns:
        Integer bar index (may fall outside the frame; callers check).
    """
    t0 = int(bars["time"].iloc[0]) - bar_seconds
    return int((onset_epoch - t0) // bar_seconds)


def auction_episode(bars: pd.DataFrame, onset_idx: int, signed_imbalance: float,
                    rng: np.random.RandomState, pre_bars: int = 90, post_bars: int = 31,
                    window_bars: int = 3, n_queries: int = 1,
                    extra_metadata: dict | None = None) -> Episode:
    """Encode one closing auction as a soft-intervention Episode.

    The dose is ``signed_imbalance / window_bars`` shares per bar: the cross
    executes the imbalance at once, but the PFN's injection format is a
    short additive-flow window, so the announced quantity is spread over it.

    Args:
        bars: Bar frame for the day.
        onset_idx: Bar index of the 15:50 publication.
        signed_imbalance: Shares, buy imbalance positive.
        rng: Query RNG.
        pre_bars, post_bars, window_bars, n_queries: Episode shape.
        extra_metadata: Merged into ``Episode.metadata``.

    Returns:
        Episode with structure ``xnas_closing_auction``.
    """
    meta = {"signed_imbalance": float(signed_imbalance)}
    if extra_metadata:
        meta.update(extra_metadata)
    return make_flow_episode(
        bars, onset_idx, float(signed_imbalance) / window_bars, rng, pre_bars=pre_bars,
        post_bars=post_bars, window_bars=window_bars, n_queries=n_queries,
        structure=_STRUCTURE, extra_metadata=meta,
    )


def announced_channel_episode(bars: pd.DataFrame, onset_idx: int, signed_imbalance: float,
                              rng: np.random.RandomState, pre_bars: int = 90, post_bars: int = 31,
                              window_bars: int = 3, n_queries: int = 1, lead_bars: int = 3,
                              extra_metadata: dict | None = None) -> Episode:
    """Announced encoding for a prior trained with the announcement channel.

    Four-column canon of the announced prior — ``(flow, hidden zeros,
    announcement channel, price)``. The channel equals the per-bar announced
    dose from the 15:50 publication bar onward, in the same standardized
    flow units as the intervention value, and the intervention onset is
    placed ``lead_bars`` after the publication so the announcement sits in
    the causally visible context, exactly as in training where the channel
    switches on at ``t_a < t_0``. (An onset at the publication bar itself
    would leave the channel entirely inside the masked post-onset region:
    the first v21/v22 acceptance runs had that flaw and showed no channel
    effect.) The post window is shortened by ``lead_bars`` so the episode
    still ends at the cross bar.

    Args:
        bars: Bar frame for the day.
        onset_idx: Bar index of the 15:50 publication.
        signed_imbalance: First-published imbalance (shares, buy positive).
        rng: Query RNG.
        pre_bars, post_bars, window_bars, n_queries: Episode shape as for
            :func:`auction_episode` (measured from the publication bar).
        lead_bars: Bars between publication and intervention onset (>= 1).
        extra_metadata: Merged into ``Episode.metadata``.

    Returns:
        Episode with structure ``xnas_closing_auction_announced``; price is
        column 3 and the query target.

    Raises:
        ValueError: If ``lead_bars`` is not in ``[1, post_bars)``.
    """
    if not 1 <= lead_bars < post_bars:
        raise ValueError("lead_bars must satisfy 1 <= lead_bars < post_bars")
    base = auction_episode(bars, onset_idx + lead_bars, signed_imbalance, rng, pre_bars=pre_bars,
                           post_bars=post_bars - lead_bars, window_bars=window_bars,
                           n_queries=n_queries, extra_metadata=extra_metadata)
    T = base.x_obs.shape[0]
    dose = float(base.intervention.values)  # announced per-bar dose, standardized
    w = torch.zeros(T)
    w[pre_bars - lead_bars:] = dose  # visible from the publication bar, before the onset
    x = torch.column_stack([base.x_obs[:, 0], torch.zeros(T), w, base.x_obs[:, 1]])
    q_idx = torch.tensor(base.metadata["query_idx"], dtype=torch.long)
    return Episode(
        x_obs=x, x_int=x.clone(), intervention=base.intervention,
        y_true=x[q_idx, 3].to(torch.float32),
        query_target=torch.full((n_queries,), 3, dtype=torch.long),
        query_time=base.query_time, structure=_STRUCTURE + "_announced", scm_id=base.scm_id,
        metadata={**base.metadata, "lead_bars": lead_bars},
    )


def cross_episode(bars: pd.DataFrame, cross_idx: int, cross_price: float, final_imbalance: float,
                  rng: np.random.RandomState, pre_bars: int = 90, n_queries: int = 1,
                  anchor_price: float | None = None,
                  extra_metadata: dict | None = None) -> Episode:
    """Law-matched encoding: the cross as a one-bar soft intervention.

    The cross bar's price is overwritten with the cross print (late
    off-cross prints in the same 20 s would otherwise define the close)
    and its flow channel is set to the signed final imbalance, the net
    quantity the continuous book absorbs at 16:00. Bars after the cross
    are dropped, so the single query is the cross bar.

    Args:
        bars: Bar frame for the day (not modified).
        cross_idx: Bar containing the cross print.
        cross_price: Cross print price (absolute).
        final_imbalance: Signed final NOII imbalance (shares, buy positive).
        rng: Query RNG (degenerate: one legal query).
        pre_bars: Context bars before the cross.
        n_queries: Queries (all land on the cross bar).
        anchor_price: Absolute price behind ``price_bps == 0``; defaults to
            ``bars.attrs["anchor_price"]`` set by the bar builder.
        extra_metadata: Merged into ``Episode.metadata``.

    Returns:
        Episode with structure ``xnas_closing_cross``.

    Raises:
        ValueError: If no anchor price is available.
    """
    anchor = anchor_price if anchor_price is not None else bars.attrs.get("anchor_price")
    if anchor is None:
        raise ValueError("anchor_price required (bars.attrs['anchor_price'] missing)")
    b = bars.iloc[: cross_idx + 1].copy()
    b.loc[b.index[cross_idx], "price_bps"] = 1e4 * np.log(cross_price / anchor)
    b.loc[b.index[cross_idx], "flow"] = float(final_imbalance)
    meta = {"final_imbalance": float(final_imbalance), "cross_price": float(cross_price)}
    if extra_metadata:
        meta.update(extra_metadata)
    return make_flow_episode(
        b, cross_idx, float(final_imbalance), rng, pre_bars=pre_bars, post_bars=1,
        window_bars=1, n_queries=n_queries, structure=_STRUCTURE_CROSS, extra_metadata=meta,
    )


def announced_path_episode(bars: pd.DataFrame, cross_idx: int, cross_price: float,
                           noii_day: pd.DataFrame, day: str, rng: np.random.RandomState,
                           pre_bars: int = 90, n_queries: int = 1,
                           extra_metadata: dict | None = None) -> Episode:
    """Encoding for models trained with the relaxation announcement layer.

    Mirrors the ``relax`` prior on real data. The onset is the closing cross, where
    the residual imbalance executes. The dose the model is told is the FIRST
    published imbalance (the announced dose). The announcement channel carries the
    imbalance actually published from 15:50 until the cross, forward-filled to each
    bar end, so the absorption path, and in particular its last value before the cross
    (the executed residual), is in the causally visible context exactly as in training.

    Args:
        bars: Bar frame for the day; ``time`` is bar-end epoch seconds.
        cross_idx: Bar containing the closing-cross print.
        cross_price: Cross print price.
        noii_day: This symbol-day's closing-cross NOII rows (``secs``,
            ``signed_imbalance``), sorted by time, from 15:50 on.
        day: ``YYYY-MM-DD``.
        rng: Query RNG.
        pre_bars: Context bars before the cross.
        n_queries: Queries (all at the cross bar).
        extra_metadata: Merged into ``Episode.metadata``.

    Returns:
        Four-column episode (flow, hidden zeros, announcement channel, price) with
        structure ``xnas_closing_announced_path``.

    Raises:
        ValueError: If ``noii_day`` is empty.
    """
    if noii_day.empty:
        raise ValueError("no NOII publications for this symbol-day")
    announced = float(noii_day["signed_imbalance"].iloc[0])
    final = float(noii_day["signed_imbalance"].iloc[-1])
    meta = {"announced_imbalance": announced, "final_imbalance": final}
    if extra_metadata:
        meta.update(extra_metadata)
    # cross_episode standardizes by the pre-cross flow sd and uses the value it is given
    # as the one-bar dose; passing the announced imbalance makes the dose D_ann.
    base = cross_episode(bars, cross_idx, cross_price, announced, rng, pre_bars=pre_bars,
                         n_queries=n_queries, extra_metadata=meta)
    flow_scale = float(base.metadata["flow_scale"])
    seg_times = bars["time"].to_numpy()[cross_idx - pre_bars: cross_idx + 1]
    midnight = et_epoch(day, 0, 0)
    secs_end = seg_times - midnight
    pub_secs = noii_day["secs"].to_numpy(dtype=float)
    pub_vals = noii_day["signed_imbalance"].to_numpy(dtype=float)
    idx = np.searchsorted(pub_secs, secs_end, side="right") - 1
    w = np.where(idx >= 0, pub_vals[np.clip(idx, 0, None)], 0.0) / flow_scale
    T = base.x_obs.shape[0]
    x = torch.column_stack([base.x_obs[:, 0], torch.zeros(T),
                            torch.tensor(w, dtype=torch.float32), base.x_obs[:, 1]])
    q_idx = torch.tensor(base.metadata["query_idx"], dtype=torch.long)
    return Episode(
        x_obs=x, x_int=x.clone(), intervention=base.intervention,
        y_true=x[q_idx, 3].to(torch.float32),
        query_target=torch.full((n_queries,), 3, dtype=torch.long),
        query_time=base.query_time, structure=_STRUCTURE + "_announced_path",
        scm_id=base.scm_id,
        metadata={**base.metadata, "channel_last_before_cross": float(w[pre_bars - 1])},
    )


def build_auction_suite(
    symbols: list[str],
    dates: list[str],
    bar_seconds: int = 20,
    pre_bars: int = 90,
    post_bars: int = 31,
    window_bars: int = 3,
    n_queries: int = 1,
    seed: int = 0,
    cache_dir: str | Path = RAW_DIR / "trades",
    bbo_dir: str | Path | None = None,
    noii: pd.DataFrame | None = None,
    encoding: str = "announced",
    name: str = "xnas-closing-auctions",
) -> tuple[BenchmarkSuite, pd.DataFrame]:
    """Closing-auction episodes plus a per-event table for the slope check.

    Args:
        symbols: Raw symbols with cached trades.
        dates: Trading days.
        bar_seconds: Bar width (20 s keeps the burst-tier episode shape).
        pre_bars, post_bars, window_bars, n_queries, seed: Episode settings
            (``post_bars``/``window_bars`` apply to the ``announced`` encoding only).
        cache_dir: Trades cache; ``bbo_dir`` optionally switches the context price to mids.
        noii: Pre-loaded closing-imbalance frame (loaded if ``None``).
        encoding: ``"announced"`` (15:50 publication as dose), ``"announced_channel"``
            (same, in the four-column canon of the announced prior with the
            announcement channel filled) or ``"cross"`` (final imbalance at
            the cross as dose; law-matched).
        name: Suite name.

    Returns:
        ``(suite, table)`` with one table row per episode: ``symbol, day,
        signed_imbalance`` (the dose source: first or final publication),
        ``final_imbalance, paired_qty, flow_scale, dose`` (standardized
        intervention value), ``ret_bps`` (announced: cross bar minus onset
        bar; cross: cross print minus the last pre-cross bar), ``day_volume``.

    Raises:
        ValueError: If ``encoding`` is unknown.
    """
    if encoding not in ("announced", "announced_channel", "cross", "announced_path"):
        raise ValueError("encoding must be 'announced', 'announced_channel', 'cross' "
                         "or 'announced_path'")
    if noii is None:
        noii = load_closing_imbalance()
    rng = np.random.RandomState(seed)
    episodes, rows = [], []
    for symbol in symbols:
        for day in dates:
            try:
                trades = load_trades(symbol, day, cache_dir=cache_dir)
            except FileNotFoundError:
                continue
            first = first_published_imbalance(noii, symbol, day)
            final = final_published_imbalance(noii, symbol, day)
            if first is None or final is None:
                continue
            bbo = load_bbo(symbol, day, cache_dir=bbo_dir) if bbo_dir is not None else None
            bars = bin_trades(trades, bar_seconds=bar_seconds, bbo=bbo)
            cross = cross_bar_index(trades, bars, day, bar_seconds)
            if cross is None:
                continue
            cross_idx, cross_price, cross_size = cross
            price = bars["price_bps"].to_numpy()
            common = {"symbol": symbol, "date": day, "paired_qty": float(final["paired_qty"]),
                      "cross_size": cross_size}
            if encoding in ("announced", "announced_channel"):
                if first["signed_imbalance"] == 0:
                    continue  # unsigned (balanced) book at publication
                idx = onset_index(bars, et_epoch(day, 15, 50), bar_seconds)
                if idx - pre_bars < 0 or idx + post_bars > len(bars):
                    continue
                builder = auction_episode if encoding == "announced" else announced_channel_episode
                ep = builder(bars, idx, first["signed_imbalance"], rng, pre_bars=pre_bars,
                             post_bars=post_bars, window_bars=window_bars,
                             n_queries=n_queries, extra_metadata=common)
                dose_src = float(first["signed_imbalance"])
                ret = float(price[idx + post_bars - 1] - price[idx])
            elif encoding == "announced_path":
                if first["signed_imbalance"] == 0 or cross_idx - pre_bars < 0:
                    continue
                day_rows = noii[(noii["symbol"] == symbol) & (noii["day"] == day)
                                & (noii["secs"] >= _NOII_START_SECS)]
                ep = announced_path_episode(bars, cross_idx, cross_price, day_rows, day, rng,
                                            pre_bars=pre_bars, n_queries=n_queries,
                                            extra_metadata=common)
                dose_src = float(first["signed_imbalance"])
                ret = float(ep.metadata["price_scale"]
                            * (ep.x_obs[pre_bars, 3] - ep.x_obs[pre_bars - 1, 3]))
            else:
                if final["signed_imbalance"] == 0 or cross_idx - pre_bars < 0:
                    continue
                ep = cross_episode(bars, cross_idx, cross_price, final["signed_imbalance"], rng,
                                   pre_bars=pre_bars, n_queries=n_queries, extra_metadata=common)
                dose_src = float(final["signed_imbalance"])
                # The episode is re-indexed so the cross sits at pre_bars; undo the
                # standardization to report the cross move in bps.
                ret = float(ep.metadata["price_scale"]
                            * (ep.x_obs[pre_bars, 1] - ep.x_obs[pre_bars - 1, 1]))
            episodes.append(ep)
            rows.append({
                "symbol": symbol, "day": day, "signed_imbalance": dose_src,
                "final_imbalance": float(final["signed_imbalance"]),
                "paired_qty": float(final["paired_qty"]),
                "flow_scale": ep.metadata["flow_scale"], "dose": ep.metadata["excess_flow"],
                "ret_bps": ret, "day_volume": float(bars["volume"].sum()),
            })
    structure = {"announced": _STRUCTURE, "announced_channel": _STRUCTURE + "_announced",
                 "announced_path": _STRUCTURE + "_announced_path",
                 "cross": _STRUCTURE_CROSS}[encoding]
    meta = SuiteMetadata(
        name=f"{name}-{encoding}", version="0.0.1", zenodo_record_id="", doi="",
        description=f"Nasdaq closing cross, {encoding} encoding (see data/auction.py).",
        n_episodes=len(episodes), structures=(structure,),
    )
    return BenchmarkSuite(meta, episodes), pd.DataFrame(rows)
