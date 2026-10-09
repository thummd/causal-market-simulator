"""FOMC announcement windows on ES futures: scheduled-time flow episodes.

The paper plan's §5.3 backstop: the 14:00 ET statement is a known-time
intervention on the information state. For the flow-intervention PFN the
natural encoding is the order-flow burst the announcement triggers, taken
at the *scheduled* time rather than detected, and compared with the same
clock window on control Wednesdays one week earlier (same weekday, same
month, no scheduled macro release at 14:00). The contrast isolates what an
information shock does to the flow-to-price response: under the prior's
reading, announcement-window flow is heavily informational (high γ), so the
factual response per unit flow should exceed the control days' and the
do-head's factual gain should shrink.

Data: ``glbx_es_trades_{fomc,ctrl}_{day}.dbn.zst`` from
``scripts/31_databento_pull.py`` (parent symbol ``ES.FUT``, all expiries
and calendar spreads). The front-month outright is chosen per day by trade
count, which handles the June roll (ESM6 -> ESU6) without a calendar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dotime.benchmarks import BenchmarkSuite, SuiteMetadata

from .auction import onset_index
from .binance import bin_signed_trades
from .databento import RAW_DIR, _PX_SCALE, _instrument_symbols, et_epoch
from .liquidation import burst_episode

__all__ = ["FOMC_DAYS", "CONTROL_DAYS", "load_es_trades", "build_scheduled_suite"]

# 2026 FOMC statement days (second day of each two-day meeting through July)
# and control Wednesdays exactly one week earlier.
FOMC_DAYS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29"]
CONTROL_DAYS = ["2026-01-21", "2026-03-11", "2026-04-22", "2026-06-10", "2026-07-22"]
_STRUCTURES = {"fomc": "glbx_es_fomc", "ctrl": "glbx_es_control"}


def load_es_trades(day: str, kind: str, raw_dir: str | Path = RAW_DIR) -> tuple[pd.DataFrame, str]:
    """Front-month ES outright trades for one day.

    Args:
        day: ``YYYY-MM-DD`` (UTC file day; the 14:00 ET window sits inside it).
        kind: ``"fomc"`` or ``"ctrl"`` (file-name tag).
        raw_dir: Directory holding the DBN files.

    Returns:
        ``(frame, symbol)`` where the frame has ``price, quantity,
        transact_time`` (ms), ``sign`` and ``symbol`` is the chosen contract.

    Raises:
        FileNotFoundError: If the day's DBN file is missing.
    """
    import databento as db

    path = Path(raw_dir) / f"glbx_es_trades_{kind}_{day}.dbn.zst"
    if not path.exists():
        raise FileNotFoundError(path)
    store = db.DBNStore.from_file(path)
    sym_of = _instrument_symbols(store)
    arr = store.to_ndarray()
    iid = arr["instrument_id"].astype(np.int64)
    symbols = np.array([sym_of.get(i, str(i)) for i in iid])
    # Outrights only ("ESU6"); spreads carry a dash ("ESM6-ESU6").
    outright = np.array(["-" not in s for s in symbols])
    counts = pd.Series(symbols[outright]).value_counts()
    front = str(counts.index[0])
    keep = symbols == front
    sign = np.zeros(int(keep.sum()))
    side = arr["side"][keep]
    sign[side == b"B"] = 1.0
    sign[side == b"A"] = -1.0
    df = pd.DataFrame({
        "price": arr["price"][keep].astype(np.float64) * _PX_SCALE,
        "quantity": arr["size"][keep].astype(np.float64),
        "transact_time": arr["ts_event"][keep].astype(np.int64) // 1_000_000,
        "sign": sign,
    }).sort_values("transact_time", kind="stable").reset_index(drop=True)
    return df, front


def build_scheduled_suite(
    days: list[str],
    kind: str,
    bar_seconds: int = 60,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    n_queries: int = 5,
    seed: int = 0,
    announce_hh: int = 14,
    announce_mm: int = 0,
    raw_dir: str | Path = RAW_DIR,
) -> BenchmarkSuite:
    """One episode per day at the scheduled 14:00 ET window.

    The dose is the realized excess flow in the window (as for detected
    bursts) so that event and control episodes share the same encoding; the
    difference between the suites is only whether a statement was released.

    Args:
        days: Trading days.
        kind: ``"fomc"`` or ``"ctrl"``; selects files and the structure label.
        bar_seconds, pre_bars, post_bars, window_bars: Episode shape.
        n_queries: Post-onset queries per episode (several, since n_days is small).
        seed: Query RNG seed.
        announce_hh, announce_mm: Wall-clock ET onset.
        raw_dir: Directory holding the DBN files.

    Returns:
        BenchmarkSuite with structure ``glbx_es_fomc`` or ``glbx_es_control``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in _STRUCTURES:
        raise ValueError(f"kind must be one of {list(_STRUCTURES)}")
    rng = np.random.RandomState(seed)
    episodes = []
    for day in days:
        trades, symbol = load_es_trades(day, kind, raw_dir=raw_dir)
        bars = bin_signed_trades(trades["transact_time"].to_numpy() // 1000,
                                 trades["sign"].to_numpy(), trades["quantity"].to_numpy(),
                                 trades["price"].to_numpy(), bar_seconds=bar_seconds)
        idx = onset_index(bars, et_epoch(day, announce_hh, announce_mm), bar_seconds)
        if idx - pre_bars < 0 or idx + post_bars > len(bars):
            continue
        episodes.append(burst_episode(
            bars, idx, rng=rng, pre_bars=pre_bars, post_bars=post_bars,
            window_bars=window_bars, n_queries=n_queries, structure=_STRUCTURES[kind],
            extra_metadata={"symbol": symbol, "date": day, "kind": kind},
        ))
    meta = SuiteMetadata(
        name=f"glbx-es-{kind}", version="0.0.1", zenodo_record_id="", doi="",
        description=f"ES front-month, {announce_hh:02d}:{announce_mm:02d} ET window, {kind} days.",
        n_episodes=len(episodes), structures=(_STRUCTURES[kind],),
    )
    return BenchmarkSuite(meta, episodes)
