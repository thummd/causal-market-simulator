"""Databento loader: Nasdaq ITCH / CME trades -> the shared per-bar flow frame.

Data arrives as DBN files pulled by ``scripts/31_databento_pull.py`` into
``data/raw/databento/`` (gitignored). This module turns them into the same
per-trade frame the Binance adapter produces (``price, quantity,
transact_time`` in ms, plus an aggressor ``sign``) and bins them with the
venue-agnostic :func:`~.binance.bin_signed_trades`, so the burst detector,
the episode encoder, and the evaluation harness run unchanged on equities
and futures.

Sign convention: Databento's trade ``side`` is the aggressor side (``B`` buy
aggressor -> +1, ``A`` sell aggressor -> -1). Nasdaq reports non-displayed
executions without a side (``N`` -> 0); those count in ``volume`` and
``n_trades`` but not in ``flow``. Verified 2026-09-15: ~69% of AAPL Nasdaq
trades carry a side.

Layout of the parquet cache (built once from the monthly DBN files, so a
symbol-day never re-parses a multi-GB file):

    data/raw/databento/trades/{SYMBOL}-trades-{YYYY-MM-DD}.parquet
    data/raw/databento/bbo/{SYMBOL}-bbo1s-{YYYY-MM-DD}.parquet
    data/raw/databento/closing_imbalance.parquet

Trading days are Eastern-time sessions; the trades cache keeps
09:30:00 <= t < 16:00:30 ET so the 16:00 closing-cross print is the last
record of each day.
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dotime.benchmarks import BenchmarkSuite, SuiteMetadata

from .binance import bin_signed_trades
from .liquidation import burst_episode, detect_flow_bursts

__all__ = [
    "RAW_DIR", "ET", "build_trade_cache", "build_bbo_cache", "load_trades", "load_bbo",
    "bin_trades", "trading_days", "load_closing_imbalance", "load_statistics",
    "build_databento_suite", "et_epoch",
]

RAW_DIR = Path("data/raw/databento")
ET = ZoneInfo("America/New_York")
_PX_SCALE = 1e-9  # DBN fixed-point prices: 1 unit = 1e-9
_SESSION_START = 9 * 3600 + 30 * 60
_SESSION_END = 16 * 3600 + 30  # keep the 16:00:00.x closing-cross print
_STRUCTURE = "xnas_flow_burst"


def et_epoch(date: str, hh: int, mm: int = 0, ss: int = 0) -> int:
    """Epoch seconds of a wall-clock Eastern time on ``date`` (DST-aware).

    Args:
        date: ``YYYY-MM-DD`` trading day.
        hh: Hour (24h), ``mm``/``ss`` minute and second.

    Returns:
        Integer epoch seconds.
    """
    ts = pd.Timestamp(f"{date} {hh:02d}:{mm:02d}:{ss:02d}", tz=ET)
    return int(ts.timestamp())


def _instrument_symbols(store) -> dict[int, str]:
    """Map instrument ids -> raw symbols from the DBN store's own metadata."""
    out: dict[int, str] = {}
    for symbol, intervals in store.metadata.mappings.items():
        for iv in intervals:
            if iv["symbol"]:
                out[int(iv["symbol"])] = symbol
    return out


def _et_day_and_secs(ts_ns: np.ndarray) -> tuple[pd.Categorical, np.ndarray]:
    """Eastern-time trading day (``YYYY-MM-DD``) and seconds-of-day per timestamp.

    Returns the day as a Categorical: formatting is done once per unique day
    rather than once per row, which matters at ~40M rows per month.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(ts_ns, unit="ns", utc=True)).tz_convert(ET)
    day = idx.normalize()
    secs = (idx - day).total_seconds().to_numpy()
    codes, uniques = pd.factorize(day.asi8)
    labels = pd.DatetimeIndex(uniques, tz="UTC").tz_convert(ET).strftime("%Y-%m-%d")
    return pd.Categorical.from_codes(codes, categories=labels), secs


def _symbol_categorical(iid: np.ndarray, sym_of: dict[int, str]) -> pd.Categorical:
    """Instrument ids -> symbol Categorical via the unique ids only (no per-row Python)."""
    codes, uniques = pd.factorize(iid)
    labels = [sym_of.get(int(i), str(i)) for i in uniques]
    return pd.Categorical.from_codes(codes, categories=pd.Index(labels))


def build_trade_cache(dbn_path: str | Path, cache_dir: str | Path = RAW_DIR / "trades",
                      force: bool = False) -> list[Path]:
    """Split one monthly trades DBN file into per-symbol-day parquet files.

    Uses ``to_ndarray`` rather than ``to_df`` because a month of Nasdaq
    trades for ten names is ~40M rows; the structured array keeps the peak
    under ~4 GB where the pandas conversion would not.

    Args:
        dbn_path: ``xnas_trades_*.dbn.zst`` (or any trades-schema file).
        cache_dir: Output directory for the parquet files.
        force: Rebuild even if the marker file says this DBN was already split.

    Returns:
        Paths written (empty if the marker existed and ``force`` is False).

    Raises:
        ValueError: If the file is not the trades schema.
    """
    import databento as db  # local import: the package is an optional extra

    dbn_path = Path(dbn_path)
    cache_dir = Path(cache_dir)
    marker = cache_dir / f".{dbn_path.stem}.done"
    if marker.exists() and not force:
        return []
    store = db.DBNStore.from_file(dbn_path)
    if str(store.metadata.schema) != "trades":
        raise ValueError(f"{dbn_path} is schema {store.metadata.schema}, expected trades")
    sym_of = _instrument_symbols(store)
    arr = store.to_ndarray()
    ts = arr["ts_event"].astype(np.int64)
    iid = arr["instrument_id"].astype(np.int64)
    price = arr["price"].astype(np.float64) * _PX_SCALE
    qty = arr["size"].astype(np.float64)
    side = arr["side"]
    del arr
    sign = np.zeros(len(side))
    sign[side == b"B"] = 1.0
    sign[side == b"A"] = -1.0
    day, secs = _et_day_and_secs(ts)
    keep = (secs >= _SESSION_START) & (secs < _SESSION_END)
    df = pd.DataFrame({
        "symbol": _symbol_categorical(iid[keep], sym_of),
        "day": day[keep],
        "price": price[keep],
        "quantity": qty[keep],
        "transact_time": ts[keep] // 1_000_000,  # ns -> ms, matching Binance
        "sign": sign[keep],
    })
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (symbol, d), g in df.groupby(["symbol", "day"], observed=True, sort=True):
        out = cache_dir / f"{symbol}-trades-{d}.parquet"
        g.drop(columns=["symbol", "day"]).sort_values("transact_time", kind="stable") \
            .reset_index(drop=True).to_parquet(out)
        written.append(out)
    marker.write_text("\n".join(str(p) for p in written))
    return written


def build_bbo_cache(dbn_path: str | Path, cache_dir: str | Path = RAW_DIR / "bbo",
                    force: bool = False) -> list[Path]:
    """Split one monthly ``bbo-1s`` DBN file into per-symbol-day mid-price parquets.

    Args:
        dbn_path: ``xnas_bbo1s_*.dbn.zst``.
        cache_dir: Output directory.
        force: Rebuild even if already split.

    Returns:
        Paths written. Each parquet has ``time`` (epoch seconds of the
        1-second sample) and ``mid``.
    """
    import databento as db

    dbn_path = Path(dbn_path)
    cache_dir = Path(cache_dir)
    marker = cache_dir / f".{dbn_path.stem}.done"
    if marker.exists() and not force:
        return []
    store = db.DBNStore.from_file(dbn_path)
    sym_of = _instrument_symbols(store)
    arr = store.to_ndarray()
    ts = arr["ts_recv"].astype(np.int64)
    iid = arr["instrument_id"].astype(np.int64)
    bid = arr["bid_px_00"].astype(np.float64) * _PX_SCALE
    ask = arr["ask_px_00"].astype(np.float64) * _PX_SCALE
    del arr
    # A one-sided book (no bid or no ask) has no mid; drop rather than fake it.
    ok = (bid > 0) & (ask > 0) & (ask < 1e6) & (bid < 1e6)
    day, secs = _et_day_and_secs(ts)
    keep = ok & (secs >= _SESSION_START) & (secs < _SESSION_END)
    df = pd.DataFrame({
        "symbol": _symbol_categorical(iid[keep], sym_of),
        "day": day[keep],
        "time": ts[keep] // 1_000_000_000,
        "mid": 0.5 * (bid[keep] + ask[keep]),
    })
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (symbol, d), g in df.groupby(["symbol", "day"], observed=True, sort=True):
        out = cache_dir / f"{symbol}-bbo1s-{d}.parquet"
        g.drop(columns=["symbol", "day"]).sort_values("time", kind="stable") \
            .reset_index(drop=True).to_parquet(out)
        written.append(out)
    marker.write_text("\n".join(str(p) for p in written))
    return written


def load_trades(symbol: str, date: str, cache_dir: str | Path = RAW_DIR / "trades") -> pd.DataFrame:
    """Load one cached symbol-day of trades.

    Args:
        symbol: Raw symbol, e.g. ``"AAPL"``.
        date: ``YYYY-MM-DD`` Eastern trading day.
        cache_dir: Parquet cache root.

    Returns:
        Frame with ``price, quantity, transact_time`` (ms), ``sign``.

    Raises:
        FileNotFoundError: If the day is not cached (run :func:`build_trade_cache`).
    """
    path = Path(cache_dir) / f"{symbol}-trades-{date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — build the cache with build_trade_cache() first")
    return pd.read_parquet(path)


def load_bbo(symbol: str, date: str, cache_dir: str | Path = RAW_DIR / "bbo") -> pd.DataFrame:
    """Load one cached symbol-day of 1-second mids (``time`` epoch s, ``mid``).

    Raises:
        FileNotFoundError: If the day is not cached (run :func:`build_bbo_cache`).
    """
    path = Path(cache_dir) / f"{symbol}-bbo1s-{date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — build the cache with build_bbo_cache() first")
    return pd.read_parquet(path)


def bin_trades(df: pd.DataFrame, bar_seconds: int = 60,
               bbo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bin a Databento trade frame into the shared per-bar flow/price frame.

    Args:
        df: Output of :func:`load_trades` (needs ``price, quantity,
            transact_time`` in ms and ``sign``).
        bar_seconds: Bar width.
        bbo: Optional 1-second mid frame from :func:`load_bbo`. When given,
            ``price_bps`` is rebuilt from the last mid at or before each bar
            end (forward-filled), which removes bid-ask bounce from the
            outcome; ``flow``/``volume`` are unaffected.

    Returns:
        Frame with ``time, flow, price_bps, volume, n_trades`` as documented
        on :func:`~.binance.bin_signed_trades`.
    """
    t = df["transact_time"].to_numpy() // 1000
    bars = bin_signed_trades(t, df["sign"].to_numpy(), df["quantity"].to_numpy(),
                             df["price"].to_numpy(), bar_seconds=bar_seconds)
    if bbo is not None and len(bbo):
        # Bar i spans [end_i - bar_seconds, end_i); the mid "as of" the bar end
        # is the last sample strictly before it, i.e. searchsorted(side='left') - 1.
        bt = bbo["time"].to_numpy()
        pos = np.searchsorted(bt, bars["time"].to_numpy(), side="left") - 1
        mid = np.where(pos >= 0, bbo["mid"].to_numpy()[np.clip(pos, 0, None)], np.nan)
        s = pd.Series(mid).ffill().bfill().to_numpy()
        bars["price_bps"] = 1e4 * np.log(s / s[0])
    return bars


def trading_days(symbol: str, cache_dir: str | Path = RAW_DIR / "trades",
                 start: str | None = None, end: str | None = None) -> list[str]:
    """Cached trading days for ``symbol`` (optionally within ``[start, end]``).

    Args:
        symbol: Raw symbol.
        cache_dir: Parquet cache root.
        start: Inclusive ``YYYY-MM-DD`` lower bound.
        end: Inclusive upper bound.

    Returns:
        Sorted list of ``YYYY-MM-DD`` strings.
    """
    days = sorted(p.stem.split("-trades-")[1] for p in Path(cache_dir).glob(f"{symbol}-trades-*.parquet"))
    return [d for d in days if (start is None or d >= start) and (end is None or d <= end)]


def load_closing_imbalance(dbn_path: str | Path = RAW_DIR / "xnas_imbalance_eq50_12mo.dbn.zst",
                           cache: str | Path = RAW_DIR / "closing_imbalance.parquet",
                           force: bool = False) -> pd.DataFrame:
    """Closing-cross NOII messages (``auction_type == 'C'``) as a flat frame.

    Nasdaq disseminates the closing-cross Net Order Imbalance Indicator from
    15:50 ET (every 10 s, then every 1 s from 15:55) until the 16:00 cross.
    The frame keeps every message so callers can pick the first publication
    (the ex-ante intervention) or the final one (what actually crossed).

    Args:
        dbn_path: The imbalance DBN file.
        cache: Parquet cache path.
        force: Rebuild the cache.

    Returns:
        Frame with ``symbol, day, secs`` (ET seconds of day), ``signed_imbalance``
        (shares; buy-side imbalance positive, unsigned ``N`` -> 0),
        ``total_imbalance_qty, paired_qty, ref_price, ind_match_price``.
    """
    import databento as db

    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    df = db.DBNStore.from_file(dbn_path).to_df()
    df = df[df["auction_type"] == "C"]
    ts = df.index.tz_convert(ET)
    day = ts.normalize()
    side = df["side"].map({"B": 1.0, "A": -1.0}).fillna(0.0).to_numpy()
    out = pd.DataFrame({
        "symbol": df["symbol"].to_numpy(),
        "day": day.strftime("%Y-%m-%d").to_numpy(),
        "secs": (ts - day).total_seconds().to_numpy(),
        "signed_imbalance": side * df["total_imbalance_qty"].to_numpy(),
        "total_imbalance_qty": df["total_imbalance_qty"].to_numpy(),
        "paired_qty": df["paired_qty"].to_numpy(),
        "ref_price": df["ref_price"].to_numpy(),
        "ind_match_price": df["ind_match_price"].to_numpy(),
    }).sort_values(["symbol", "day", "secs"], kind="stable").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def load_statistics(dbn_path: str | Path = RAW_DIR / "xnas_statistics_all_2026q3.dbn.zst",
                    symbology: str | Path = RAW_DIR / "xnas_statistics_symbology.json",
                    stat_types: tuple[int, ...] = (1, 11)) -> pd.DataFrame:
    """Official Nasdaq statistics with symbols resolved from the symbology map.

    The ALL_SYMBOLS pull carries instrument ids only; the symbology endpoint
    result (``xnas_statistics_symbology.json``, id -> dated intervals) is
    applied per day because 393 ids changed symbol inside the window.

    Args:
        dbn_path: Statistics DBN file.
        symbology: JSON produced by ``client.symbology.resolve``.
        stat_types: ``StatType`` codes to keep (1 = opening price, 11 = close).

    Returns:
        Frame with ``symbol, day, stat_type, price``.
    """
    import databento as db

    df = db.DBNStore.from_file(dbn_path).to_df()
    df = df[df["stat_type"].isin(stat_types)]
    ts = df.index.tz_convert(ET)
    day = ts.normalize().strftime("%Y-%m-%d").to_numpy()
    mapping = json.loads(Path(symbology).read_text())
    iid = df["instrument_id"].astype(int).to_numpy()
    symbols = np.empty(len(df), dtype=object)
    for k, (i, d) in enumerate(zip(iid, day)):
        symbols[k] = next((iv["s"] for iv in mapping.get(str(i), []) if iv["d0"] <= d < iv["d1"]),
                          None)
    out = pd.DataFrame({
        "symbol": symbols, "day": day, "stat_type": df["stat_type"].to_numpy(),
        "price": df["price"].to_numpy(),
    })
    return out.dropna(subset=["symbol"]).reset_index(drop=True)


def build_databento_suite(
    symbols: list[str],
    dates: list[str],
    bar_seconds: int = 60,
    cache_dir: str | Path = RAW_DIR / "trades",
    bbo_dir: str | Path | None = None,
    z_thresh: float = 5.0,
    pre_bars: int = 90,
    post_bars: int = 30,
    window_bars: int = 3,
    n_queries: int = 1,
    seed: int = 0,
    name: str = "xnas-flow-bursts",
) -> BenchmarkSuite:
    """Detected-burst episodes over a symbol panel and a list of trading days.

    Mirrors :func:`~.liquidation.build_binance_suite`; the only differences
    are the loader and the optional mid-price outcome.

    Args:
        symbols: Raw symbols in the trades cache.
        dates: ``YYYY-MM-DD`` trading days; days missing for a symbol are skipped.
        bar_seconds: Bar width.
        cache_dir: Trades parquet cache.
        bbo_dir: If given, outcomes use the 1-second mid from this cache.
        z_thresh, pre_bars, post_bars, window_bars: Detector / episode settings.
        n_queries: Post-onset queries per episode.
        seed: Query-time RNG seed.
        name: Suite name.

    Returns:
        In-memory BenchmarkSuite; each episode's metadata carries ``symbol`` and ``date``.
    """
    rng = np.random.RandomState(seed)
    episodes = []
    for symbol in symbols:
        for date in dates:
            try:
                trades = load_trades(symbol, date, cache_dir=cache_dir)
            except FileNotFoundError:
                continue
            bbo = load_bbo(symbol, date, cache_dir=bbo_dir) if bbo_dir is not None else None
            bars = bin_trades(trades, bar_seconds=bar_seconds, bbo=bbo)
            for idx in detect_flow_bursts(bars, z_thresh=z_thresh, pre_bars=pre_bars,
                                          post_bars=post_bars, window_bars=window_bars):
                episodes.append(burst_episode(
                    bars, idx, rng=rng, pre_bars=pre_bars, post_bars=post_bars,
                    window_bars=window_bars, n_queries=n_queries, structure=_STRUCTURE,
                    extra_metadata={"symbol": symbol, "date": date},
                ))
    meta = SuiteMetadata(
        name=name, version="0.0.1", zenodo_record_id="", doi="",
        description=f"Nasdaq ITCH extreme one-sided flow bursts, {len(symbols)} names.",
        n_episodes=len(episodes), structures=(_STRUCTURE,),
    )
    return BenchmarkSuite(meta, episodes)
