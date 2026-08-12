"""Binance public-dump loader: signed aggTrades -> per-bar flow/price series.

Data source: https://data.binance.vision (public S3 bucket, no auth) —
``data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip``,
CSV with header ``agg_trade_id, price, quantity, first_trade_id,
last_trade_id, transact_time (ms), is_buyer_maker``.

Sign convention: ``is_buyer_maker == true`` means the buyer was the passive
side, i.e. the *aggressor sold* -> negative signed flow; ``false`` is an
aggressive buy -> positive. Flow is in base-asset units per bar.

NOTE (verified 2026-07-13): the ``liquidationSnapshot`` dataset the
architecture doc planned around has been REMOVED from the public dumps
(404 for all daily/monthly paths; absent from the S3 prefix listing).
Liquidation-style events are therefore *detected* from aggTrades as extreme
one-sided flow bursts — see :mod:`.liquidation`.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["download_agg_trades", "bin_agg_trades"]

_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"


def download_agg_trades(
    symbol: str,
    date: str,
    cache_dir: str | Path = "data/raw/binance",
) -> pd.DataFrame:
    """Fetch one day of aggTrades (cached as parquet after first download).

    Parameters
    ----------
    symbol : str
        e.g. ``"BTCUSDT"`` (USD-M futures).
    date : str
        ``YYYY-MM-DD``.
    cache_dir : path
        Local cache root (gitignored under ``data/raw/``).
    """
    cache = Path(cache_dir) / f"{symbol}-aggTrades-{date}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{_BASE}/{symbol}/{symbol}-aggTrades-{date}.zip"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(
                f,
                usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
                dtype={
                    "price": np.float64,
                    "quantity": np.float64,
                    "is_buyer_maker": bool,
                },
            )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def bin_agg_trades(df: pd.DataFrame, bar_seconds: int = 60) -> pd.DataFrame:
    """Aggregate trades into fixed bars of signed flow and log price.

    Returns a DataFrame indexed 0..T-1 with columns:

    - ``time``      : bar-end epoch seconds
    - ``flow``      : signed base-asset volume (aggressive buys − sells)
    - ``price_bps`` : ``1e4 * log(last_price / first_bar_last_price)``
    - ``volume``    : unsigned base-asset volume (for participation rates)

    Empty bars forward-fill the price and carry zero flow/volume.
    """
    t = df["transact_time"].to_numpy() // 1000  # ms -> s
    bar = (t - t[0]) // bar_seconds
    sign = np.where(df["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    qty = df["quantity"].to_numpy()
    price = df["price"].to_numpy()

    n_bars = int(bar[-1]) + 1
    flow = np.zeros(n_bars)
    volume = np.zeros(n_bars)
    last_price = np.full(n_bars, np.nan)
    np.add.at(flow, bar.astype(int), sign * qty)
    np.add.at(volume, bar.astype(int), qty)
    n_trades = np.zeros(n_bars)
    np.add.at(n_trades, bar.astype(int), 1.0)
    last_price[bar.astype(int)] = price  # last write per bar wins (time-ordered)

    # Forward-fill empty bars' price.
    mask = np.isnan(last_price)
    if mask.any():
        idx = np.where(~mask, np.arange(n_bars), 0)
        np.maximum.accumulate(idx, out=idx)
        last_price = last_price[idx]

    return pd.DataFrame(
        {
            "time": t[0] + bar_seconds * (np.arange(n_bars) + 1),
            "flow": flow,
            "price_bps": 1e4 * np.log(last_price / last_price[0]),
            "volume": volume,
            "n_trades": n_trades,
        }
    )
