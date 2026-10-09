#!/usr/bin/env python
"""Equity transfer: detected flow-burst episodes on Nasdaq ITCH trades.

The traditional-finance replication of ``04_crypto_transfer.py``: the same
burst detector, episode encoder, method set, and paired do-head readout,
applied to the 10-name Nasdaq panel pulled by ``31_databento_pull.py``.
Real data has no counterfactual, so ``y_true`` is the realized post-burst
price and the causal readout is the paired delta between the causal PFN and
its do-ablated twin on identical episodes.

First run builds the per-symbol-day parquet caches from the monthly DBN
files (a few minutes; idempotent afterwards).

Usage:
    python scripts/32_equity_transfer.py --start 2026-06-01 --end 2026-08-31 \
        --checkpoint checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt
    python scripts/32_equity_transfer.py --price-source mid   # 1-second mid outcome
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotime_market.data.databento import (
    RAW_DIR,
    build_bbo_cache,
    build_databento_suite,
    build_trade_cache,
    trading_days,
)
from dotime_market.evaluation.transfer import build_models, paired_results, vol_split

EQ10 = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "INTC", "QQQ"]


def ensure_caches(raw_dir: Path, need_bbo: bool) -> None:
    """Split every monthly DBN file under ``raw_dir`` that is not yet cached."""
    for f in sorted(raw_dir.glob("xnas_trades_*.dbn.zst")):
        n = build_trade_cache(f, cache_dir=raw_dir / "trades")
        if n:
            print(f"cached {f.name}: {len(n)} symbol-days", flush=True)
    if need_bbo:
        for f in sorted(raw_dir.glob("xnas_bbo1s_*.dbn.zst")):
            n = build_bbo_cache(f, cache_dir=raw_dir / "bbo")
            if n:
                print(f"cached {f.name}: {len(n)} symbol-days", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=EQ10)
    ap.add_argument("--start", default="2026-06-01", metavar="YYYY-MM-DD")
    ap.add_argument("--end", default="2026-08-31", metavar="YYYY-MM-DD")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--bar-seconds", type=int, default=60)
    ap.add_argument("--z-thresh", type=float, default=5.0)
    ap.add_argument("--price-source", choices=["trade", "mid"], default="trade",
                    help="outcome price: last trade (Binance parity) or 1-second mid")
    ap.add_argument("--n-queries", type=int, default=1)
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--canon-width", type=int, default=2,
                    help="re-lay (flow, price) episodes into an N-column canon (price last) for "
                         "checkpoints trained with extra observed channels, e.g. 4 for the announced prior")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--extra-methods", nargs="*", default=[], metavar="NAME")
    ap.add_argument("--out", default="output/equity_transfer.json")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    ensure_caches(raw, need_bbo=args.price_source == "mid")
    # Union of cached days across the panel; symbol-days missing for one name are skipped.
    dates = sorted({d for s in args.symbols
                    for d in trading_days(s, raw / "trades", args.start, args.end)})
    print(f"building burst suite: {len(args.symbols)} names x {len(dates)} days ...", flush=True)
    suite = build_databento_suite(
        args.symbols, dates, bar_seconds=args.bar_seconds, cache_dir=raw / "trades",
        bbo_dir=(raw / "bbo") if args.price_source == "mid" else None,
        z_thresh=args.z_thresh, n_queries=args.n_queries,
    )
    print(f"{len(suite)} burst episodes detected", flush=True)
    if args.canon_width > 2 and suite[0].x_obs.shape[1] == 2:
        from dotime.benchmarks import BenchmarkSuite
        from dotime_market.data.liquidation import widen_to_canon
        suite = BenchmarkSuite(suite.meta, [widen_to_canon(ep, args.canon_width) for ep in suite])
    if len(suite) == 0:
        raise SystemExit("no bursts detected — lower --z-thresh or widen the date range")
    per_symbol = {}
    for ep in suite:
        per_symbol[ep.metadata["symbol"]] = per_symbol.get(ep.metadata["symbol"], 0) + 1
    print("episodes per symbol:", per_symbol, flush=True)

    models = build_models(args.checkpoint, args.ablated_checkpoint, args.device,
                          tuple(args.extra_methods))
    results = {"args": vars(args), "n_episodes": len(suite), "dates": dates,
               "episodes_per_symbol": per_symbol}
    results["methods"] = paired_results(models, suite, bootstrap=args.bootstrap)
    results["vol_split"] = vol_split(results["methods"], suite, bootstrap=args.bootstrap)
    results["price_scales"] = [float(ep.metadata["price_scale"]) for ep in suite]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
