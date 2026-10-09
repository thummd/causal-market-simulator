#!/usr/bin/env python
"""Closing-auction tier: Nasdaq NOII imbalance as an announced soft intervention.

Two readouts on the same events:

1. **Dose-response slope** (model-free sanity check, the real-data analogue
   of the analytic slope tests): OLS of the 15:50-to-close price move on the
   standardized signed imbalance, with a bootstrap CI. If the announced
   imbalance does not move the close, the tier carries no causal signal
   and the PFN comparison is moot.
2. **Paired do-head delta**: every method predicts the realized price path
   after the 15:50 publication; the causal PFN's gain over its do-ablated
   twin measures whether reading the announced dose helps.

Two encodings (``--encoding``, see ``data/auction.py``): ``announced`` uses
the 15:50 publication as the dose with a 31-bar post window through the
cross; ``cross`` (law-matched) uses the final imbalance that actually sweeps
the book at 16:00, with the cross bar as the single query. 20-second bars
throughout. Requires the trade caches built by ``32_equity_transfer.py``.

Usage:
    python scripts/33_closing_auction.py --encoding announced
    python scripts/33_closing_auction.py --encoding cross --out output/closing_cross.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dotime.evaluation import bootstrap_ci
from dotime_market.data.auction import build_auction_suite
from dotime_market.data.databento import RAW_DIR, load_closing_imbalance, trading_days
from dotime_market.evaluation.transfer import build_models, paired_results, vol_split

EQ10 = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "INTC", "QQQ"]


def dose_response(table: pd.DataFrame, bootstrap: int, seed: int = 0) -> dict:
    """OLS slope of ``ret_bps`` on the standardized dose with a bootstrap CI.

    Args:
        table: Per-event frame from :func:`build_auction_suite`.
        bootstrap: Number of resamples.
        seed: Resampling seed.

    Returns:
        Dict with ``n``, ``slope_bps_per_sd`` and its CI, the sign agreement
        rate between imbalance and close move, and the mean signed move
        (``ret_bps`` oriented by the imbalance sign) with its CI.
    """
    x = table["dose"].to_numpy()
    y = table["ret_bps"].to_numpy()
    sign = np.sign(x)

    def slope(idx):
        xs, ys = x[idx], y[idx]
        xc = xs - xs.mean()
        return float((xc * (ys - ys.mean())).sum() / max((xc**2).sum(), 1e-12))

    rng = np.random.RandomState(seed)
    n = len(x)
    boots = [slope(rng.randint(0, n, size=n)) for _ in range(bootstrap)]
    m, _, lo, hi = bootstrap_ci([float(v) for v in sign * y], n=bootstrap)
    return {
        "n": int(n),
        "slope_bps_per_sd": slope(np.arange(n)),
        "slope_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "sign_agreement": float((np.sign(y) == sign).mean()),
        "mean_signed_move_bps": m,
        "mean_signed_move_ci": [lo, hi],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=EQ10)
    ap.add_argument("--start", default="2026-06-01", metavar="YYYY-MM-DD")
    ap.add_argument("--end", default="2026-08-31", metavar="YYYY-MM-DD")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--bar-seconds", type=int, default=20)
    ap.add_argument("--encoding", choices=["announced", "announced_channel", "cross", "announced_path"],
                default="announced")
    ap.add_argument("--price-source", choices=["trade", "mid"], default="trade")
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
    ap.add_argument("--out", default="output/closing_auction.json")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    dates = sorted({d for s in args.symbols
                    for d in trading_days(s, raw / "trades", args.start, args.end)})
    if not dates:
        raise SystemExit("no cached trading days — run scripts/32_equity_transfer.py first")
    noii = load_closing_imbalance(raw / "xnas_imbalance_eq50_12mo.dbn.zst",
                                  cache=raw / "closing_imbalance.parquet")
    print(f"building auction suite: {len(args.symbols)} names x {len(dates)} days ...", flush=True)
    suite, table = build_auction_suite(
        args.symbols, dates, bar_seconds=args.bar_seconds, n_queries=args.n_queries,
        cache_dir=raw / "trades", bbo_dir=(raw / "bbo") if args.price_source == "mid" else None,
        noii=noii, encoding=args.encoding,
    )
    print(f"{len(suite)} auction episodes ({args.encoding} encoding)", flush=True)
    if args.canon_width > 2 and suite[0].x_obs.shape[1] == 2:
        from dotime.benchmarks import BenchmarkSuite
        from dotime_market.data.liquidation import widen_to_canon
        suite = BenchmarkSuite(suite.meta, [widen_to_canon(ep, args.canon_width) for ep in suite])
    if len(suite) == 0:
        raise SystemExit("no auction episodes built")

    dr = dose_response(table, args.bootstrap)
    print(f"dose-response: slope {dr['slope_bps_per_sd']:+.3f} bps per SD-flow "
          f"[{dr['slope_ci'][0]:+.3f}, {dr['slope_ci'][1]:+.3f}]  sign agreement "
          f"{dr['sign_agreement']:.2f}  mean signed move {dr['mean_signed_move_bps']:+.2f} bps "
          f"[{dr['mean_signed_move_ci'][0]:+.2f}, {dr['mean_signed_move_ci'][1]:+.2f}]",
          flush=True)

    models = build_models(args.checkpoint, args.ablated_checkpoint, args.device,
                          tuple(args.extra_methods))
    results = {"args": vars(args), "n_episodes": len(suite), "dates": dates,
               "dose_response": dr, "table": table.to_dict(orient="list")}
    results["methods"] = paired_results(models, suite, bootstrap=args.bootstrap)
    results["vol_split"] = vol_split(results["methods"], suite, bootstrap=args.bootstrap)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
