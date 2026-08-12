#!/usr/bin/env python
"""Phase 4 — crypto transfer: detected flow-burst episodes from Binance.

Real data has no counterfactual, so methods predict the *realized* post-burst
price given the burst's signed size (see data/liquidation.py). The causal
readout is the paired delta between the causal PFN and its do-ablated twin
on identical episodes — does the do-information (burst size/direction)
improve factual prediction? — reported with bootstrap CIs like the ABIDES
study.

Downloads aggTrades daily zips on first use (cached under data/raw/binance).

Usage:
    python scripts/04_crypto_transfer.py --symbol BTCUSDT \
        --dates 2026-06-01 2026-06-02 2026-06-03 \
        --checkpoint checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime.evaluation import bootstrap_ci
from dotime_market.data.liquidation import build_binance_suite


def score(model, suite) -> dict:
    preds, targets, signs = [], [], []
    for ep in suite:
        p = model.predict(ep).reshape(-1)
        y = ep.y_true.reshape(-1)
        c = float(ep.intervention.values)
        preds.append(p)
        targets.append(y)
        signs.extend([math.copysign(1.0, c)] * y.numel())
    pred = torch.cat(preds).double()
    target = torch.cat(targets).double()
    sign = torch.tensor(signs, dtype=torch.float64)
    err = pred - target
    signed = sign * err
    return {
        "rmse": float(torch.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(signed.mean()),
        "n": int(err.numel()),
        "signed_errors": [float(v) for v in signed],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dates", nargs="+", required=True, metavar="YYYY-MM-DD")
    ap.add_argument("--bar-seconds", type=int, default=60)
    ap.add_argument("--z-thresh", type=float, default=5.0)
    ap.add_argument("--cache-dir", default="data/raw/binance")
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="output/crypto_transfer.json")
    args = ap.parse_args()

    print(f"building burst suite: {args.symbol}, {len(args.dates)} day(s) ...", flush=True)
    suite = build_binance_suite(
        symbol=args.symbol,
        dates=args.dates,
        bar_seconds=args.bar_seconds,
        cache_dir=args.cache_dir,
        z_thresh=args.z_thresh,
    )
    print(f"{len(suite)} burst episodes detected", flush=True)
    if len(suite) == 0:
        raise SystemExit("no bursts detected — lower --z-thresh or add dates")

    models = {
        "Mean": get("Mean"),
        "AR1": get("AR1"),
        "AlmgrenChriss": get("AlmgrenChriss"),
        "OWPropagator": get("OWPropagator"),
        "ReturnRegression": get("ReturnRegression"),
        "SquareRootLaw": get("SquareRootLaw"),
        "market-dotpfn": get("market-dotpfn", checkpoint=args.checkpoint, device=args.device),
        "ablated-dotpfn": get(
            "ablated-dotpfn", checkpoint=args.ablated_checkpoint, device=args.device
        ),
    }
    results = {"args": vars(args), "n_episodes": len(suite), "methods": {}}
    # Local-vol proxy per episode: the pre-onset price std used as the
    # episode's standardization scale (metadata) — split at the median.
    results["price_scales"] = [float(ep.metadata["price_scale"]) for ep in suite]
    ref = "ablated-dotpfn"
    for name, model in models.items():
        results["methods"][name] = score(model, suite)

    print(f"{'method':>16s}  {'rmse':>8s}  {'mae':>8s}  {'bias':>8s}  "
          f"{'d_vs_ablated':>12s}  {'95% CI':>20s}")
    for name in models:
        s = results["methods"][name]
        if name != ref:
            deltas = [a - b for a, b in zip(s["signed_errors"],
                                            results["methods"][ref]["signed_errors"])]
            dm, _, dlo, dhi = bootstrap_ci(deltas, n=args.bootstrap)
            s.update({"delta_vs_ablated": dm, "delta_ci": [dlo, dhi]})
            d_str = f"{dm:12.4f}  [{dlo:8.4f}, {dhi:8.4f}]"
        else:
            d_str = f"{'--':>12s}"
        print(f"{name:>16s}  {s['rmse']:8.4f}  {s['mae']:8.4f}  {s['bias']:8.4f}  {d_str}",
              flush=True)

    # Robustness split: calm vs stressed by median pre-onset vol (price_scale).
    import numpy as np

    scales = np.array(results["price_scales"])
    median = float(np.median(scales))
    stressed = scales > median
    results["vol_split"] = {"median_price_scale_bps": median, "regimes": {}}
    pfn = results["methods"]["market-dotpfn"]["signed_errors"]
    abl = results["methods"][ref]["signed_errors"]
    deltas = np.array(pfn) - np.array(abl)
    print(f"\nvol split (median pre-onset sigma = {median:.2f} bps):")
    for regime, mask in (("calm", ~stressed), ("stressed", stressed)):
        dm, _, dlo, dhi = bootstrap_ci([float(d) for d in deltas[mask]], n=args.bootstrap)
        results["vol_split"]["regimes"][regime] = {
            "n": int(mask.sum()), "delta_vs_ablated": dm, "delta_ci": [dlo, dhi],
        }
        print(f"  {regime:9s} n={int(mask.sum()):4d}  do-head delta {dm:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
