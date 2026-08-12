#!/usr/bin/env python
"""Generic discrete-time DoT-PFN baseline .

Evaluates the public general-purpose discrete-time DoT-PFN checkpoints
(public checkpoints of the cited discrete-time model, loaded via the released
``dotime`` package's own baseline wrapper) on the market evaluation tiers.
The comparison isolates what the market prior buys: same paradigm
(interventional in-context prediction), same harness and episodes, but a
generic training prior instead of the confounded market prior. Regular
bar-time episodes are exactly the discrete model's home format, so time
handling is not the bottleneck in this comparison --- prior specialization
is.

Usage:
    python scripts/07_generic_baseline.py --tiers synthetic crypto
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime.evaluation import bootstrap_ci


def score(model, suite, n_boot: int = 10000) -> dict:
    signed = []
    for ep in suite:
        p = model.predict(ep).reshape(-1)
        y = ep.y_true.reshape(-1)
        sgn = math.copysign(1.0, float(ep.intervention.values))
        signed.extend(sgn * float(a - b) for a, b in zip(p, y))
    m, _, lo, hi = bootstrap_ci(signed, n=n_boot)
    return {
        "bias": m, "bias_ci": [lo, hi],
        "rmse": float(np.sqrt(np.mean(np.square(signed)))),
        "n": len(signed),
        "signed_errors": [float(v) for v in signed],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["synthetic", "crypto"],
                    choices=["synthetic", "crypto", "abides"])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--generic-checkpoints", nargs="+",
                    default=["checkpoints/hf_generic/s9btm_all_causal.pt",
                             "checkpoints/hf_generic/s9ho_all_causal.pt"])
    ap.add_argument("--continuous-generic-checkpoints", nargs="*", default=[],
                    help="continuous-time checkpoints trained on a generic prior "
                    "(the same-architecture clean control for the prior attribution)")
    ap.add_argument("--market-checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/generic_baseline.json")
    args = ap.parse_args()

    models = {
        f"generic-{Path(c).stem}": get("DoOverTimePFN", checkpoint=c, device=args.device)
        for c in args.generic_checkpoints
    }
    models["market-dotpfn"] = get("market-dotpfn", checkpoint=args.market_checkpoint,
                                  device=args.device)
    for c in args.continuous_generic_checkpoints:
        # same continuous-time architecture, generic (non-market) training prior
        models[f"generic-ct-{Path(c).parent.name}"] = get(
            "market-dotpfn", checkpoint=c, device=args.device)

    results = {"args": vars(args), "tiers": {}}

    if "synthetic" in args.tiers:
        from dotime_market.data.synthetic import build_market_suite
        from dotime_market.prior.market_scm import MarketDoTime
        rows = {}
        for g in args.gammas:
            prior = MarketDoTime(coupling_gamma=float(g), core_share=0.7, seed=0,
                                 impact_lambda=(0.25, 1.0), theta_range=(0.5, 1.0),
                                 sigma_range=(0.2, 0.6), num_substeps=2)
            suite = build_market_suite(prior, n_episodes=args.n_episodes, T=120, seed=0)
            rows[g] = {name: {k: v for k, v in score(m, suite).items()
                              if k != "signed_errors"}
                       for name, m in models.items()}
            print(f"synthetic g={g}: " + "  ".join(
                f"{n}: b{rows[g][n]['bias']:+.3f}/r{rows[g][n]['rmse']:.2f}"
                for n in rows[g]), flush=True)
        results["tiers"]["synthetic"] = rows

    if "crypto" in args.tiers:
        from datetime import date, timedelta
        from dotime_market.data.liquidation import build_binance_suite
        dates = [str(date(2026, 4, 1) + timedelta(d)) for d in range(91)]
        suite = build_binance_suite(symbol="BTCUSDT", dates=dates)
        row = {}
        for name, m in models.items():
            try:
                row[name] = {k: v for k, v in score(m, suite).items()
                             if k != "signed_errors"}
            except Exception as e:  # soft interventions may be unsupported
                row[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"crypto BTC {name}: {row[name]}", flush=True)
        results["tiers"]["crypto_BTCUSDT"] = row

    if "abides" in args.tiers:
        from dotime_market.data.abides_builder import build_abides_suite
        suite = build_abides_suite(n_episodes=1000, seed0=1000, intervention_kind="soft",
                                   num_noise_agents=500, num_value_agents=50)
        row = {}
        for name, m in models.items():
            try:
                row[name] = {k: v for k, v in score(m, suite).items()
                             if k != "signed_errors"}
            except Exception as e:
                row[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"abides {name}: {row[name]}", flush=True)
        results["tiers"]["abides_heldout"] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
