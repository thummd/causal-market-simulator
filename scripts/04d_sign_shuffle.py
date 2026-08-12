#!/usr/bin/env python
"""Sign-shuffle null for the crypto burst tier .

Adopted from Maitrier, Loeper & Bouchaud (arXiv:2503.18199), whose core sanity
check is that shuffling trade signs must send measured impact to zero ---
impact is a signed, direction-locked quantity, not repackaged volatility.

Two readouts on the detected-burst episodes:

1. Data-side null: the measured post-event move signed by the TRUE burst
   direction, ``E[sign(c) * (y_q - y_pre)]``, versus the same statistic under
   randomly flipped signs. True signs must give the large directional move the
   paper reports; flipped signs must give ~0.
2. Model-side: causal and ablated bias when episodes carry randomly flipped
   intervention signs. The ablated twin ignores the intervention inputs, so
   its flipped-sign bias must be ~0 (the direction label is random relative
   to the realized outcome); the causal model follows the (now wrong half the
   time) label, so its response manufactures error --- the same signature as
   the anchor-placebo test, from the sign axis instead of the timing axis.

Usage:
    python scripts/04d_sign_shuffle.py --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime.evaluation import bootstrap_ci
from dotime_market.data.liquidation import build_binance_suite


def signed_stats(episodes, model=None) -> tuple[float, float, float]:
    """Mean, CI of sign(c)*(pred_or_realized - reference) over episodes."""
    vals = []
    for ep in episodes:
        sgn = math.copysign(1.0, float(ep.intervention.values))
        if model is None:
            # data-side: realized outcome vs last pre-onset price
            onset = min(ep.intervention.times)
            y = float(ep.y_true.reshape(-1)[0])
            ref = float(ep.x_obs[onset - 1, 1])
            vals.append(sgn * (y - ref))
        else:
            p = float(model.predict(ep).reshape(-1)[0])
            y = float(ep.y_true.reshape(-1)[0])
            vals.append(sgn * (p - y))
    m, _, lo, hi = bootstrap_ci(vals, n=10000)
    return m, lo, hi


def flip_signs(episodes, rng) -> list:
    out = []
    for ep in episodes:
        ep2 = copy.deepcopy(ep)
        if rng.rand() < 0.5:
            ep2.intervention.values = -float(ep2.intervention.values)
        out.append(ep2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--out", default="output/sign_shuffle.json")
    args = ap.parse_args()

    dates = [str(date(2026, 4, 1) + timedelta(d)) for d in range(91)]
    episodes = list(build_binance_suite(symbol=args.symbol, dates=dates))
    rng = np.random.RandomState(args.seed)
    flipped = flip_signs(episodes, rng)

    causal = get("market-dotpfn", checkpoint=args.checkpoint)
    ablated = get("ablated-dotpfn", checkpoint=args.ablated_checkpoint)

    results = {"args": vars(args), "n": len(episodes)}
    results["impact_true_signs"] = signed_stats(episodes)
    results["impact_shuffled_signs"] = signed_stats(flipped)
    results["causal_bias_true"] = signed_stats(episodes, causal)
    results["causal_bias_shuffled"] = signed_stats(flipped, causal)
    results["ablated_bias_true"] = signed_stats(episodes, ablated)
    results["ablated_bias_shuffled"] = signed_stats(flipped, ablated)

    for k, v in results.items():
        if isinstance(v, tuple):
            print(f"{k:26s} {v[0]:+.3f}  [{v[1]:+.3f}, {v[2]:+.3f}]")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
