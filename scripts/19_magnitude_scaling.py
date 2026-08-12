"""Magnitude-scaling: soft-trained response vs. encoded intervention magnitude.

Discriminates the two readings of the soft-trained transformer's collapsed
BTC do-head response :

- *Global scale shrinkage* (the paper's prior-scale hypothesis): the paired
  do-head response should scale ~linearly with the encoded intervention
  magnitude, with a small slope — the response pathway works, its gain is
  calibrated to the prior's λ-scale.
- *Lost magnitude sensitivity*: the response should be ~flat in the encoded
  magnitude — the soft pathway ignores the intervention's size.

Protocol: real BTC burst episodes (same detector as the paper's crypto
tier), each episode's intervention value scaled by a factor
f ∈ {0.25, 0.5, 1, 2, 4}; paired do-head delta (causal − ablated, signed
toward the intervention) recorded per factor, for BOTH the soft-trained
pair (v4s) and the hard-trained headline pair (v4) as the reference that
shows normal scaling.

Args:
    --days N     number of BTC days from 2026-04-01 (default 20).
    --out PATH   output JSON (default output/magnitude_scaling_v4s.json).

Returns (writes): per-factor mean paired delta + episode-bootstrap 95% CI
    per model pair, plus a per-pair OLS slope of delta on factor.

Raises:
    SystemExit: if no burst episodes are detected.
"""
import argparse
import copy
import json
import math
from datetime import date, timedelta

import numpy as np

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.liquidation import build_binance_suite

FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]

PAIRS = {
    "v4_hard": ("checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt",
                "checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt"),
    "v4s_soft": ("checkpoints/market_v4s_softmix/continuous_do_over_time_pfn_best.pt",
                 "checkpoints/market_v4s_softmix_ablated/continuous_do_over_time_pfn_best.pt"),
    # Seed replication of the soft pair: does the
    # compressed-scale reading hold at the low-response (+0.02) seed?
    "v4s_s43": ("checkpoints/market_v4s_s43/continuous_do_over_time_pfn_best.pt",
                "checkpoints/market_v4s_s43_ablated/continuous_do_over_time_pfn_best.pt"),
    "v4s_s44": ("checkpoints/market_v4s_s44/continuous_do_over_time_pfn_best.pt",
                "checkpoints/market_v4s_s44_ablated/continuous_do_over_time_pfn_best.pt"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--out", default="output/magnitude_scaling_v4s.json")
    args = ap.parse_args()

    dates = [str(date(2026, 4, 1) + timedelta(k)) for k in range(args.days)]
    suite = build_binance_suite(symbol="BTCUSDT", dates=dates, bar_seconds=60,
                                cache_dir="data/raw/binance", z_thresh=5.0)
    print(f"{len(suite)} burst episodes", flush=True)
    if not suite:
        raise SystemExit("no bursts detected")

    rng = np.random.default_rng(0)
    out = {"n_episodes": len(suite), "factors": FACTORS, "pairs": {}}
    for name, (ck, ack) in PAIRS.items():
        causal = get("market-dotpfn", checkpoint=ck, device="cpu")
        ablated = get("ablated-dotpfn", checkpoint=ack, device="cpu")
        rows = {f: [] for f in FACTORS}
        for ep in suite:
            base = float(np.asarray(ep.intervention.values).reshape(-1)[0])
            sgn = math.copysign(1.0, base)
            for f in FACTORS:
                e2 = copy.deepcopy(ep)
                e2.intervention.values = base * f
                d = float(causal.predict(e2).reshape(-1)[0]
                          - ablated.predict(e2).reshape(-1)[0])
                rows[f].append(sgn * d)
        per_f = {}
        means = []
        for f in FACTORS:
            x = np.asarray(rows[f])
            idx = rng.integers(0, len(x), (2000, len(x)))
            bs = x[idx].mean(axis=1)
            per_f[str(f)] = {"mean_delta": float(x.mean()),
                             "ci": [float(np.percentile(bs, 2.5)),
                                    float(np.percentile(bs, 97.5))]}
            means.append(x.mean())
        # OLS slope of response on factor: >0 and ~linear => scale shrinkage,
        # ~0 => lost magnitude sensitivity.
        fa = np.asarray(FACTORS)
        slope = float(((fa - fa.mean()) * (np.asarray(means) - np.mean(means))).sum()
                      / ((fa - fa.mean()) ** 2).sum())
        out["pairs"][name] = {"per_factor": per_f, "slope_delta_per_factor": slope}
        print(name, {k: round(v["mean_delta"], 3) for k, v in per_f.items()},
              "slope:", round(slope, 4), flush=True)

    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
