#!/usr/bin/env python
"""Split-conformal (CQR) recalibration of the counterfactual interval on ABIDES.

The transfer-calibration finding (scripts/05_calibration.py, Table 2) is that
the quantile head's nominal 80% interval collapses to ~0.30 coverage on
ABIDES. Seed replay supplies *counterfactual* outcomes for calibration, so the
conformalized-quantile-regression recipe applies directly to the
counterfactual interval — the exact setting sketched in the paper's §6.3.

Protocol: episodes are built from the exploratory seed range (seed0=0), split
chronologically by seed into a calibration half and an evaluation half. The
held-out seed range 1000-1999 (the paper's headline evaluation) is never
touched. For each model the CQR nonconformity score on calibration episode i
is E_i = max(q10(x_i) - y_i, y_i - q90(x_i)) with y_i the seed-replay
counterfactual outcome; the correction is the ceil((n+1)(1-alpha))/n empirical
quantile of {E_i}, added symmetrically to both interval endpoints on the
evaluation half.

Usage:
    python scripts/05b_conformal.py --n-cal 500 --n-eval 500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.abides_builder import build_abides_suite


def interval_stats(model, suite, correction: float = 0.0, k_scale=None, ablated_model=None) -> dict:
    taus = model.tau_levels
    i10, i90 = taus.index(0.1), taus.index(0.9)
    cov, width, scores = [], [], []
    for ep in suite:
        q = model.predict_quantiles(ep)  # (n_queries, n_tau)
        if k_scale is not None and ablated_model is not None:
            shift = (1.0 - k_scale) * (
                float(model.predict(ep).reshape(-1)[0])
                - float(ablated_model.predict(ep).reshape(-1)[0]))
            q = q - shift
        y = ep.y_true.reshape(-1)
        for k in range(y.numel()):
            lo = float(q[k, i10]) - correction
            hi = float(q[k, i90]) + correction
            cov.append(float(lo <= float(y[k]) <= hi))
            width.append(hi - lo)
            scores.append(max(lo - float(y[k]), float(y[k]) - hi) + correction)
    t = torch.tensor
    return {
        "n": len(cov),
        "coverage80": float(t(cov).mean()),
        "median_width80": float(t(width).median()),
        "scores": scores,
    }


def cqr_correction(scores: list[float], alpha: float = 0.2) -> float:
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    return float(sorted(scores)[min(rank, n) - 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cal", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--k-scale", type=float, default=None,
                    help="rescale the paired do-shift by k before conformal: "
                         "q_i <- q_i - (1-k)*(mean_causal_i - mean_ablated_i); "
                         "composes domain-scale calibration with interval "
                         "calibration (W4)")
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed; cal/eval halves are consecutive from here "
                    "(keep inside the exploratory range, away from 1000-1999)")
    ap.add_argument("--intervention-kind", choices=["hard", "soft"], default="soft")
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/conformal_abides.json")
    args = ap.parse_args()

    n_total = args.n_cal + args.n_eval
    print(f"building {n_total} ABIDES episodes (seeds {args.seed0}..{args.seed0 + n_total - 1}) ...",
          flush=True)
    suite = build_abides_suite(
        n_episodes=n_total, seed0=args.seed0, intervention_kind=args.intervention_kind,
        num_noise_agents=500, num_value_agents=50,
    )
    episodes = list(suite)
    cal, ev = episodes[: args.n_cal], episodes[args.n_cal:]

    models = {
        "causal": get("market-dotpfn", checkpoint=args.checkpoint, device=args.device),
        "ablated": get("ablated-dotpfn", checkpoint=args.ablated_checkpoint, device=args.device),
    }
    results = {"args": vars(args), "models": {}}
    print(f"{'model':>8s}  {'cov80 raw':>10s}  {'cov80 conf':>10s}  {'w80 raw':>8s}  {'w80 conf':>9s}  {'corr':>7s}")
    for name, model in models.items():
        # k-scale applies only to the causal model (the paired do-shift).
        kk = args.k_scale if name == "causal" else None
        ab = models["ablated"] if name == "causal" else None
        cal_stats = interval_stats(model, cal, k_scale=kk, ablated_model=ab)
        corr = cqr_correction(cal_stats["scores"], alpha=0.2)
        raw = interval_stats(model, ev, k_scale=kk, ablated_model=ab)
        conf = interval_stats(model, ev, correction=corr, k_scale=kk, ablated_model=ab)
        results["models"][name] = {
            "correction": corr,
            "raw": {k: v for k, v in raw.items() if k != "scores"},
            "conformal": {k: v for k, v in conf.items() if k != "scores"},
        }
        print(f"{name:>8s}  {raw['coverage80']:10.3f}  {conf['coverage80']:10.3f}  "
              f"{raw['median_width80']:8.3f}  {conf['median_width80']:9.3f}  {corr:7.3f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
