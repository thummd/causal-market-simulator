#!/usr/bin/env python
"""Phase 2 — ABIDES ground-truth study: the same harness on an agent-based market.

Builds metaorder episodes with seed-replay counterfactuals
(dotime_market.data.abides_builder) and scores the same method set as the
synthetic evaluation (scripts/01b_methods_vs_gamma.py). Here nobody knows the
true impact parameters, so there is no param-oracle row — the exact
counterfactual y_true from seed replay IS the ground truth.

Requires the "abides" extra. ABIDES runs are minutes each at full scale; use
--n-episodes and the agent-count flags to trade fidelity for time.

Usage:
    python scripts/03_abides_study.py --n-episodes 50 \
        --checkpoint checkpoints/market_v1/continuous_do_over_time_pfn_best.pt \
        --ablated-checkpoint checkpoints/market_v1_ablated/continuous_do_over_time_pfn_best.pt
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
        # per-episode signed errors, aligned across methods (same suite order)
        "signed_errors": [float(v) for v in signed],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--quantity-per-wake", type=int, default=40)
    ap.add_argument("--bar", default="30s")
    ap.add_argument("--end-time", default="10:30:00")
    ap.add_argument("--num-noise-agents", type=int, default=500)
    ap.add_argument("--num-value-agents", type=int, default=50)
    ap.add_argument("--intervention-kind", choices=["hard", "soft"], default="hard",
                    help="episode encoding: soft (additive flow — semantically faithful; "
                    "needs a soft-trained checkpoint, e.g. market_v4) or hard (v1-v3 models)")
    ap.add_argument("--checkpoint", default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="bootstrap resamples for bias CIs and paired deltas vs the ablated twin")
    ap.add_argument("--out", default="output/abides_study.json")
    args = ap.parse_args()

    print(f"building {args.n_episodes} ABIDES metaorder episodes ...", flush=True)
    suite = build_abides_suite(
        n_episodes=args.n_episodes,
        seed0=args.seed0,
        quantity_per_wake=args.quantity_per_wake,
        bar=args.bar,
        end_time=args.end_time,
        num_noise_agents=args.num_noise_agents,
        num_value_agents=args.num_value_agents,
        intervention_kind=args.intervention_kind,
    )

    # Per-episode seed-replay ground truth: the exact counterfactual impact in
    # the intervention direction, sign(c) * (Y_int - Y_obs) at the query. This
    # is the committed source of the paper's measured-ground-truth numbers.
    gt, vals = [], []
    for ep in suite:
        c = float(ep.intervention.values)
        for k in range(ep.query_target.numel()):
            q = ep.metadata["query_idx"][k]
            gt.append(math.copysign(1.0, c) * float(ep.x_int[q, 1] - ep.x_obs[q, 1]))
            vals.append(c)

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
    results = {"args": vars(args), "methods": {},
               "ground_truth": {"impact": gt, "intervention_values": vals}}
    print(f"{'method':>16s}  {'rmse':>9s}  {'mae':>9s}  {'bias':>9s}")
    for name, model in models.items():
        s = score(model, suite)
        results["methods"][name] = s
        print(f"{name:>16s}  {s['rmse']:9.4f}  {s['mae']:9.4f}  {s['bias']:9.4f}", flush=True)

    # Bias CIs + paired deltas vs the ablated twin (same episodes -> the
    # per-episode differences cancel shared market noise).
    from dotime.evaluation import bootstrap_ci

    gm, _, glo, ghi = bootstrap_ci(gt, n=args.bootstrap)
    results["ground_truth"].update({"mean": gm, "ci": [glo, ghi]})
    print(f"\nseed-replay ground-truth impact: {gm:+.4f}  [{glo:+.4f}, {ghi:+.4f}]  (n={len(gt)})")

    ref = "ablated-dotpfn"
    print(f"\n{'method':>16s}  {'bias':>8s}  {'95% CI':>20s}  {'d_vs_ablated':>12s}  {'95% CI':>20s}")
    for name in models:
        errs = results["methods"][name]["signed_errors"]
        _, _, lo, hi = bootstrap_ci(errs, n=args.bootstrap)
        entry = {"bias_ci": [lo, hi]}
        if name != ref:
            deltas = [a - b for a, b in zip(errs, results["methods"][ref]["signed_errors"])]
            dm, _, dlo, dhi = bootstrap_ci(deltas, n=args.bootstrap)
            entry.update({"delta_vs_ablated": dm, "delta_ci": [dlo, dhi]})
            d_str = f"{dm:12.4f}  [{dlo:8.4f}, {dhi:8.4f}]"
        else:
            d_str = f"{'--':>12s}  {'':20s}"
        results["methods"][name].update(entry)
        b = results["methods"][name]["bias"]
        print(f"{name:>16s}  {b:8.4f}  [{lo:8.4f}, {hi:8.4f}]  {d_str}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
