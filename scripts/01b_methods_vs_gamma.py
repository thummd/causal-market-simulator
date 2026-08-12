#!/usr/bin/env python
"""Phase 2 — methods-vs-gamma evaluation: the Figure-2 comparison.

For each gamma on the grid, builds a market suite (pinned gamma/core_share,
prior hyperparameters matched to configs/train_market.yaml so the PFN is
evaluated in-distribution) and scores every method on the same episodes:

- rmse : pooled RMSE against the exact counterfactual y_true.
- bias : mean of sign(c) * (pred - y_true) — the signed error projected on
  the clamp direction. Confounded predictive methods drift positive with
  gamma; causal methods stay centred (see tests/test_oracle_and_pfn.py).

Methods: stock naive (Mean, AR1), naive impact models (AlmgrenChriss,
OWPropagator), the true-parameter causal reference (param-oracle), and the
trained PFN pair (market-dotpfn causal / ablated-dotpfn predictive twin).

Usage:
    python scripts/01b_methods_vs_gamma.py \
        --checkpoint checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt \
        --ablated-checkpoint checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.synthetic import build_market_suite
from dotime_market.prior.market_scm import MarketDoTime

# Matched to configs/train_market.yaml (the PFN's training distribution).
PRIOR_KW = dict(
    impact_lambda=(0.25, 1.0),
    theta_range=(0.5, 1.0),
    sigma_range=(0.2, 0.6),
    num_substeps=2,
)
T_EPISODE = 120


def score(model, suite, n_boot: int = 10000) -> dict:
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
    sq = err**2
    # Episode-level bootstrap CIs for both reported metrics (audit item C3).
    from dotime.evaluation import bootstrap_ci

    _, _, b_lo, b_hi = bootstrap_ci([float(v) for v in signed], n=n_boot)
    import numpy as np

    rng = np.random.RandomState(0)
    sq_np = sq.numpy()
    boots = np.sqrt(
        sq_np[rng.randint(0, len(sq_np), size=(n_boot, len(sq_np)))].mean(axis=1)
    )
    return {
        "rmse": float(torch.sqrt(sq.mean())),
        "rmse_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "mae": float(err.abs().mean()),
        "bias": float(signed.mean()),
        "bias_ci": [b_lo, b_hi],
        "n": int(err.numel()),
        # per-episode signed errors, aligned across methods (paired analyses)
        "signed_errors": [float(v) for v in signed],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/coupling_sweep.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/methods_vs_gamma.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["sweep"]

    models = {
        "Mean": get("Mean"),
        "AR1": get("AR1"),
        "AlmgrenChriss": get("AlmgrenChriss"),
        "OWPropagator": get("OWPropagator"),
        "ReturnRegression": get("ReturnRegression"),
        "SquareRootLaw": get("SquareRootLaw"),
        "param-oracle": get("param-oracle"),
        "market-dotpfn": get("market-dotpfn", checkpoint=args.checkpoint, device=args.device),
        "ablated-dotpfn": get(
            "ablated-dotpfn", checkpoint=args.ablated_checkpoint, device=args.device
        ),
    }

    results: dict = {"config": cfg, "n_episodes": args.n_episodes, "per_gamma": []}
    names = list(models)
    print("gamma  metric  " + "  ".join(f"{n:>14s}" for n in names))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for gamma in cfg["gamma_grid"]:
        prior = MarketDoTime(
            coupling_gamma=float(gamma),
            core_share=float(cfg["core_share"]),
            seed=int(cfg["seed"]),
            **PRIOR_KW,
        )
        suite = build_market_suite(
            prior, n_episodes=args.n_episodes, T=T_EPISODE, seed=int(cfg["seed"])
        )
        row = {"gamma": float(gamma), "methods": {}}
        for name, model in models.items():
            row["methods"][name] = score(model, suite)
        results["per_gamma"].append(row)
        for metric in ("rmse", "bias"):
            vals = "  ".join(f"{row['methods'][n][metric]:14.4f}" for n in names)
            print(f"{gamma:5.2f}  {metric:6s}  {vals}", flush=True)
        out_path.write_text(json.dumps(results, indent=2))  # incremental

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
