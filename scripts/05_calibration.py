#!/usr/bin/env python
"""Counterfactual calibration study: interval coverage and width per tier.

The quantile head emits per-tau counterfactual quantiles; point-estimate
impact models (AC, OW) produce no intervals at all — this study reports the
axis where a distributional causal model differs. For each tier we report
the empirical coverage of the nominal 80% interval [q10, q90] and 50%
interval [q25, q75], plus median interval widths, for the causal PFN and
its do-ablated twin (same head, no do-information).

Tiers: synthetic suites (counterfactual y_true; gamma in {0, 1, 2}),
ABIDES seed-replay episodes (counterfactual y_true), and Binance flow
bursts (factual y_true; cached data required).
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import torch

import dotime_market.baselines  # noqa: F401
from dotime.baselines import get
from dotime_market.data.synthetic import build_market_suite
from dotime_market.prior.market_scm import MarketDoTime

PRIOR_KW = dict(
    impact_lambda=(0.25, 1.0),
    theta_range=(0.5, 1.0),
    sigma_range=(0.2, 0.6),
    num_substeps=2,
)


def calibration(model, suite) -> dict:
    taus = model.tau_levels
    i10, i25, i75, i90 = (taus.index(t) for t in (0.1, 0.25, 0.75, 0.9))
    cov80, cov50, w80, w50 = [], [], [], []
    for ep in suite:
        q = model.predict_quantiles(ep)  # (n_queries, n_tau)
        y = ep.y_true.reshape(-1)
        for k in range(y.numel()):
            cov80.append(float(q[k, i10] <= y[k] <= q[k, i90]))
            cov50.append(float(q[k, i25] <= y[k] <= q[k, i75]))
            w80.append(float(q[k, i90] - q[k, i10]))
            w50.append(float(q[k, i75] - q[k, i25]))
    t = torch.tensor
    return {
        "n": len(cov80),
        "coverage80": float(t(cov80).mean()),
        "coverage50": float(t(cov50).mean()),
        "median_width80": float(t(w80).median()),
        "median_width50": float(t(w50).median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--n-synthetic", type=int, default=300)
    ap.add_argument("--n-abides", type=int, default=300)
    ap.add_argument("--crypto-days", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/calibration.json")
    args = ap.parse_args()

    causal = get("market-dotpfn", checkpoint=args.checkpoint, device=args.device)
    ablated = get("ablated-dotpfn", checkpoint=args.ablated_checkpoint, device=args.device)

    suites = {}
    for gamma in (0.0, 1.0, 2.0):
        prior = MarketDoTime(coupling_gamma=gamma, core_share=0.7, seed=0, **PRIOR_KW)
        suites[f"synthetic g={gamma:.0f}"] = build_market_suite(
            prior, n_episodes=args.n_synthetic, T=120, seed=0
        )
    if args.n_abides > 0:
        from dotime_market.data.abides_builder import build_abides_suite
        print(f"building {args.n_abides} ABIDES episodes ...", flush=True)
        suites["ABIDES"] = build_abides_suite(
            n_episodes=args.n_abides, seed0=0, intervention_kind="soft",
            num_noise_agents=500, num_value_agents=50,
        )
    if args.crypto_days > 0:
        from dotime_market.data.liquidation import build_binance_suite
        dates = [str(date(2026, 6, 1) + timedelta(d)) for d in range(args.crypto_days)]
        for sym in ("BTCUSDT", "ETHUSDT"):
            # burst episodes are SOFT by construction (see data/liquidation.py)
            suites[f"{sym[:3]} bursts"] = build_binance_suite(symbol=sym, dates=dates)

    results = {"args": vars(args), "tiers": {}}
    print(f"\n{'tier':>16s}  {'model':>8s}  {'n':>5s}  {'cov80':>6s}  {'cov50':>6s}  "
          f"{'w80':>7s}  {'w50':>7s}   (nominal: 0.80 / 0.50)")
    for tier, suite in suites.items():
        results["tiers"][tier] = {}
        for label, model in (("causal", causal), ("ablated", ablated)):
            c = calibration(model, suite)
            results["tiers"][tier][label] = c
            print(f"{tier:>16s}  {label:>8s}  {c['n']:5d}  {c['coverage80']:6.3f}  "
                  f"{c['coverage50']:6.3f}  {c['median_width80']:7.3f}  {c['median_width50']:7.3f}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
