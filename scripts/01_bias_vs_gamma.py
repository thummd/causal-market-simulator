#!/usr/bin/env python
"""Phase 1 — bias-vs-gamma sweep: the data behind the headline figure.

For each gamma on the grid (configs/coupling_sweep.yaml), samples market SCMs
with core_share pinned, and per episode measures:

- naive_emp : OLS slope of Y on A over a long observational path (what a
  predictive model fits),
- do_emp    : empirical interventional slope from hard do(A = +/-2) clamp arms,
- plus the episode's analytic values (naive slope, do slope = lambda/theta_Y,
  and the pure confounding bias — see dotime_market.prior.coupling).

Expected shape: do_emp is flat in gamma (~ mean lambda/theta_Y); naive_emp
rises linearly; bias_emp := naive_emp - analytic gamma=0 naive slope tracks
bias_ana and passes through 0 at gamma = 0.

The `methods` field of the config is the Phase-2 baseline-level version of
this figure (evaluated via dotime.evaluation on Episode suites); this script
is the slope-level Phase-1 deliverable. Plotting is deferred until the Phase-2
results exist — the JSON written here has everything the figure needs.

Usage:
    python scripts/01_bias_vs_gamma.py                        # full config run
    python scripts/01_bias_vs_gamma.py --n-episodes 20        # smoke run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import yaml

from dotime.continuous import regular_schedule
from dotime_market.evaluation.slopes import interventional_slope, naive_ols_slope
from dotime_market.prior.market_scm import FLOW_TOPO, PRICE_TOPO, MarketContinuousSCMSampler

T_OBS = 500
NUM_SUBSTEPS = 4  # theta*h <= 0.5 at the hyperprior's upper end


def sweep_point(
    gamma: float,
    core_share: float,
    n_episodes: int,
    seed: int,
    with_do: bool = True,
    do_episodes: int | None = None,
) -> dict:
    """One grid point. ``do_episodes`` caps the (dominant-cost) empirical
    interventional arms to the first N episodes — the do-slope is flat in
    gamma and analytically validated, so a subsample suffices for the figure
    while the naive/bias curves use all episodes."""
    sampler = MarketContinuousSCMSampler(
        core_share=core_share, coupling_gamma=gamma, seed=seed
    )
    gen = torch.Generator().manual_seed(seed + 1)
    times, dts = regular_schedule(T=T_OBS, dt=1.0)

    naive_emp, do_emp, naive_ana, do_ana, bias_emp, bias_ana = [], [], [], [], [], []
    for ep in range(n_episodes):
        scm, conf, coup = sampler.sample()
        _, x = scm.simulate(times, dts, generator=gen, num_substeps=NUM_SUBSTEPS)
        ne = naive_ols_slope(x, a_idx=FLOW_TOPO, y_idx=PRICE_TOPO)
        naive_emp.append(ne)
        naive_ana.append(coup.naive_slope(conf))
        do_ana.append(coup.do_slope())
        gamma0 = dataclasses.replace(coup, gamma=0.0)
        bias_emp.append(ne - gamma0.naive_slope(conf))
        bias_ana.append(coup.confounding_bias(conf))
        if with_do and (do_episodes is None or ep < do_episodes):
            do_emp.append(
                interventional_slope(
                    scm,
                    a_idx=FLOW_TOPO,
                    y_idx=PRICE_TOPO,
                    generator=gen,
                    num_substeps=NUM_SUBSTEPS,
                )
            )

    def _stats(vals: list) -> dict:
        t = torch.tensor(vals)
        return {"mean": float(t.mean()), "std": float(t.std()), "n": len(vals)}

    out = {
        "gamma": gamma,
        "naive_emp": _stats(naive_emp),
        "naive_ana": _stats(naive_ana),
        "do_ana": _stats(do_ana),
        "bias_emp": _stats(bias_emp),
        "bias_ana": _stats(bias_ana),
    }
    if with_do:
        out["do_emp"] = _stats(do_emp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/coupling_sweep.yaml")
    ap.add_argument("--n-episodes", type=int, default=None,
                    help="override n_episodes_per_gamma (smoke runs)")
    ap.add_argument("--out", default="output/bias_vs_gamma.json")
    ap.add_argument("--no-do", action="store_true",
                    help="skip the (slow) empirical interventional arms")
    ap.add_argument("--do-episodes", type=int, default=None,
                    help="run the interventional arms on only the first N episodes per gamma")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["sweep"]
    n_episodes = args.n_episodes or cfg["n_episodes_per_gamma"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    print(
        f"{'gamma':>6}  {'naive_emp':>10}  {'naive_ana':>10}  {'do_emp':>8}  "
        f"{'do_ana':>8}  {'bias_emp':>9}  {'bias_ana':>9}"
    )
    for gamma in cfg["gamma_grid"]:
        r = sweep_point(
            gamma=float(gamma),
            core_share=float(cfg["core_share"]),
            n_episodes=n_episodes,
            seed=int(cfg["seed"]),
            with_do=not args.no_do,
            do_episodes=args.do_episodes,
        )
        results.append(r)
        do_emp = f"{r['do_emp']['mean']:8.4f}" if "do_emp" in r else "      --"
        print(
            f"{gamma:6.2f}  {r['naive_emp']['mean']:10.4f}  {r['naive_ana']['mean']:10.4f}  "
            f"{do_emp}  {r['do_ana']['mean']:8.4f}  "
            f"{r['bias_emp']['mean']:9.4f}  {r['bias_ana']['mean']:9.4f}",
            flush=True,
        )
        # Incremental write: a multi-hour run should survive interruption.
        out_path.write_text(json.dumps({"config": cfg, "n_episodes": n_episodes,
                                        "results": results}, indent=2))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
