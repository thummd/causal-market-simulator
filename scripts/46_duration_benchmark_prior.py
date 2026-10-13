#!/usr/bin/env python
"""Metaorder duration benchmark on the synthetic prior, with exact counterfactual truth.

The real-data metaorder benchmark (scripts/11) compares the paired do-head response
with *measured* impact per (duration x magnitude) cell, where measured impact is a
factual population mean. On the prior the same construction can be scored against
the exact per-episode counterfactual effect, because every synthetic episode is a
shared-noise pair (``x_obs``, ``x_int``). This script draws soft (additive-drift)
interventions whose window length spans the benchmark's four duration classes,
pins the query at the last window bar (end-pinned, as in scripts/12), and writes
the same JSON schema as scripts/11 so scripts/26 (bootstrap CI) and scripts/30
(paired delta) apply unchanged. The scale constant of the crypto protocol,
k = sum(m n) / sum(r n) over cells, is reported alongside: on the prior an unbiased
model has k = 1.

Usage:
    python scripts/46_duration_benchmark_prior.py --n-episodes 4000 \
        --checkpoint checkpoints/market_v10s_softmix/recurrent_dotpfn_best.pt \
        --ablated-checkpoint checkpoints/market_v10s_softmix_ablated/recurrent_dotpfn_best.pt \
        --out output/benchmark_prior_v10s.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.synthetic import build_market_suite
from dotime_market.prior.market_scm import MarketDoTime

# Synthetic-tier conventions of scripts/01b and 06: pinned gamma, core share 0.7.
PRIOR_KW = dict(
    impact_lambda=(0.25, 1.0),
    theta_range=(0.5, 1.0),
    sigma_range=(0.2, 0.6),
    num_substeps=2,
)
T_EPISODE = 120
# Same duration classes as the real-data benchmark (scripts/11), in bars.
DUR_CLASSES = [(2, 3), (4, 7), (8, 15), (16, 31)]


def end_pinned(ep):
    """Return a copy of ``ep`` whose single query sits on the last window bar.

    The benchmark reads the response at the end of execution, where accrued
    impact is largest, so the query is moved from its sampled post-onset position
    to the final bar of the intervention window. ``y_true`` is re-read from the
    interventional arm at that bar and the observational value is stored in the
    metadata for the exact effect.

    Args:
        ep: A synthetic ``Episode`` from :func:`build_market_suite` (regular
            ``dt = 1`` schedule, so bar index equals time step).

    Returns:
        A new ``Episode`` with one end-pinned query.

    Raises:
        ValueError: If the intervention window is empty.
    """
    times = list(ep.intervention.times)
    if not times:
        raise ValueError("episode has an empty intervention window")
    q = int(max(times))
    price = int(ep.query_target.reshape(-1)[0])
    span = float(ep.length - 1)
    meta = dict(ep.metadata)
    meta["query_idx"] = [q]
    meta["y_obs"] = [float(ep.x_obs[q, price])]
    return dataclasses.replace(
        ep,
        y_true=ep.x_int[q, price].reshape(1).to(torch.float32),
        query_target=torch.tensor([price], dtype=torch.long),
        query_time=torch.tensor([q / span], dtype=torch.float32),
        metadata=meta,
    )


def bin_cells(df: pd.DataFrame) -> list[dict]:
    """Form the 12 benchmark cells (duration class x |c| tertile) as scripts/11 does.

    Args:
        df: Episode table with ``dur``, ``c_abs``, ``measured``, ``response``.

    Returns:
        One dict per cell with the cell means and count, in scripts/11's format.
    """
    cells = []
    for lo, hi in DUR_CLASSES:
        cls = df[(df["dur"] >= lo) & (df["dur"] <= hi)]
        if len(cls) < 20:
            continue
        edges = cls["c_abs"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
        for b in range(3):
            sub = cls[(cls["c_abs"] >= edges[b]) & (cls["c_abs"] <= edges[b + 1])]
            if len(sub) < 10:
                continue
            cells.append({"dur": f"{lo}-{hi}", "c_bin": b, "n": int(len(sub)),
                          "c_mean": float(sub["c_abs"].mean()),
                          "measured": float(sub["measured"].mean()),
                          "response": float(sub["response"].mean())})
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodes", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--core-share", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ablated-checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Soft interventions only (metaorder semantics) and windows of 2 to 32 bars so
    # every benchmark duration class is populated; everything else is the grid prior.
    prior = MarketDoTime(
        coupling_gamma=args.gamma, core_share=args.core_share, seed=args.seed,
        intervention_kind_probs=(0.0, 1.0, 0.0),
        intervention_window_frac=(2 / T_EPISODE, 32 / T_EPISODE),
        **PRIOR_KW,
    )
    suite = build_market_suite(prior, n_episodes=args.n_episodes, T=T_EPISODE, seed=args.seed)
    causal = get("market-dotpfn", checkpoint=args.checkpoint, device=args.device)
    ablated = get("ablated-dotpfn", checkpoint=args.ablated_checkpoint, device=args.device)

    rows = []
    for ep0 in suite:
        if str(ep0.intervention.intervention_type).lower().endswith("hard"):
            continue  # defensive: the prior is configured soft-only
        ep = end_pinned(ep0)
        c = float(ep.intervention.values)
        if c == 0:
            continue
        sgn = math.copysign(1.0, c)
        q = ep.metadata["query_idx"][0]
        price = int(ep.query_target[0])
        effect = float(ep.x_int[q, price] - ep.x_obs[q, price])
        p_c = float(causal.predict(ep).reshape(-1)[0])
        p_a = float(ablated.predict(ep).reshape(-1)[0])
        rows.append({"dur": len(ep.intervention.times), "c_abs": abs(c),
                     "measured": sgn * effect, "response": sgn * (p_c - p_a)})
    df = pd.DataFrame(rows)
    cells = bin_cells(df)
    results = {"args": vars(args), "tier": "synthetic prior (exact counterfactual effect)",
               "classes": cells, "episodes": rows}
    ms = [c["measured"] for c in cells]
    rs = [c["response"] for c in cells]
    if len(ms) >= 4:
        results["curve_correlation"] = float(np.corrcoef(ms, rs)[0, 1])
    results["k_cells"] = (sum(c["measured"] * c["n"] for c in cells)
                          / sum(c["response"] * c["n"] for c in cells))
    print(f"{len(df)} episodes; durations {df['dur'].min()}-{df['dur'].max()} bars")
    print(f"{'duration':>9} {'|c| bin':>7} {'n':>5} {'effect':>8} {'response':>9}")
    for c in cells:
        print(f"{c['dur']:>9} {c['c_bin']:>7} {c['n']:>5} {c['measured']:>+8.3f} {c['response']:>+9.3f}")
    print(f"curve correlation {results.get('curve_correlation', float('nan')):+.3f}; k = {results['k_cells']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
