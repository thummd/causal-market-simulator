#!/usr/bin/env python
"""Metaorder duration benchmark on ABIDES seed replay, with exact counterfactual truth.

Fills the ABIDES duration-shape cell of the evidence table. The paper's ABIDES tier
uses one 15-minute execution window; here the window length sweeps the four
duration classes of the real-data benchmark (scripts/11) and the parent order size
sweeps three magnitudes, each at ``--seeds-per-cell`` fresh seeds disjoint from the
exploratory (0-999) and held-out (1000-1999) ranges. Every episode is a seed-replay
pair, so the measured impact at the end-pinned query is the exact counterfactual
effect (in pre-onset sigma-hat units, like the paper's ABIDES tier). The simulations
run once; both model pairs score the same episodes, and the output follows the
scripts/11 schema so scripts/26 and scripts/30 apply unchanged.

Usage:
    python scripts/47_duration_benchmark_abides.py --seeds-per-cell 50 \
        --out-prefix output/benchmark_abides_duration
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.abides_builder import abides_episode, run_metaorder_pair

_spec = importlib.util.spec_from_file_location(
    "prior_bench", Path(__file__).parent / "46_duration_benchmark_prior.py")
_prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior)

# Execution windows chosen so the 30 s bar count lands inside each duration class
# of the real-data benchmark: 3 bars (2-3), 6 bars (4-7), 12 bars (8-15), 24 bars (16-31).
WINDOWS = {
    "2-3": ("09:45:00", "09:46:30"),
    "4-7": ("09:45:00", "09:48:00"),
    "8-15": ("09:45:00", "09:51:00"),
    "16-31": ("09:45:00", "09:57:00"),
}
SIZES = (20, 40, 80)  # shares per wake; 40 is the paper's setting


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds-per-cell", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=3000,
                    help="first seed; the paper used 0-999 (exploratory) and 1000-1999 (held-out)")
    ap.add_argument("--recurrent", default="checkpoints/market_v10s_softmix/recurrent_dotpfn_best.pt")
    ap.add_argument("--recurrent-ablated",
                    default="checkpoints/market_v10s_softmix_ablated/recurrent_dotpfn_best.pt")
    ap.add_argument("--transformer", default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--transformer-ablated",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-prefix", default="output/benchmark_abides_duration")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed0)
    episodes = []
    seed = args.seed0
    for cls, win in WINDOWS.items():
        for qpw in SIZES:
            for _ in range(args.seeds_per_cell):
                pair = run_metaorder_pair(seed, quantity_per_wake=qpw, bar="30s", exec_window=win)
                ep = abides_episode(pair, rng=rng, n_queries=1, standardize=True,
                                    intervention_kind="soft")
                episodes.append((cls, qpw, seed, _prior.end_pinned(ep)))
                seed += 1
        print(f"simulated duration class {cls}: {len(episodes)} episodes so far", flush=True)

    pairs = {
        "v10s": (get("market-dotpfn", checkpoint=args.recurrent, device=args.device),
                 get("ablated-dotpfn", checkpoint=args.recurrent_ablated, device=args.device)),
        "v4": (get("market-dotpfn", checkpoint=args.transformer, device=args.device),
               get("ablated-dotpfn", checkpoint=args.transformer_ablated, device=args.device)),
    }
    for tag, (causal, ablated) in pairs.items():
        rows = []
        for cls, qpw, s, ep in episodes:
            c = float(ep.intervention.values)
            if c == 0:
                continue
            sgn = math.copysign(1.0, c)
            q = ep.metadata["query_idx"][0]
            price = int(ep.query_target[0])
            effect = float(ep.x_int[q, price] - ep.x_obs[q, price])
            p_c = float(causal.predict(ep).reshape(-1)[0])
            p_a = float(ablated.predict(ep).reshape(-1)[0])
            rows.append({"dur": len(ep.intervention.times), "c_abs": abs(c), "class": cls,
                         "shares_per_wake": qpw, "seed": s,
                         "measured": sgn * effect, "response": sgn * (p_c - p_a)})
        df = pd.DataFrame(rows)
        cells = _prior.bin_cells(df)
        res = {"args": vars(args), "tier": "ABIDES seed replay (exact counterfactual effect)",
               "classes": cells, "episodes": rows}
        ms = [c["measured"] for c in cells]
        rs = [c["response"] for c in cells]
        if len(ms) >= 4:
            res["curve_correlation"] = float(np.corrcoef(ms, rs)[0, 1])
        # Smoke runs with few seeds form no cells (bin_cells needs 20 per class);
        # report the raw episode-weighted ratio then instead of dividing by zero.
        if cells:
            res["k_cells"] = (sum(c["measured"] * c["n"] for c in cells)
                              / sum(c["response"] * c["n"] for c in cells))
        else:
            res["k_cells"] = float(df["measured"].sum() / df["response"].sum())
        out = Path(f"{args.out_prefix}_{tag}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"\n[{tag}] {len(df)} episodes; durations {df['dur'].min()}-{df['dur'].max()} bars")
        if cells:
            for c in cells:
                print(f"{c['dur']:>9} {c['c_bin']:>7} {c['n']:>5} {c['measured']:>+8.3f} "
                      f"{c['response']:>+9.3f}")
        else:
            print(df[["class", "dur", "shares_per_wake", "c_abs", "measured", "response"]]
                  .to_string(index=False))
        print(f"[{tag}] curve correlation {res.get('curve_correlation', float('nan')):+.3f}; "
              f"k = {res['k_cells']:.3f}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
