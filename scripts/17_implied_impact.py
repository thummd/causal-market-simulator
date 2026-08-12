#!/usr/bin/env python
"""Implied impact curve: response vs intervention magnitude (concavity check).

The roadmap diagnostic never yet run: sweep the soft-intervention value c on
FIXED real contexts and read the paired do-head response as a function of c.
The linear-drift prior implies a linear response; real metaorder impact is
concave (local exponents 0.5-0.7 on these venues, scripts 10/10b). A log-log
slope ~1 means the model inherits the prior's linearity — a level-invisible
mismatch the +0.97 cell-correlation cannot see (cells bin by |c| tercile).

Usage:
    python scripts/17_implied_impact.py --dur 7 --n-onsets 30
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.binance import bin_agg_trades, download_agg_trades
from dotime_market.data.liquidation import burst_episode

C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-04-01")
    ap.add_argument("--dur", type=int, default=7)
    ap.add_argument("--n-onsets", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v10s_softmix/recurrent_dotpfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v10s_softmix_ablated/recurrent_dotpfn_best.pt")
    ap.add_argument("--out", default="output/implied_impact.json")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    causal = get("market-dotpfn", checkpoint=args.checkpoint)
    ablated = get("ablated-dotpfn", checkpoint=args.ablated_checkpoint)
    bars = bin_agg_trades(download_agg_trades(args.symbol, args.date), 60)
    onsets = rng.randint(90, len(bars) - args.dur - 3, size=args.n_onsets)

    rows = []
    for c in C_GRID:
        vals = []
        for i0 in onsets:
            try:
                ep = burst_episode(bars, event_idx=int(i0), rng=rng, pre_bars=90,
                                   post_bars=args.dur + 1, window_bars=args.dur)
            except Exception:
                continue
            e = copy.deepcopy(ep)
            e.intervention.values = c
            t_len = e.x_obs.shape[0]
            e.query_time = torch.tensor([(90 + args.dur - 1) / (t_len - 1)],
                                        dtype=torch.float32)
            e.query_target = torch.tensor([1], dtype=torch.long)
            vals.append(float(causal.predict(e).reshape(-1)[0])
                        - float(ablated.predict(e).reshape(-1)[0]))
        rows.append({"c": c, "n": len(vals), "response": float(np.mean(vals))})
        print(f"c={c:>5.2f}  response {np.mean(vals):+7.3f}")

    pos = [(r["c"], r["response"]) for r in rows if r["response"] > 0]
    slope = float(np.polyfit(np.log([p[0] for p in pos]),
                             np.log([p[1] for p in pos]), 1)[0]) if len(pos) > 2 else None
    print(f"log-log exponent: {slope:.3f}  (linear prior -> 1.0; venue-measured 0.5-0.7)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "curve": rows, "exponent": slope}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
