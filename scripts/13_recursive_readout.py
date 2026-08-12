#!/usr/bin/env python
"""Eval-time recursive readout: does composition restore duration accrual?

The ladder's verdict (results_summary 2026-07-31): the one-shot do-head
learns an amortized, context-independent, duration-flat response, no matter
the prior (v4/v6/v7) or the training geometry (v8). CausalLongPFN
(arXiv:2606.05797) sidesteps exactly this by recursive one-step composition
— in-context g-computation. Before committing to an architecture change,
this script tests the composition hypothesis with the EXISTING checkpoints:

Roll the model through the intervention window one bar at a time. At step k
the context is the 90 pre-onset bars plus k-1 already-rolled window bars
whose price is the model's own previous prediction and whose flow is the
intervened level (pre-window mean + c); the intervention token covers the
REMAINING window; the query is the next bar. The final prediction (last
window bar) minus the identically-rolled ablated twin's is the composed
response. If it rises with duration where the one-shot response is flat,
recursion/state dynamics is confirmed as the missing ingredient — zero
retraining needed for the verdict.

Semantics note: the rolled context includes the intervened flow level, so
each step's posterior can update on the flow path (context-conditioning by
construction). This mirrors g-computation's sequential conditioning; it is
NOT the ETT estimand (no factual post-onset prices are used — prices fed
back are the model's own interventional predictions).

Usage:
    python scripts/13_recursive_readout.py --n-onsets 20
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

DURATIONS = [2, 4, 7, 15, 31]

PAIRS = {
    "v4": ("checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt"),
    "v8": ("checkpoints/market_v8_geom/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v8_geom_ablated/continuous_do_over_time_pfn_best.pt"),
    "v9": ("checkpoints/market_v9_hawkes/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v9_hawkes_ablated/continuous_do_over_time_pfn_best.pt"),
    "v10": ("checkpoints/market_v10_rnn/recurrent_dotpfn_best.pt",
            "checkpoints/market_v10_rnn_ablated/recurrent_dotpfn_best.pt"),
}


def one_shot(model, ep, dur: int) -> float:
    """Reference readout: single query at the last window bar."""
    e = copy.deepcopy(ep)
    t_len = e.x_obs.shape[0]
    e.query_time = torch.tensor([(90 + dur - 1) / max(t_len - 1, 1)],
                                dtype=torch.float32)
    e.query_target = torch.tensor([1], dtype=torch.long)
    return float(model.predict(e).reshape(-1)[0])


def rolled(model, ep, dur: int, c: float) -> float:
    """Recursive readout: predict bar-by-bar through the window, feeding
    predictions (price) and the intervened level (flow) back into context.

    Args:
        model: baseline with the Episode predict protocol.
        ep: base episode from burst_episode (standardized units).
        dur: window length in bars; window occupies bars [90, 90 + dur).
        c: soft-intervention value in standardized flow units.

    Returns:
        The prediction at the last window bar after dur one-step compositions.
    """
    e = copy.deepcopy(ep)
    t_len = e.x_obs.shape[0]
    pre_flow_mean = float(e.x_obs[:90, 0].mean())
    pred = None
    for k in range(dur):
        onset = 90 + k
        # Remaining window carries the intervention token; the wrapper's
        # causal masking zeroes x_obs[onset:], so already-rolled bars
        # (< onset) stay visible as context.
        e.intervention.times = list(range(onset, 90 + dur))
        e.query_time = torch.tensor([onset / max(t_len - 1, 1)],
                                    dtype=torch.float32)
        e.query_target = torch.tensor([1], dtype=torch.long)
        pred = float(model.predict(e).reshape(-1)[0])
        # Feed the step back: price <- own prediction, flow <- intervened level.
        e.x_obs[onset, 1] = pred
        e.x_obs[onset, 0] = pre_flow_mean + c
        e.x_int = e.x_obs.clone()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2026-04-01")
    ap.add_argument("--n-onsets", type=int, default=20)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/recursive_readout.json")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    bars = bin_agg_trades(download_agg_trades(args.symbol, args.date),
                          bar_seconds=60)
    hi = len(bars) - (max(DURATIONS) + 3)
    onsets = rng.randint(90, hi, size=args.n_onsets)

    results = {"args": vars(args), "models": {}}
    for tag, (ck, ck_a) in PAIRS.items():
        if not (Path(ck).exists() and Path(ck_a).exists()):
            continue
        causal = get("market-dotpfn", checkpoint=ck)
        ablated = get("ablated-dotpfn", checkpoint=ck_a)
        rows = []
        for dur in DURATIONS:
            os_vals, rec_vals = [], []
            for i0 in onsets:
                try:
                    ep = burst_episode(bars, event_idx=int(i0), rng=rng,
                                       pre_bars=90, post_bars=dur + 1,
                                       window_bars=dur)
                except Exception:
                    continue
                ep.intervention.values = args.c
                os_vals.append(one_shot(causal, ep, dur)
                               - one_shot(ablated, ep, dur))
                rec_vals.append(rolled(causal, ep, dur, args.c)
                                - rolled(ablated, ep, dur, args.c))
            rows.append({"dur": dur, "n": len(os_vals),
                         "one_shot": float(np.mean(os_vals)),
                         "recursive": float(np.mean(rec_vals))})
        results["models"][tag] = rows
        prof_o = "  ".join(f"{r['dur']}:{r['one_shot']:+.3f}" for r in rows)
        prof_r = "  ".join(f"{r['dur']}:{r['recursive']:+.3f}" for r in rows)
        print(f"{tag} one-shot   {prof_o}")
        print(f"{tag} recursive  {prof_r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
