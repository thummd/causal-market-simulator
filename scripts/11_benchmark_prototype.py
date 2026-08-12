#!/usr/bin/env python
"""Gate B prototype: model response vs measured impact on synthetic metaorders.

The (a2) real-market benchmark recipe, executed end-to-end once: build
episodes at synthetic-metaorder onsets (mapping-function aggregation on 1-min
bars), feed them to the causal model and its do-ablated twin as SOFT
interventions, and compare the paired do-head response against the *measured*
realized impact, per (duration x participation) class.

Per Gate C, impact on these venues is count/duration-driven, so the readout
is duration-conditioned: within each duration class, does the model's
response track the measured impact's level and Q-dependence?

Both quantities are computed per episode and binned identically:
- measured impact  = sign(c) * (y_realized - last pre-onset price)  [sigma-hat]
- model response   = causal prediction - ablated prediction, signed  [sigma-hat]

Factual-outcome caveats are the crypto tier's usual ones; the population-level
comparison is the benchmark's point.

Usage:
    python scripts/11_benchmark_prototype.py --n-days 40 --per-day 60
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get

from dotime_market.data.binance import bin_agg_trades, download_agg_trades
from dotime_market.data.liquidation import burst_episode

_spec = importlib.util.spec_from_file_location(
    "recreadout", Path(__file__).parent / "13_recursive_readout.py")
_rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rec)

DUR_CLASSES = [(2, 3), (4, 7), (8, 15), (16, 31)]


def sample_day_metaorders(bars: pd.DataFrame, rng, n_traders: int = 2,
                          per_day: int = 60, pre_bars: int = 90):
    """Synthetic metaorders on the bar tape (gross-directional volume),
    stratified across duration classes; returns (start, duration) tuples."""
    flow = bars["flow"].to_numpy()
    sign = np.sign(flow)
    trader = rng.choice(n_traders, size=len(bars))
    order = np.lexsort((np.arange(len(bars)), trader))
    t_s, s_s = trader[order], sign[order]
    new_run = np.empty(len(bars), dtype=bool)
    new_run[0] = True
    new_run[1:] = (t_s[1:] != t_s[:-1]) | (s_s[1:] != s_s[:-1])
    run_id = np.cumsum(new_run) - 1
    rec = pd.DataFrame({"run": run_id, "idx": order})
    g = rec.groupby("run")["idx"]
    runs = pd.DataFrame({"i0": g.min(), "i1": g.max()})
    runs["dur"] = runs["i1"] - runs["i0"] + 1
    runs = runs[(runs["dur"] >= 2) & (runs["i0"] >= pre_bars)
                & (runs["i1"] < len(bars) - 2)]
    picks = []
    per_class = max(per_day // len(DUR_CLASSES), 1)
    for lo, hi in DUR_CLASSES:
        cls = runs[(runs["dur"] >= lo) & (runs["dur"] <= hi)]
        if len(cls):
            take = cls.sample(n=min(per_class, len(cls)), random_state=rng)
            picks.extend((int(r.i0), int(r.dur)) for r in take.itertuples())
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dates-from", default="2026-04-01")
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--per-day", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--count-handle", action="store_true",
                    help="Gate-C-aware handle: intervention value = signed "
                         "standardized excess trade-arrival COUNT over the "
                         "window (impact is count-driven on these venues)")
    ap.add_argument("--recursive", action="store_true",
                    help="use the eval-time recursive readout (script 13): "
                         "roll one-step predictions through the window "
                         "instead of the one-shot query")
    ap.add_argument("--out", default="output/benchmark_prototype.json")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    causal = get("market-dotpfn", checkpoint=args.checkpoint)
    ablated = get("ablated-dotpfn", checkpoint=args.ablated_checkpoint)

    y0, m0, d0 = map(int, args.dates_from.split("-"))
    rows = []
    for k in range(args.n_days):
        day = str(date(y0, m0, d0) + timedelta(k))
        try:
            bars = bin_agg_trades(download_agg_trades(args.symbol, day), bar_seconds=60)
        except Exception:
            continue
        for i0, dur in sample_day_metaorders(bars, rng, per_day=args.per_day):
            try:
                ep = burst_episode(bars, event_idx=i0, rng=rng, pre_bars=90,
                                   post_bars=dur + 1, window_bars=dur)
            except Exception:
                continue
            if args.count_handle:
                cnt = bars["n_trades"].to_numpy()
                pre = cnt[i0 - 90:i0]
                exc = (cnt[i0:i0 + dur].mean() - pre.mean()) / max(pre.std(), 1e-9)
                ep.intervention.values = float(np.sign(ep.intervention.values) * abs(exc))
            c = float(ep.intervention.values)
            if c == 0:
                continue
            sgn = math.copysign(1.0, c)
            onset = min(ep.intervention.times)
            pre_ref = float(ep.x_obs[onset - 1, 1])
            y = float(ep.y_true.reshape(-1)[0])
            if args.recursive:
                p_c = _rec.rolled(causal, ep, dur, c)
                p_a = _rec.rolled(ablated, ep, dur, c)
            else:
                p_c = float(causal.predict(ep).reshape(-1)[0])
                p_a = float(ablated.predict(ep).reshape(-1)[0])
            rows.append({"dur": dur, "c_abs": abs(c),
                         "measured": sgn * (y - pre_ref),
                         "response": sgn * (p_c - p_a)})
    df = pd.DataFrame(rows)
    print(f"{len(df)} episodes evaluated")

    results = {"args": vars(args), "classes": []}
    print(f"\n{'duration':>9} {'|c| bin':>10} {'n':>5} {'measured':>9} {'response':>9}")
    for lo, hi in DUR_CLASSES:
        cls = df[(df["dur"] >= lo) & (df["dur"] <= hi)]
        if len(cls) < 20:
            continue
        # participation proxy: standardized intervention magnitude terciles
        edges = cls["c_abs"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
        for b in range(3):
            sub = cls[(cls["c_abs"] >= edges[b]) & (cls["c_abs"] <= edges[b + 1])]
            if len(sub) < 10:
                continue
            m, r = sub["measured"].mean(), sub["response"].mean()
            results["classes"].append({
                "dur": f"{lo}-{hi}", "c_bin": b, "n": int(len(sub)),
                "c_mean": float(sub["c_abs"].mean()),
                "measured": float(m), "response": float(r)})
            print(f"{lo}-{hi:>4} {edges[b]:>5.2f}-{edges[b+1]:<4.2f} {len(sub):>5} "
                  f"{m:>+9.3f} {r:>+9.3f}")
    results["episodes"] = rows
    ms = [c["measured"] for c in results["classes"]]
    rs = [c["response"] for c in results["classes"]]
    if len(ms) >= 4:
        corr = float(np.corrcoef(ms, rs)[0, 1])
        results["curve_correlation"] = corr
        print(f"\ncurve correlation (measured vs response, across classes): {corr:+.3f}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
