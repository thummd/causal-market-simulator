#!/usr/bin/env python
"""Placebo test for the crypto flow-burst study (audit item 4).

Construction: for each symbol/day, sample anchor bars that are demonstrably
NOT events (|z| < 2 against the preceding window, at least min_gap bars from
every detected real burst), matched in count to the real events, and attach
to each a *fake* intervention whose signed magnitude is resampled from the
real events' value distribution. Everything else (episode encoding,
standardization, scoring) is identical to the real study.

What it establishes (and what it cannot): the paired do-head delta is
algebraically y-independent (it is the mean prediction shift), so it will
appear at placebos too — the informative readout is the BIAS pattern:

- placebo anchors: realized post-anchor moves in the fake-intervention
  direction should be ~0 (ablated/mean bias ~0), so the causal model's
  response should CREATE positive bias roughly equal to its shift;
- real events: the same response REDUCES bias (ablated ~ -2.3, causal much
  closer to 0).

Together: the large directional post-event move is event-locked (not an
artifact of the detection pipeline), and the do-head's response magnitude is
only accurate where a real forced-flow event occurred. This does NOT prove
the mechanism is causal impact rather than momentum — real data has no
counterfactual; the paper's language is softened accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import dotime_market.baselines  # noqa: F401
from dotime.baselines import get
from dotime.evaluation import bootstrap_ci
from dotime_market.data.binance import bin_agg_trades, download_agg_trades
from dotime_market.data.liquidation import burst_episode, detect_flow_bursts
from dotime.benchmarks import BenchmarkSuite, SuiteMetadata


def placebo_anchors(bars, real_events, n_wanted, rng, pre_bars=90, post_bars=30,
                    z_max=2.0, min_gap=60):
    flow = bars["flow"].to_numpy()
    n = len(flow)
    candidates = []
    for i in range(pre_bars, n - post_bars - 3):
        if any(abs(i - j) < min_gap for j in real_events):
            continue
        pre = flow[i - pre_bars : i]
        sd = pre.std()
        if sd <= 0:
            continue
        if abs(flow[i] - pre.mean()) / sd < z_max:
            candidates.append(i)
    rng.shuffle(candidates)
    # Enforce non-overlap among selected placebos as well.
    chosen = []
    for i in candidates:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
        if len(chosen) >= n_wanted:
            break
    return sorted(chosen)


def score(model, suite):
    signed = []
    for ep in suite:
        p = float(model.predict(ep)[0])
        y = float(ep.y_true[0])
        c = float(ep.intervention.values)
        signed.append(math.copysign(1.0, c) * (p - y))
    return signed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dates-from", default="2026-04-01")
    ap.add_argument("--n-days", type=int, default=91)
    ap.add_argument("--bar-seconds", type=int, default=60)
    ap.add_argument("--cache-dir", default="data/raw/binance")
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/placebo_test.json")
    args = ap.parse_args()

    y0, m0, d0 = (int(x) for x in args.dates_from.split("-"))
    dates = [str(date(y0, m0, d0) + timedelta(d)) for d in range(args.n_days)]
    rng = np.random.RandomState(args.seed)

    real_eps, placebo_eps, real_values = [], [], []
    for day in dates:
        bars = bin_agg_trades(
            download_agg_trades(args.symbol, day, cache_dir=args.cache_dir),
            bar_seconds=args.bar_seconds,
        )
        events = detect_flow_bursts(bars)
        for idx in events:
            ep = burst_episode(bars, idx, rng=rng)
            real_eps.append(ep)
            real_values.append(float(ep.intervention.values))
        for idx in placebo_anchors(bars, events, n_wanted=len(events), rng=rng):
            ep = burst_episode(bars, idx, rng=rng)
            # Fake intervention: magnitude+sign resampled from real events.
            ep.intervention.values = float(rng.choice(real_values)) if real_values else 1.0
            placebo_eps.append(ep)

    def suite(eps, name):
        meta = SuiteMetadata(name=name, version="0.0.1", zenodo_record_id="",
                             doi="", description=name, n_episodes=len(eps),
                             structures=("binance_flow_burst",))
        return BenchmarkSuite(meta, eps)

    causal = get("market-dotpfn", checkpoint=args.checkpoint)
    ablated = get("ablated-dotpfn", checkpoint=args.ablated_checkpoint)

    results = {"args": vars(args), "sets": {}}
    print(f"{'set':>8s}  {'n':>5s}  {'bias_causal':>12s}  {'bias_ablated':>13s}  {'paired delta':>22s}")
    for name, eps in (("real", real_eps), ("placebo", placebo_eps)):
        sc = score(causal, suite(eps, name))
        sa = score(ablated, suite(eps, name))
        deltas = [a - b for a, b in zip(sc, sa)]
        bc, _, bc_lo, bc_hi = bootstrap_ci(sc, n=10000)
        ba, _, ba_lo, ba_hi = bootstrap_ci(sa, n=10000)
        dm, _, d_lo, d_hi = bootstrap_ci(deltas, n=10000)
        results["sets"][name] = {
            "n": len(eps),
            "bias_causal": bc, "bias_causal_ci": [bc_lo, bc_hi],
            "bias_ablated": ba, "bias_ablated_ci": [ba_lo, ba_hi],
            "delta": dm, "delta_ci": [d_lo, d_hi],
        }
        print(f"{name:>8s}  {len(eps):5d}  {bc:+7.3f} [{bc_lo:+.2f},{bc_hi:+.2f}]"
              f"  {ba:+7.3f} [{ba_lo:+.2f},{ba_hi:+.2f}]"
              f"  {dm:+7.3f} [{d_lo:+.3f}, {d_hi:+.3f}]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
