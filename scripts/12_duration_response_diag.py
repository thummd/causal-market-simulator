#!/usr/bin/env python
"""Duration-response diagnostic: is the do-head sensitive to window length?

Motivated by the prior-fix invariance: v4 (OU), v6 (slow theta), and v7
(unit-root carve-out) all show FLAT response-per-unit-c across metaorder
duration on the Gate-B benchmark, while measured impact-per-unit-c rises
~3.6x. Three different priors, one identical profile — the suspect is the
episode geometry / encoding, not the prior dynamics.

Controlled measurement: hold the pre-onset context fixed (same real BTC
bars), pin the soft-intervention value at c = +1 (standardized), vary ONLY
the window length, and read the paired do-head response at the first
post-window bar. Two geometries:

- ``end-pinned``  — the Gate-B construction: the episode ends right after
  the window, so normalized intervention_time_end == 1.0 for every
  duration and the mixer's only duration signal is
  ``dur / (pre + dur)`` (0.02-0.26; training support is 0.1-0.3).
- ``mid-window``  — 30 post-window bars are appended, so the window sits
  inside the episode like a training-time window does.

If the response is flat in BOTH geometries for the v7 pair (whose prior
provably contains linear-accrual worlds — tests/test_perm_impact.py), the
duration signal is being lost in encoding/training geometry, and the fix
belongs there (window-fraction curriculum, geometry-matched training),
not in yet another prior mechanism.

Usage:
    python scripts/12_duration_response_diag.py --n-onsets 30
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.binance import bin_agg_trades, download_agg_trades
from dotime_market.data.liquidation import burst_episode
from dotime_market.prior.market_scm import MarketContinuousSCMSampler

DURATIONS = [2, 3, 4, 7, 11, 15, 23, 31]

PAIRS = {
    "v4": ("checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt"),
    "v6": ("checkpoints/market_v6_slowtheta/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v6_slowtheta_ablated/continuous_do_over_time_pfn_best.pt"),
    "v7": ("checkpoints/market_v7_permimpact/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v7_permimpact_ablated/continuous_do_over_time_pfn_best.pt"),
    "v8": ("checkpoints/market_v8_geom/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v8_geom_ablated/continuous_do_over_time_pfn_best.pt"),
    "v9": ("checkpoints/market_v9_hawkes/continuous_do_over_time_pfn_best.pt",
           "checkpoints/market_v9_hawkes_ablated/continuous_do_over_time_pfn_best.pt"),
    "v10": ("checkpoints/market_v10_rnn/recurrent_dotpfn_best.pt",
            "checkpoints/market_v10_rnn_ablated/recurrent_dotpfn_best.pt"),
    "v10s": ("checkpoints/market_v10s_softmix/recurrent_dotpfn_best.pt",
             "checkpoints/market_v10s_softmix_ablated/recurrent_dotpfn_best.pt"),
}


def response_curve(causal, ablated, bars, onsets, rng, extra_post: int) -> list:
    """Mean paired response per duration, query at the first post-window bar.

    Args:
        causal, ablated: registered baseline models (predict protocol).
        bars: one day's bar DataFrame.
        onsets: bar indices with >= 90 pre-bars and enough post-bars.
        rng: numpy RandomState (burst_episode signature; queries overridden).
        extra_post: bars appended after the window (0 = end-pinned).

    Returns:
        One dict per duration with the mean and sd of the paired response.
    """
    rows = []
    for dur in DURATIONS:
        vals = []
        for i0 in onsets:
            try:
                ep = burst_episode(bars, event_idx=i0, rng=rng, pre_bars=90,
                                   post_bars=dur + 1 + extra_post, window_bars=dur)
            except Exception:
                continue
            # Pin the handle: identical soft intervention for every duration.
            ep.intervention.values = 1.0
            # Query the first post-window bar (index 90 + dur), in the
            # normalized units burst_episode uses for query_time.
            t_len = ep.x_obs.shape[0]
            ep.query_time = torch.tensor(
                [(90 + dur) / max(t_len - 1, 1)], dtype=torch.float32)
            ep.query_target = torch.tensor([1], dtype=torch.long)
            ep.y_true = torch.tensor([0.0])
            p_c = float(causal.predict(ep).reshape(-1)[0])
            p_a = float(ablated.predict(ep).reshape(-1)[0])
            vals.append(p_c - p_a)
        rows.append({"dur": dur, "n": len(vals),
                     "response_mean": float(np.mean(vals)),
                     "response_sd": float(np.std(vals))})
    return rows


def simulate_bars(world: str, n_bars: int, seed: int) -> pd.DataFrame:
    """Bars from a pinned prior world, in the frame layout burst_episode reads.

    Args:
        world: ``"perm"`` (theta_price = 0 integrator — the rung-2 carve-out)
            or ``"ou"`` (legacy transient price).
        n_bars: number of dt = 1 bars to simulate.
        seed: sampler + noise seed.

    Returns:
        DataFrame with ``time`` / ``flow`` / ``price_bps`` columns. On-prior
        context: the model's posterior has no transfer excuse here — a flat
        response on ``perm`` bars means the checkpoint cannot express
        duration accrual at all.
    """
    sampler = MarketContinuousSCMSampler(
        core_share=0.7, coupling_gamma=1.0, impact_lambda=0.6,
        perm_prob=1.0 if world == "perm" else 0.0, seed=seed)
    scm, _, _ = sampler.sample()
    times = torch.arange(n_bars, dtype=torch.float32)
    dts = torch.ones(n_bars - 1)
    gen = torch.Generator().manual_seed(seed + 1)
    noise = scm._draw_noise((n_bars - 1) * 2, generator=gen)
    _, x = scm.simulate(times, dts, intervention=None, noise=noise, num_substeps=2)
    return pd.DataFrame({"time": np.arange(n_bars),
                         "flow": x[:, 1].numpy(),
                         "price_bps": x[:, 2].numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dates-from", default="2026-04-01")
    ap.add_argument("--n-days", type=int, default=3)
    ap.add_argument("--n-onsets", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", choices=["perm", "ou"], default=None,
                    help="replace real bars with bars simulated from a pinned "
                         "prior world (on-prior posterior control)")
    ap.add_argument("--out", default="output/duration_response_diag.json")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    y0, m0, d0 = map(int, args.dates_from.split("-"))
    day_bars = []
    for k in range(args.n_days):
        if args.synthetic:
            day_bars.append(simulate_bars(args.synthetic, 2000, args.seed + 10 * k))
        else:
            day = str(date(y0, m0, d0) + timedelta(k))
            day_bars.append(bin_agg_trades(download_agg_trades(args.symbol, day),
                                           bar_seconds=60))

    results = {"args": vars(args), "models": {}}
    for tag, (ck, ck_a) in PAIRS.items():
        if not (Path(ck).exists() and Path(ck_a).exists()):
            continue
        causal, ablated = get("market-dotpfn", checkpoint=ck), get(
            "ablated-dotpfn", checkpoint=ck_a)
        results["models"][tag] = {}
        for mode, extra in (("end_pinned", 0), ("mid_window", 30)):
            per_day = []
            for bars in day_bars:
                hi = len(bars) - (max(DURATIONS) + 1 + extra + 2)
                onsets = rng.randint(90, hi, size=args.n_onsets // args.n_days)
                per_day.append(response_curve(causal, ablated, bars, onsets,
                                              rng, extra))
            merged = []
            for d_i, dur in enumerate(DURATIONS):
                ns = [pd[d_i]["n"] for pd in per_day]
                ms = [pd[d_i]["response_mean"] for pd in per_day]
                w = np.array(ns, dtype=float)
                merged.append({"dur": dur, "n": int(w.sum()),
                               "response": float(np.average(ms, weights=w))})
            results["models"][tag][mode] = merged
            prof = "  ".join(f"{r['dur']}:{r['response']:+.3f}" for r in merged)
            print(f"{tag} {mode:>10}  {prof}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
