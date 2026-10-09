#!/usr/bin/env python
"""Split-sample scale calibration: fit the PFN's in-domain scale factor k, test out of sample.

Component (4) of the closing frame: the PFN's response *shape* and *direction*
transfer but its *level* does not, so every deployment needs an in-domain
scale. This script measures how much a one-number calibration buys, honestly:
k is fitted on one half of the trading days and applied to the other half,
in both directions, so the reported gain is never in-sample.

Definitions (all in the episode's standardized price units):

* ``last``  = price at the last context bar before the onset,
* ``move``  = ``y_true - last`` (realized), ``pred_move = pred - last`` (PFN),
* k         = through-origin least squares of ``move`` on ``pred_move`` over
  the fitting half (``sum(move * pred_move) / sum(pred_move**2)``),
* calibrated prediction = ``last + k * pred_move``.

Reported on the held-out half: RMSE and signed bias of the raw PFN, the
calibrated PFN, the do-ablated twin, AR1 and Mean, plus the paired delta of
the calibrated PFN vs the ablated twin with a bootstrap CI. Works on the
closing-cross tier (default), on the Nasdaq burst tier (``--tier burst``) and on
the published-path tier (``--tier path``, the announcement-channel encoding), so k
can be compared across mechanisms and across operator-argument rules.

Two further analyses answer "can the scale be inferred, and how much
in-domain data does it take?": (a) a few-shot learning curve — k fitted on
n randomly drawn labeled events from the fitting half (n = 5 ... 200, 50
draws each) and scored on the held-out half; (b) a cross-symbol test — per
name, the realized and the PFN-predicted response per unit dose; if the
model inferred scale from context, the two would correlate across names.
Per-query arrays are saved next to the JSON as ``.parquet`` for reuse.

Usage:
    python scripts/36_scale_calibration.py --tier cross
    python scripts/36_scale_calibration.py --tier burst
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dotime.evaluation import bootstrap_ci
from dotime_market.data.auction import build_auction_suite
from dotime_market.data.databento import RAW_DIR, build_databento_suite, load_closing_imbalance, trading_days
from dotime_market.evaluation.transfer import build_models

EQ10 = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "INTC", "QQQ"]


def collect(models: dict, suite) -> dict:
    """Per-query arrays needed for calibration.

    Args:
        models: name -> model with ``predict(episode)``.
        suite: Factual suite.

    Returns:
        Dict of equal-length arrays: ``y``, ``last``, ``sign`` (intervention
        direction), ``day`` (trading day per query), ``symbol``, and one
        prediction array per model.
    """
    out = {"y": [], "last": [], "sign": [], "day": [], "symbol": [], "dose": []}
    preds = {name: [] for name in models}
    for ep in suite:
        onset = ep.intervention.times[0]
        # The outcome is the last canonical column in every tier's layout (two columns
        # for cross and burst, four when the announcement channel is present).
        last = float(ep.x_obs[onset - 1, -1])
        y = ep.y_true.reshape(-1)
        n = y.numel()
        out["y"].extend(y.tolist())
        out["last"].extend([last] * n)
        out["sign"].extend([float(np.sign(float(ep.intervention.values)))] * n)
        out["dose"].extend([float(ep.intervention.values)] * n)
        out["day"].extend([ep.metadata["date"]] * n)
        out["symbol"].extend([ep.metadata["symbol"]] * n)
        for name, model in models.items():
            preds[name].extend(model.predict(ep).reshape(-1).tolist())
    res = {k: np.array(v) for k, v in out.items()}
    res["pred"] = {name: np.array(v) for name, v in preds.items()}
    return res


def fit_k(move: np.ndarray, pred_move: np.ndarray) -> float:
    """Through-origin least-squares scale of realized on predicted moves."""
    return float((move * pred_move).sum() / max((pred_move**2).sum(), 1e-12))


def evaluate_half(d: dict, fit_mask: np.ndarray, test_mask: np.ndarray, bootstrap: int) -> dict:
    """Fit k on ``fit_mask`` queries, score every method on ``test_mask`` queries.

    Args:
        d: Output of :func:`collect`.
        fit_mask: Boolean mask of fitting queries.
        test_mask: Boolean mask of held-out queries.
        bootstrap: Resamples for the paired-delta CI.

    Returns:
        Dict with ``k``, per-method ``rmse``/``bias`` on the test half, and the
        paired delta of the calibrated PFN vs the ablated twin (with CI) and
        vs the raw PFN.
    """
    y, last, sign = d["y"], d["last"], d["sign"]
    move = y - last
    pfn = d["pred"]["market-dotpfn"]
    k = fit_k(move[fit_mask], (pfn - last)[fit_mask])
    preds = dict(d["pred"])
    preds["market-dotpfn-calibrated"] = last + k * (pfn - last)
    out = {"k": k, "n_fit": int(fit_mask.sum()), "n_test": int(test_mask.sum()), "methods": {}}
    signed = {}
    for name, p in preds.items():
        err = (p - y)[test_mask]
        signed[name] = (sign[test_mask] * err)
        out["methods"][name] = {"rmse": float(np.sqrt((err**2).mean())),
                                "bias": float(signed[name].mean())}
    for name in ("market-dotpfn-calibrated", "market-dotpfn"):
        deltas = [float(v) for v in signed[name] - signed["ablated-dotpfn"]]
        dm, _, lo, hi = bootstrap_ci(deltas, n=bootstrap)
        out["methods"][name]["delta_vs_ablated"] = dm
        out["methods"][name]["delta_ci"] = [lo, hi]
    # RMSE improvement of calibration over the raw PFN, paired by query (squared errors).
    se_cal = ((preds["market-dotpfn-calibrated"] - y)[test_mask]) ** 2
    se_raw = ((pfn - y)[test_mask]) ** 2
    dm, _, lo, hi = bootstrap_ci([float(v) for v in se_raw - se_cal], n=bootstrap)
    out["mse_gain_calibrated_vs_raw"] = {"mean": dm, "ci": [lo, hi]}
    return out


def few_shot_curve(d: dict, fit_mask: np.ndarray, test_mask: np.ndarray,
                   sizes=(5, 10, 20, 50, 100, 200), draws: int = 50, seed: int = 0) -> dict:
    """Held-out RMSE of the calibrated PFN as a function of labeled fitting events.

    Args:
        d: Output of :func:`collect`.
        fit_mask: Pool of fitting queries to draw from.
        test_mask: Held-out queries scored at every size.
        sizes: Numbers of labeled events per draw.
        draws: Random subsets per size.
        seed: RNG seed.

    Returns:
        Dict size -> ``{"k_median", "k_iqr", "rmse_median", "rmse_iqr"}`` plus
        the full-pool reference (all fitting events) and the AR1 RMSE.
    """
    y, last = d["y"], d["last"]
    pfn = d["pred"]["market-dotpfn"]
    move, pmove = y - last, pfn - last
    pool = np.where(fit_mask)[0]
    rng = np.random.RandomState(seed)

    def rmse_for(k):
        return float(np.sqrt((((last + k * pmove) - y)[test_mask] ** 2).mean()))

    out = {"sizes": {}}
    for n in sizes:
        if n > len(pool):
            continue
        ks, rm = [], []
        for _ in range(draws):
            idx = rng.choice(pool, size=n, replace=False)
            k = fit_k(move[idx], pmove[idx])
            ks.append(k)
            rm.append(rmse_for(k))
        out["sizes"][str(n)] = {
            "k_median": float(np.median(ks)), "k_iqr": [float(np.percentile(ks, 25)), float(np.percentile(ks, 75))],
            "rmse_median": float(np.median(rm)), "rmse_iqr": [float(np.percentile(rm, 25)), float(np.percentile(rm, 75))],
        }
    k_all = fit_k(move[fit_mask], pmove[fit_mask])
    out["full_pool"] = {"n": int(len(pool)), "k": k_all, "rmse": rmse_for(k_all)}
    out["ar1_rmse"] = float(np.sqrt(((d["pred"]["AR1"] - y)[test_mask] ** 2).mean()))
    out["raw_pfn_rmse"] = rmse_for(1.0)
    return out


def cross_symbol_scale(d: dict, min_n: int = 20) -> dict:
    """Realized vs PFN-predicted response per unit dose, per symbol.

    Args:
        d: Output of :func:`collect`; needs ``dose`` per query.
        min_n: Minimum queries per symbol.

    Returns:
        Dict with per-symbol ``{n, realized_slope, pfn_slope}`` and the
        cross-symbol Spearman correlation between the two slopes.
    """
    from scipy.stats import spearmanr

    y, last, dose = d["y"], d["last"], d["dose"]
    pfn = d["pred"]["market-dotpfn"]
    per = {}
    for s in sorted(set(d["symbol"])):
        m = d["symbol"] == s
        if m.sum() < min_n:
            continue
        per[s] = {"n": int(m.sum()),
                  "realized_slope": fit_k((y - last)[m], dose[m]),
                  "pfn_slope": fit_k((pfn - last)[m], dose[m])}
    a = np.array([v["realized_slope"] for v in per.values()])
    b = np.array([v["pfn_slope"] for v in per.values()])
    rho = float(spearmanr(a, b).correlation) if len(a) > 2 else float("nan")
    return {"per_symbol": per, "n_symbols": len(per), "spearman_realized_vs_pfn": rho,
            "realized_slope_range": [float(a.min()), float(a.max())] if len(a) else None,
            "pfn_slope_range": [float(b.min()), float(b.max())] if len(b) else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["cross", "burst", "path"], default="cross")
    ap.add_argument("--symbols", nargs="+", default=EQ10)
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--split", choices=["alternate", "halves"], default="alternate",
                    help="alternate: even/odd trading days; halves: first/second half of the range")
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"output/scale_calibration_{args.tier}.json")

    raw = Path(args.raw_dir)
    dates = sorted({d for s in args.symbols for d in trading_days(s, raw / "trades", args.start, args.end)})
    if args.tier in ("cross", "path"):
        noii = load_closing_imbalance(raw / "xnas_imbalance_eq50_12mo.dbn.zst",
                                      cache=raw / "closing_imbalance.parquet")
        suite, _ = build_auction_suite(args.symbols, dates, cache_dir=raw / "trades", noii=noii,
                                       encoding="cross" if args.tier == "cross" else "announced_path")
    else:
        suite = build_databento_suite(args.symbols, dates, cache_dir=raw / "trades")
    print(f"{args.tier} tier: {len(suite)} episodes over {len(dates)} days", flush=True)

    models = build_models(args.checkpoint, args.ablated_checkpoint, args.device)
    models = {k: models[k] for k in ("Mean", "AR1", "market-dotpfn", "ablated-dotpfn")}
    torch.set_grad_enabled(False)
    d = collect(models, suite)

    day_rank = {day: i for i, day in enumerate(dates)}
    ranks = np.array([day_rank[x] for x in d["day"]])
    if args.split == "alternate":
        a = ranks % 2 == 0
    else:
        a = ranks < len(dates) / 2
    results = {"args": vars(args), "n_episodes": len(suite), "n_queries": int(len(d["y"])),
               "k_full_sample": fit_k(d["y"] - d["last"], d["pred"]["market-dotpfn"] - d["last"]),
               "folds": {}}
    for label, fit, test in (("fit_A_test_B", a, ~a), ("fit_B_test_A", ~a, a)):
        r = evaluate_half(d, fit, test, args.bootstrap)
        results["folds"][label] = r
        print(f"\n== {label}: k = {r['k']:.4f}  (fit n={r['n_fit']}, test n={r['n_test']})")
        print(f"{'method':>26s}  {'rmse':>8s}  {'bias':>8s}  {'d_vs_ablated':>12s}  {'95% CI':>20s}")
        for name, m in r["methods"].items():
            extra = ""
            if "delta_vs_ablated" in m:
                extra = f"{m['delta_vs_ablated']:12.4f}  [{m['delta_ci'][0]:8.4f}, {m['delta_ci'][1]:8.4f}]"
            print(f"{name:>26s}  {m['rmse']:8.4f}  {m['bias']:8.4f}  {extra}")
        g = r["mse_gain_calibrated_vs_raw"]
        print(f"   MSE gain, calibrated vs raw PFN: {g['mean']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]")
    # Per-symbol k on the full sample: is one number enough, or is scale name-specific?
    per_symbol = {}
    for s in sorted(set(d["symbol"])):
        m = d["symbol"] == s
        per_symbol[s] = {"n": int(m.sum()),
                         "k": fit_k((d["y"] - d["last"])[m], (d["pred"]["market-dotpfn"] - d["last"])[m])}
    results["k_per_symbol"] = per_symbol
    print("\nk full sample:", round(results["k_full_sample"], 4), "| per symbol:",
          {s: round(v["k"], 3) for s, v in per_symbol.items()})

    fs = few_shot_curve(d, a, ~a)
    results["few_shot"] = fs
    print(f"\nfew-shot learning curve (fit on A subsets, test on B; AR1 {fs['ar1_rmse']:.4f}, "
          f"raw PFN {fs['raw_pfn_rmse']:.4f}, full pool n={fs['full_pool']['n']} -> {fs['full_pool']['rmse']:.4f}):")
    for n, r in fs["sizes"].items():
        print(f"  n={int(n):4d}  k {r['k_median']:.4f} [{r['k_iqr'][0]:.4f}, {r['k_iqr'][1]:.4f}]  "
              f"rmse {r['rmse_median']:.4f} [{r['rmse_iqr'][0]:.4f}, {r['rmse_iqr'][1]:.4f}]")
    cs = cross_symbol_scale(d)
    results["cross_symbol"] = cs
    print(f"\ncross-symbol: {cs['n_symbols']} names, realized slope range {cs['realized_slope_range']}, "
          f"PFN slope range {cs['pfn_slope_range']}, Spearman(realized, PFN) = {cs['spearman_realized_vs_pfn']:+.3f}")

    import pandas as pd
    frame = pd.DataFrame({k: d[k] for k in ("y", "last", "sign", "dose", "day", "symbol")})
    for name, p in d["pred"].items():
        frame[f"pred_{name}"] = p
    frame.to_parquet(out.with_suffix(".parquet"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
