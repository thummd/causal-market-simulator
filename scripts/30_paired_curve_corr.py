#!/usr/bin/env python
"""Paired bootstrap of the curve-correlation DIFFERENCE between two configurations.

Every metaorder-benchmark run scores the same 2,158 seed-0 episodes, so two
configurations' 12-cell curve correlations are strongly positively dependent
and comparing their marginal CIs (scripts/26) is conservative. This script
resamples episodes ONCE per replicate, re-forms the cells on that resample
for both configurations, and bootstraps the difference Δr = r_A − r_B — the
statistic that actually justifies "A beats B" on shared episodes.

Alignment is asserted, not assumed: the measured outcomes and durations must
be identical across the two files, which holds because all runs consume the
same tape with the same seed.

Usage:
    python scripts/30_paired_curve_corr.py --pairs A.json:B.json ... --out out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_S26 = importlib.util.spec_from_file_location(
    "corr_ci", Path(__file__).resolve().parent / "26_benchmark_corr_ci.py")
cc = importlib.util.module_from_spec(_S26)
_S26.loader.exec_module(cc)


def load(path: str) -> dict[str, np.ndarray]:
    """Per-episode arrays from a benchmark JSON (or its ``_ep`` companion)."""
    eps = json.load(open(path))["episodes"]
    return {k: np.array([e[k] for e in eps], dtype=float)
            for k in ("dur", "c_abs", "measured", "response")}


def paired_delta(a: dict, b: dict, B: int, seed: int) -> dict:
    """Point estimates and a joint-resample CI for r_A − r_B.

    Args:
        a: Episode arrays of configuration A.
        b: Episode arrays of configuration B (same episodes, same order).
        B: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        Dict with both correlations, the difference, and its percentile CI.

    Raises:
        SystemExit: If the two files are not episode-aligned.
    """
    if len(a["measured"]) != len(b["measured"]) or not np.allclose(a["measured"], b["measured"]) \
            or not np.array_equal(a["dur"], b["dur"]):
        raise SystemExit("files are not episode-aligned; paired comparison is invalid")
    n = len(a["measured"])
    r_a = cc.curve_corr(a["dur"], a["c_abs"], a["measured"], a["response"])
    r_b = cc.curve_corr(b["dur"], b["c_abs"], b["measured"], b["response"])
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        # One index draw serves both configurations: that shared resample is
        # what makes the difference paired.
        ra = cc.curve_corr(a["dur"][idx], a["c_abs"][idx], a["measured"][idx], a["response"][idx])
        rb = cc.curve_corr(b["dur"][idx], b["c_abs"][idx], b["measured"][idx], b["response"][idx])
        deltas.append(ra - rb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"r_a": r_a, "r_b": r_b, "delta": r_a - r_b, "delta_ci": [float(lo), float(hi)],
            "n_episodes": n, "B": B}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", required=True, metavar="LABEL=A.json:B.json")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    print(f"{'comparison (A vs B)':44s} {'r_A':>7s} {'r_B':>7s} {'Δr':>7s}  95% CI")
    for spec in args.pairs:
        label, files = spec.split("=", 1)
        fa, fb = files.split(":")
        res = paired_delta(load(fa), load(fb), args.B, args.seed)
        res.update(file_a=fa, file_b=fb)
        out[label] = res
        lo, hi = res["delta_ci"]
        flag = "EXCLUDES 0" if lo > 0 or hi < 0 else "spans 0"
        print(f"{label:44s} {res['r_a']:+.3f} {res['r_b']:+.3f} {res['delta']:+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]  {flag}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
