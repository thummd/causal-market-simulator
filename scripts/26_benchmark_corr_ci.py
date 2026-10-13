#!/usr/bin/env python
"""Episode-bootstrap CI for the metaorder benchmark's curve correlation.

Reproduces the paper's protocol for the 12-cell curve correlation between
measured impact and the paired do-head response: episodes are resampled with
replacement (``B`` resamples, fixed seed), and on every resample the cells are
RE-FORMED — four duration classes times three |c| tertiles within each class,
tertile edges recomputed on the resample — before the correlation of cell
means is taken. This is a correlation of cell means, not an episode-level
fit, so the interval reflects both sampling noise and cell-assignment noise.

Validation: on ``benchmark_v10s_heldout_jun.json`` it must reproduce the
stored ``benchmark_corr_ci_v10s_heldout.json`` (+0.969 [0.934, 0.984]).

Usage:
    python scripts/26_benchmark_corr_ci.py output/benchmark_v12_perstep.json \
        --out output/benchmark_corr_ci_v12_perstep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DUR_CLASSES = [(2, 3), (4, 7), (8, 15), (16, 31)]


def cell_means(dur: np.ndarray, c_abs: np.ndarray, measured: np.ndarray,
               response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell means of measured and response over duration class x |c| tertile.

    Args:
        dur: Episode durations in bars.
        c_abs: Absolute intervention magnitudes.
        measured: Measured (realized) impact per episode.
        response: Paired do-head response per episode.

    Returns:
        Two arrays of cell means (measured, response), cells with no episodes dropped.
    """
    ms, rs = [], []
    for lo, hi in DUR_CLASSES:
        sel = (dur >= lo) & (dur <= hi)
        if sel.sum() < 3:
            continue
        edges = np.quantile(c_abs[sel], [1 / 3, 2 / 3])
        bins = np.digitize(c_abs[sel], edges)
        for b in range(3):
            cell = sel.copy()
            cell[sel] = bins == b
            if cell.sum() == 0:
                continue
            ms.append(measured[cell].mean())
            rs.append(response[cell].mean())
    return np.array(ms), np.array(rs)


def curve_corr(dur, c_abs, measured, response) -> float:
    """Pearson correlation of the cell means."""
    m, r = cell_means(dur, c_abs, measured, response)
    return float(np.corrcoef(m, r)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmark_json")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.benchmark_json))
    eps = d["episodes"]
    dur = np.array([e["dur"] for e in eps], dtype=float)
    c_abs = np.array([e["c_abs"] for e in eps], dtype=float)
    measured = np.array([e["measured"] for e in eps], dtype=float)
    response = np.array([e["response"] for e in eps], dtype=float)
    n = len(eps)
    point = curve_corr(dur, c_abs, measured, response)
    rng = np.random.RandomState(args.seed)
    boots = []
    for _ in range(args.B):
        idx = rng.randint(0, n, size=n)
        boots.append(curve_corr(dur[idx], c_abs[idx], measured[idx], response[idx]))
    boots = np.array(boots)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    out = {"corr": point, "ci": [lo, hi], "n_episodes": n, "B": args.B, "seed": args.seed,
           "source": args.benchmark_json}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"curve correlation {point:+.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={n} B={args.B}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
