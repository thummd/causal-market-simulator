#!/usr/bin/env python
"""Fig 3 — paired-delta forest across the three evidence tiers.

One row per method, one panel per tier (ABIDES / BTC / ETH), dot = paired
delta vs the do-ablated twin, whisker = bootstrap 95% CI, dotted line at 0
(the ablated reference). Reads the three study JSONs produced by
scripts/03_abides_study.py and scripts/04_crypto_transfer.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Entity-fixed colors (same hues as fig2 where entities overlap).
C = {
    "market-dotpfn": "#2a78d6",
    "OWPropagator": "#008300",
    "AlmgrenChriss": "#eda100",
    "AR1": "#e87ba4",
    "Mean": "#e34948",
}
LABEL = {
    "market-dotpfn": "M-DoT-PFN (causal)",
    "OWPropagator": "OW propagator",
    "AlmgrenChriss": "Almgren–Chriss",
    "AR1": "AR1 (last value)",
    "Mean": "Mean (pre-window)",
}
ORDER = ["market-dotpfn", "OWPropagator", "AlmgrenChriss", "AR1", "Mean"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abides", default="output/abides_heldout_v4_full.json")
    ap.add_argument("--btc", default="output/crypto_BTCUSDT_q2_full.json")
    ap.add_argument("--eth", default="output/crypto_ETHUSDT_q2_full.json")
    ap.add_argument("--sol", default="output/crypto_SOLUSDT_q2_full.json")
    ap.add_argument("--out", default="output/figures")
    args = ap.parse_args()

    def load(path, title):
        d = json.loads(Path(path).read_text())
        n = d.get("n_episodes") or d["methods"]["market-dotpfn"]["n"]
        return (f"{title}\n(n={n})", d["methods"])

    tiers = [
        load(args.abides, "ABIDES"),
        load(args.btc, "BTC bursts"),
        load(args.eth, "ETH bursts"),
        load(args.sol, "SOL bursts"),
    ]

    fig, axes = plt.subplots(1, len(tiers), figsize=(7.0, 2.0), sharey=True,
                             constrained_layout=True)
    y_pos = list(range(len(ORDER)))[::-1]
    for ax, (title, methods) in zip(axes, tiers):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.axvline(0.0, color=MUTED, lw=0.9, ls=":")
        for m, y in zip(ORDER, y_pos):
            d = methods[m]["delta_vs_ablated"]
            lo, hi = methods[m]["delta_ci"]
            ax.plot([lo, hi], [y, y], color=C[m], lw=1.8, solid_capstyle="round")
            ax.plot([d], [y], marker="o", color=C[m], ms=5.5)
        ax.set_title(title, fontsize=8.5, color=TEXT)
    axes[0].set_yticks(y_pos, [LABEL[m] for m in ORDER])
    axes[1].set_xlabel(
        r"paired effect vs.\ do-ablated twin (pre-onset $\sigma$, 95% CI)"
        if False else
        "paired effect vs. do-ablated twin (pre-onset σ, 95% CI)"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_forest.{ext}")
    plt.close(fig)
    print(f"wrote {out}/fig3_forest.(pdf|png)")


if __name__ == "__main__":
    main()
