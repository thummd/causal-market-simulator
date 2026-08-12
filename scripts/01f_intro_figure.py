#!/usr/bin/env python
"""Intro figure: realized vs counterfactual price on one shared-noise pair.

Not a cartoon — a real episode from MarketDoTime: the same Wiener noise with
and without the metaorder (the prior's counterfactual training pair). The
wedge between the paths IS the causal impact; a forecaster sees only the
history left of the window and has no way to produce the no-trade path.

Seed is picked deterministically: first seed whose episode has a clearly
visible buy (value > 1.5) and positive post-onset separation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from dotime_market.prior.market_scm import MarketDoTime

matplotlib.use("Agg")

REALIZED, COUNTERFACTUAL = "#2a78d6", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def pick_episode(max_seed: int = 60):
    for seed in range(max_seed):
        prior = MarketDoTime(
            coupling_gamma=1.5, core_share=0.7, impact_lambda=1.0,
            seed=seed, num_substeps=4,
        )
        s = prior.generate_sample(T=120)
        value = float(s["intervention_value"])
        onset = int(s["int_onset_idx"])
        y_cf = s["X_obs_full"][:, 2]
        y_re = s["X_int"][:, 2]
        sep = float((y_re[onset:] - y_cf[onset:]).mean())
        if value > 1.5 and sep > 0.4 and onset > 30:
            print(f"seed {seed}: value={value:.2f}, onset={onset}, mean sep={sep:.2f}")
            return s, seed
    raise SystemExit("no suitable seed found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/figures")
    args = ap.parse_args()

    s, seed = pick_episode()
    onset = int(s["int_onset_idx"])
    t = s["times"]
    t_end = float(s["t_int_end"])
    y_cf = s["X_obs_full"][:, 2]
    y_re = s["X_int"][:, 2]

    fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.axvspan(float(t[onset]), t_end, color="#eda100", alpha=0.12, lw=0)
    ax.plot(t, y_re, color=REALIZED, lw=1.8, label="realized (with trade)")
    ax.plot(t, y_cf, color=COUNTERFACTUAL, lw=1.8, ls="--",
            label="counterfactual (no trade)")

    # Impact wedge late in the window, where the separation is at its peak.
    end_idx = int(min(len(t) - 1, t_end - float(t[0])))
    qi = onset + int(0.75 * (end_idx - onset))
    ax.annotate("", xy=(float(t[qi]), float(y_re[qi])),
                xytext=(float(t[qi]), float(y_cf[qi])),
                arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.0))
    ax.annotate("impact", xy=(float(t[qi]) + 2.5, float((y_re[qi] + y_cf[qi]) / 2)),
                fontsize=8.5, color=TEXT, va="center")
    ylo = float(min(y_re.min(), y_cf.min()))
    ax.annotate("metaorder active", xy=(0.5 * (float(t[onset]) + t_end), ylo),
                fontsize=7.5, color=MUTED, ha="center", va="bottom")

    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.set_yticklabels([])
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig0_intro.{ext}")
    plt.close(fig)
    print(f"wrote {out}/fig0_intro.(pdf|png)  [seed {seed}]")


if __name__ == "__main__":
    main()
