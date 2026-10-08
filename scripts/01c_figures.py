#!/usr/bin/env python
"""Paper figures from the two sweep JSONs (Phase 1/2 headline results).

- Figure 1 (bias-vs-gamma, slope level; from output/bias_vs_gamma.json):
  naive vs interventional slope as a function of the informational coupling
  gamma — empirical points on analytic curves. The widening gap IS the
  confounding bias.
- Figure 2 (methods-vs-gamma; from output/methods_vs_gamma.json): signed
  impact-direction bias and RMSE per method.

Colors: validated categorical palette (fixed slot order, one hue per entity
across both figures; aqua/yellow carry direct labels per the relief rule).
Outputs PDF + PNG to output/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Validated categorical slots (light mode), one per entity everywhere.
C = {
    "market-dotpfn": "#2a78d6",   # slot 1 blue    — the protagonist
    "param-oracle": "#1baf7a",    # slot 2 aqua    — direct-label (relief)
    "AlmgrenChriss": "#eda100",   # slot 3 yellow  — direct-label (relief)
    "OWPropagator": "#008300",    # slot 4 green
    "ablated-dotpfn": "#4a3aa7",  # slot 5 violet
    "Mean": "#e34948",            # slot 6 red
    "ReturnRegression": "#8a6d3b",  # slot 7 brown
    "SquareRootLaw": "#b04fc4",     # slot 8 magenta
    # Figure 1 entities reuse slots 1/2: naive->blue, do->aqua.
    "naive": "#2a78d6",
    "do": "#1baf7a",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    # ACM production rejects Type 3 fonts; 42 embeds TrueType outlines instead.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_bias_vs_gamma(data: dict, out: Path) -> None:
    rows = data["results"]
    g = [r["gamma"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    _style(ax)

    ax.plot(g, [r["naive_ana"]["mean"] for r in rows], "--", color=C["naive"],
            lw=1.4, alpha=0.75)
    ax.plot(g, [r["naive_emp"]["mean"] for r in rows], "o-", color=C["naive"],
            lw=1.8, ms=4.5, label="Naive (observational OLS)")
    ax.plot(g, [r["do_ana"]["mean"] for r in rows], "--", color=C["do"],
            lw=1.4, alpha=0.75)
    ax.plot(g, [r["do_emp"]["mean"] for r in rows], "s-", color=C["do"],
            lw=1.8, ms=4.5, label="Interventional (do)")

    naive_end = rows[-1]["naive_emp"]["mean"]
    do_end = rows[-1]["do_emp"]["mean"]
    ax.annotate("", xy=(g[-1], naive_end - 0.02), xytext=(g[-1], do_end + 0.02),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
    ax.annotate("Confounding\nbias", xy=(g[-1] - 0.06, (naive_end + do_end) / 2),
                ha="right", va="center", fontsize=8, color=TEXT)
    ax.annotate("Analytic (dashed)", xy=(1.02, 0.74), fontsize=7.5, color=MUTED)
    ax.set_xlabel(r"Informational coupling $\gamma$")
    ax.set_ylabel(r"Estimated impact slope  $\widehat{\beta}$")
    ax.legend(loc="upper left", fontsize=8, handlelength=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_bias_vs_gamma.{ext}")
    plt.close(fig)


def figure_methods_vs_gamma(data: dict, out: Path) -> None:
    rows = data["per_gamma"]
    g = [r["gamma"] for r in rows]
    # ReturnRegression is reported in Table 1 / text only: its RMSE (2.8-5.1)
    # is off this figure's scale and would flatten every other curve.
    methods = ["market-dotpfn", "param-oracle", "AlmgrenChriss",
               "OWPropagator", "SquareRootLaw",
               "ablated-dotpfn", "Mean"]
    label = {
        "market-dotpfn": "M-DoT-PFN (causal)",
        "param-oracle": "Param. oracle",
        "AlmgrenChriss": "Almgren-Chriss",
        "OWPropagator": "OW propagator",
        "ablated-dotpfn": "M-DoT-PFN (do-ablated)",
        "Mean": "Pre-window mean",
        "ReturnRegression": "return regression",
        "SquareRootLaw": "Square-root law",
    }
    markers = {"market-dotpfn": "o", "param-oracle": "s", "AlmgrenChriss": "^",
               "OWPropagator": "v", "SquareRootLaw": "P",
               "ablated-dotpfn": "D", "Mean": "x"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, metric, ylab in ((ax1, "bias", "Signed bias (impact direction)"),
                             (ax2, "rmse", "RMSE")):
        _style(ax)
        for m in methods:
            vals = [r["methods"][m][metric] for r in rows]
            ax.plot(g, vals, marker=markers[m], color=C[m], lw=1.8, ms=4,
                    label=label[m] if ax is ax1 else None)
            # Episode-bootstrap 95% CI band when the artifact carries it.
            ci_key = f"{metric}_ci"
            if all(ci_key in r["methods"][m] for r in rows):
                lo = [r["methods"][m][ci_key][0] for r in rows]
                hi = [r["methods"][m][ci_key][1] for r in rows]
                ax.fill_between(g, lo, hi, color=C[m], alpha=0.15, lw=0)
        ax.set_xlabel(r"Informational coupling $\gamma$")
        ax.set_ylabel(ylab)
    ax1.axhline(0.0, color=MUTED, lw=0.8, ls=":")

    # Relief rule: direct labels for the low-contrast hues (and the headline pair).
    last = rows[-1]["methods"]
    ac_mid = rows[-2]["methods"]["AlmgrenChriss"]["bias"]  # gamma=1.5 point
    ax1.annotate(label["AlmgrenChriss"], xy=(g[-2], ac_mid),
                 xytext=(4, -10), textcoords="offset points", ha="left", va="top",
                 fontsize=7.5, color=TEXT)
    ax1.annotate(label["param-oracle"], xy=(g[-1], last["param-oracle"]["bias"]),
                 xytext=(-2, -8), textcoords="offset points", ha="right", va="top",
                 fontsize=7.5, color=TEXT)
    ax1.legend(loc="upper left", fontsize=6.8, handlelength=1.4, ncol=2,
               columnspacing=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_methods_vs_gamma.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slopes", default="output/bias_vs_gamma.json")
    ap.add_argument("--methods", default="output/methods_vs_gamma_v4_full.json")
    ap.add_argument("--out", default="output/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure_bias_vs_gamma(json.loads(Path(args.slopes).read_text()), out)
    figure_methods_vs_gamma(json.loads(Path(args.methods).read_text()), out)
    print(f"wrote fig1/fig2 (pdf+png) to {out}")


if __name__ == "__main__":
    main()
