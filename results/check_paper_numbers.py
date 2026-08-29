#!/usr/bin/env python3
"""Assert every number quoted in the paper against the stored result JSONs.

Usage:
    python results/check_paper_numbers.py            # from the repo root
    python results/check_paper_numbers.py --results-dir output

Each claim names where it appears in the paper, the JSON it must come from,
and the value the paper prints. A claim passes when the stored value rounds
to the printed one (|stored - printed| <= 0.5 * 10^-decimals). Claims the
paper makes that have no stored artifact are listed at the end as UNCHECKED
rather than silently omitted. Exit status is non-zero if any claim fails.
Standard library only; runs in seconds; no GPU, no network.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=None)
def J(name: str) -> dict:
    """Load and cache one result JSON from the results directory."""
    with open(RESULTS / name) as fh:
        return json.load(fh)


def gamma_row(name: str, gamma: float, method: str) -> dict:
    """Return the per-method entry of a synthetic-grid JSON at one gamma."""
    for row in J(name)["per_gamma"]:
        if abs(row["gamma"] - gamma) < 1e-9:
            return row["methods"][method]
    raise KeyError(f"{name}: gamma={gamma} not on grid")


CLAIMS: list[tuple[str, str, object, float, int]] = []


def C(where: str, what: str, fn, printed: float, decimals: int = 2) -> None:
    """Register one claim: paper location, description, accessor, printed value."""
    CLAIMS.append((where, what, fn, printed, decimals))


def ci(where, what, fn_pair, lo, hi, decimals=2):
    """Register the two ends of a printed confidence interval."""
    C(where, what + " CI lo", lambda: fn_pair()[0], lo, decimals)
    C(where, what + " CI hi", lambda: fn_pair()[1], hi, decimals)


# --------------------------------------------------------------- Table 1
TABLE1 = {
    "M-DoT-PFN": "market-dotpfn", "OW propagator": "OWPropagator",
    "Square-root law": "SquareRootLaw", "Return regression": "ReturnRegression",
    "Almgren-Chriss": "AlmgrenChriss", "AR1": "AR1", "Mean": "Mean",
}
TABLE1_CELLS = {
    # column: (file, n, {row: (delta, lo, hi)})
    "ABIDES": ("abides_heldout_v4_full.json", 1000, {
        "M-DoT-PFN": (0.38, 0.34, 0.42), "OW propagator": (0.38, 0.23, 0.53),
        "Square-root law": (0.42, 0.34, 0.51), "Return regression": (5.97, 5.49, 6.44),
        "Almgren-Chriss": (-0.02, -0.03, -0.01), "AR1": (0.05, -0.03, 0.14),
        "Mean": (-0.00, -0.00, -0.00)}),
    "BTC": ("crypto_BTCUSDT_q2_full.json", 763, {
        "M-DoT-PFN": (1.42, 1.29, 1.56), "OW propagator": (0.88, 0.77, 1.00),
        "Square-root law": (1.86, 1.73, 1.98), "Return regression": (3.11, 2.88, 3.34),
        "Almgren-Chriss": (1.09, 0.98, 1.20), "AR1": (1.21, 1.10, 1.33),
        "Mean": (-0.00, -0.00, 0.00)}),
    "ETH": ("crypto_ETHUSDT_q2_full.json", 758, {
        "M-DoT-PFN": (1.53, 1.40, 1.67), "OW propagator": (0.96, 0.86, 1.06),
        "Square-root law": (2.11, 1.99, 2.22), "Return regression": (3.27, 3.07, 3.48),
        "Almgren-Chriss": (1.11, 1.02, 1.20), "AR1": (1.41, 1.30, 1.52),
        "Mean": (-0.00, -0.00, 0.00)}),
    "SOL": ("crypto_SOLUSDT_q2_full.json", 757, {
        "M-DoT-PFN": (1.22, 1.10, 1.34), "OW propagator": (0.74, 0.62, 0.85),
        "Square-root law": (1.60, 1.47, 1.73), "Return regression": (2.49, 2.31, 2.66),
        "Almgren-Chriss": (0.89, 0.82, 0.96), "AR1": (1.00, 0.88, 1.13),
        "Mean": (-0.00, -0.00, -0.00)}),
}
for col, (fname, n, rows) in TABLE1_CELLS.items():
    C(f"Tab.1 {col}", "n episodes", lambda f=fname: J(f).get("n_episodes") or J(f)["args"]["n_episodes"], n, 0)
    for row, (d, lo, hi) in rows.items():
        m = TABLE1[row]
        C(f"Tab.1 {col}", f"{row} delta", lambda f=fname, m=m: J(f)["methods"][m]["delta_vs_ablated"], d)
        ci(f"Tab.1 {col}", f"{row} delta", lambda f=fname, m=m: J(f)["methods"][m]["delta_ci"], lo, hi)

# ----------------------------------------------------- Sec. 4.1 ABIDES text
A = "abides_heldout_v4_full.json"
C("S4.1 ABIDES", "held-out seed range starts at 1000", lambda: J(A)["args"]["seed0"], 1000, 0)
C("S4.1 ABIDES", "measured seed-replay impact", lambda: J(A)["ground_truth"]["mean"], 0.28)
ci("S4.1 ABIDES", "measured impact", lambda: J(A)["ground_truth"]["ci"], 0.05, 0.50)
C("S4.1 ABIDES", "causal response +0.379", lambda: J(A)["methods"]["market-dotpfn"]["delta_vs_ablated"], 0.379, 3)
ci("S4.1 ABIDES", "causal response", lambda: J(A)["methods"]["market-dotpfn"]["delta_ci"], 0.342, 0.416, 3)
C("S4.1 ABIDES", "causal bias +0.08", lambda: J(A)["methods"]["market-dotpfn"]["bias"], 0.08)
ci("S4.1 ABIDES", "causal bias", lambda: J(A)["methods"]["market-dotpfn"]["bias_ci"], -0.27, 0.42)
C("S4.1 ABIDES", "square-root law bias +0.13", lambda: J(A)["methods"]["SquareRootLaw"]["bias"], 0.13)
C("S4.1 ABIDES", "return regression signed bias +5.7", lambda: J(A)["methods"]["ReturnRegression"]["bias"], 5.7, 1)
C("S4.1 ABIDES", "ablated twin bias -0.30 (control paragraph)", lambda: J(A)["methods"]["ablated-dotpfn"]["bias"], -0.30)
H = "abides_hard_v4pair_n300.json"
C("S4.1 semantics", "hard-clamp encoding response +0.10", lambda: J(H)["methods"]["market-dotpfn"]["delta_vs_ablated"], 0.10)
ci("S4.1 semantics", "hard-clamp response", lambda: J(H)["methods"]["market-dotpfn"]["delta_ci"], 0.06, 0.15)
C("S4.1 semantics", "hard-clamp bias -0.31", lambda: J(H)["methods"]["market-dotpfn"]["bias"], -0.31)
C("S4.1 semantics", "hard-clamp n=300", lambda: J(H)["args"]["n_episodes"], 300, 0)

# ------------------------------------------------------ Sec. 4.2 crypto text
CRYPTO = {"BTC": "crypto_BTCUSDT_q2_full.json", "ETH": "crypto_ETHUSDT_q2_full.json",
          "SOL": "crypto_SOLUSDT_q2_full.json"}
for sym, (mb, ar1, sq) in {"BTC": (-0.84, -1.05, -0.41), "ETH": (-1.05, -1.17, -0.47),
                           "SOL": (-0.76, -0.98, -0.38)}.items():
    f = CRYPTO[sym]
    C(f"S4.2 {sym}", "M-DoT-PFN signed bias", lambda f=f: J(f)["methods"]["market-dotpfn"]["bias"], mb)
    C(f"S4.2 {sym}", "AR1 signed bias", lambda f=f: J(f)["methods"]["AR1"]["bias"], ar1)
    C(f"S4.2 {sym}", "square-root law bias", lambda f=f: J(f)["methods"]["SquareRootLaw"]["bias"], sq)
    C(f"S4.2 {sym}", "ablated twin equals pre-window mean (bias diff)",
      lambda f=f: J(f)["methods"]["ablated-dotpfn"]["bias"] - J(f)["methods"]["Mean"]["bias"], 0.0)
C("S4.2 ETH", "return regression bias +0.69", lambda: J(CRYPTO["ETH"])["methods"]["ReturnRegression"]["bias"], 0.69)
C("S4.2 SOL", "return regression bias +0.51", lambda: J(CRYPTO["SOL"])["methods"]["ReturnRegression"]["bias"], 0.51)
C("S4.2", "square-root law RMSE best, min 2.1",
  lambda: min(J(f)["methods"]["SquareRootLaw"]["rmse"] for f in CRYPTO.values()), 2.1, 1)
C("S4.2", "square-root law RMSE best, max 2.5",
  lambda: max(J(f)["methods"]["SquareRootLaw"]["rmse"] for f in CRYPTO.values()), 2.5, 1)
C("S4.2", "square-root law bias negative across prefactor range 0.5-1.0 (1=all negative)",
  lambda: float(all(v < 0 for s in J("sqrt_sensitivity.json").values() for v in s.values())), 1, 0)
C("S4.2", "square-root law bias at prefactor 0.7 equals burst-tier value (BTC)",
  lambda: J("sqrt_sensitivity.json")["BTCUSDT"]["0.7"], -0.41)
SS = "sign_shuffle.json"
C("S4.2 sign-shuffle", "measured move, true signs +1.05", lambda: J(SS)["impact_true_signs"][0], 1.05)
C("S4.2 sign-shuffle", "measured move, shuffled signs -0.13", lambda: J(SS)["impact_shuffled_signs"][0], -0.13)
C("S4.2 sign-shuffle", "n=763", lambda: J(SS)["n"], 763, 0)
PL = {"BTC": "placebo_test.json", "ETH": "placebo_ETHUSDT.json", "SOL": "placebo_SOLUSDT.json"}
for sym, (pa, pc, ra, rc) in {"BTC": (-0.03, 1.35, -2.29, -0.95), "ETH": (-0.10, 1.34, -2.59, -1.15),
                              "SOL": (0.10, 1.23, -1.96, -0.83)}.items():
    f = PL[sym]
    C(f"S4.2 placebo {sym}", "ablated bias at placebo anchors", lambda f=f: J(f)["sets"]["placebo"]["bias_ablated"], pa)
    C(f"S4.2 placebo {sym}", "causal bias created at placebo anchors", lambda f=f: J(f)["sets"]["placebo"]["bias_causal"], pc)
    C(f"S4.2 placebo {sym}", "real-event ablated bias", lambda f=f: J(f)["sets"]["real"]["bias_ablated"], ra)
    C(f"S4.2 placebo {sym}", "real-event causal bias", lambda f=f: J(f)["sets"]["real"]["bias_causal"], rc)
C("S4.2 placebo", "placebo anchors per asset, min 654", lambda: min(J(f)["sets"]["placebo"]["n"] for f in PL.values()), 654, 0)
C("S4.2 placebo", "placebo anchors per asset, max 663", lambda: max(J(f)["sets"]["placebo"]["n"] for f in PL.values()), 663, 0)
C("S4.2 multiplicity", "every per-asset headline delta exceeds 20 bootstrap SEs (1=true)",
  lambda: float(all(
      J(f)["methods"]["market-dotpfn"]["delta_vs_ablated"]
      / ((J(f)["methods"]["market-dotpfn"]["delta_ci"][1] - J(f)["methods"]["market-dotpfn"]["delta_ci"][0]) / 3.92) >= 20
      for f in CRYPTO.values())), 1, 0)

# ------------------------------------------- Sec. 3 recipe: training ablations
C("S3 recipe", "unskewed config gamma=2 bias 0.396", lambda: gamma_row("methods_vs_gamma_v1_ci.json", 2.0, "market-dotpfn")["bias"], 0.396, 3)
ci("S3 recipe", "unskewed config gamma=2 bias", lambda: gamma_row("methods_vs_gamma_v1_ci.json", 2.0, "market-dotpfn")["bias_ci"], 0.27, 0.52)
C("S3 recipe", "4x longer training gamma=2 bias 0.393", lambda: gamma_row("methods_vs_gamma_v2.json", 2.0, "market-dotpfn")["bias"], 0.393, 3)
C("S3 recipe", "gamma-skewed prior gamma=2 bias 0.169", lambda: gamma_row("methods_vs_gamma_v3.json", 2.0, "market-dotpfn")["bias"], 0.169, 3)
C("S3 recipe", "skewed re-run gamma=2 bias 0.164", lambda: gamma_row("methods_vs_gamma_v4_full.json", 2.0, "market-dotpfn")["bias"], 0.164, 3)
C("S3 recipe", "bias cut by 57% (percent)",
  lambda: 100 * (1 - gamma_row("methods_vs_gamma_v3.json", 2.0, "market-dotpfn")["bias"]
                 / gamma_row("methods_vs_gamma_v1_ci.json", 2.0, "market-dotpfn")["bias"]), 57, 0)
SOFT = ["methods_vs_gamma_v4s.json", "methods_vs_gamma_v4s_s43.json", "methods_vs_gamma_v4s_s44.json"]
C("S3 recipe", "soft-mix gamma=2 bias, three seeds, min 0.160", lambda: min(gamma_row(f, 2.0, "market-dotpfn")["bias"] for f in SOFT), 0.160, 3)
C("S3 recipe", "soft-mix gamma=2 bias, three seeds, max 0.191", lambda: max(gamma_row(f, 2.0, "market-dotpfn")["bias"] for f in SOFT), 0.191, 3)
SOFT_BTC = ["crypto_BTCUSDT_v4s.json", "crypto_BTC_v4s_s43.json", "crypto_BTC_v4s_s44.json"]
C("S3 recipe", "soft-trained BTC do-head response, min +0.02", lambda: min(J(f)["methods"]["market-dotpfn"]["delta_vs_ablated"] for f in SOFT_BTC), 0.02)
C("S3 recipe", "soft-trained BTC do-head response, max +0.72", lambda: max(J(f)["methods"]["market-dotpfn"]["delta_vs_ablated"] for f in SOFT_BTC), 0.72)
MG = "magnitude_scaling_v4s_seeds.json"
SOFT_PAIRS = ["v4s_soft", "v4s_s43", "v4s_s44"]
C("S3 magnitude", "173 burst episodes", lambda: J(MG)["n_episodes"], 173, 0)
C("S3 magnitude", "every soft seed rises monotonically from 0.5x to 4x (1=true)",
  lambda: float(all(
      [J(MG)["pairs"][p]["per_factor"][k]["mean_delta"] for k in ("0.5", "1.0", "2.0", "4.0")]
      == sorted(J(MG)["pairs"][p]["per_factor"][k]["mean_delta"] for k in ("0.5", "1.0", "2.0", "4.0"))
      for p in SOFT_PAIRS)), 1, 0)
C("S3 magnitude", "significantly nonzero by 2x-4x on every seed (1=true)",
  lambda: float(all(J(MG)["pairs"][p]["per_factor"][k]["ci"][0] > 0 for p in SOFT_PAIRS for k in ("2.0", "4.0"))), 1, 0)
C("S3 magnitude", "attenuation hard/soft at 4x, seed 42 = 1.3",
  lambda: J(MG)["pairs"]["v4_hard"]["per_factor"]["4.0"]["mean_delta"] / J(MG)["pairs"]["v4s_soft"]["per_factor"]["4.0"]["mean_delta"], 1.3, 1)
C("S3 magnitude", "attenuation hard/soft at 4x, seed 44 = 1.6",
  lambda: J(MG)["pairs"]["v4_hard"]["per_factor"]["4.0"]["mean_delta"] / J(MG)["pairs"]["v4s_s44"]["per_factor"]["4.0"]["mean_delta"], 1.6, 1)
C("S3 magnitude", "attenuation hard/soft at 4x, weakest seed ~10",
  lambda: J(MG)["pairs"]["v4_hard"]["per_factor"]["4.0"]["mean_delta"] / J(MG)["pairs"]["v4s_s43"]["per_factor"]["4.0"]["mean_delta"], 10, 0)

# ------------------------------------------------ Sec. 3 generic-prior controls
GB = "generic_baseline_full.json"
GEN = ["generic-s9btm_all_causal", "generic-s9ho_all_causal"]
C("S3 controls", "public DoT-PFN checkpoints synthetic bias, min -0.98",
  lambda: min(J(GB)["tiers"]["synthetic"][g][m]["bias"] for g in ("0.0", "1.0", "2.0") for m in GEN), -0.98)
C("S3 controls", "public DoT-PFN checkpoints synthetic bias, max -0.81",
  lambda: max(J(GB)["tiers"]["synthetic"][g][m]["bias"] for g in ("0.0", "1.0", "2.0") for m in GEN), -0.81)
C("S3 controls", "market model synthetic bias, min -0.23",
  lambda: min(J(GB)["tiers"]["synthetic"][g]["market-dotpfn"]["bias"] for g in ("0.0", "1.0", "2.0")), -0.23)
C("S3 controls", "market model synthetic bias, max +0.16",
  lambda: max(J(GB)["tiers"]["synthetic"][g]["market-dotpfn"]["bias"] for g in ("0.0", "1.0", "2.0")), 0.16)
C("S3 controls", "public checkpoint A on BTC bursts -2.20", lambda: J(GB)["tiers"]["crypto_BTCUSDT"][GEN[0]]["bias"], -2.20)
C("S3 controls", "public checkpoint B on BTC bursts -2.22", lambda: J(GB)["tiers"]["crypto_BTCUSDT"][GEN[1]]["bias"], -2.22)
C("S3 controls", "ablated twin on BTC bursts -2.26", lambda: J(CRYPTO["BTC"])["methods"]["ablated-dotpfn"]["bias"], -2.26)
C("S3 controls", "market model on BTC bursts -0.84", lambda: J(GB)["tiers"]["crypto_BTCUSDT"]["market-dotpfn"]["bias"], -0.84)
CT = "generic-ct-generic_random_s42"
C("S3 controls", "generic-prior retrain synthetic bias, min -0.76",
  lambda: min(J(GB)["tiers"]["synthetic"][g][CT]["bias"] for g in ("0.0", "1.0", "2.0")), -0.76)
C("S3 controls", "generic-prior retrain synthetic bias, max -0.72",
  lambda: max(J(GB)["tiers"]["synthetic"][g][CT]["bias"] for g in ("0.0", "1.0", "2.0")), -0.72)
C("S3 controls", "generic-prior retrain BTC bursts -2.26", lambda: J(GB)["tiers"]["crypto_BTCUSDT"][CT]["bias"], -2.26)
C("S3 controls", "generic-prior retrain held-out ABIDES -0.29", lambda: J(GB)["tiers"]["abides_heldout"][CT]["bias"], -0.29)
ci("S3 controls", "generic-prior retrain held-out ABIDES", lambda: J(GB)["tiers"]["abides_heldout"][CT]["bias_ci"], -0.64, 0.05)

# --------------------------------------------- Sec. 3 "Two instantiations"
R = "methods_vs_gamma_v10s.json"
C("S3 recurrent", "recurrent gamma=0 bias +0.16", lambda: gamma_row(R, 0.0, "market-dotpfn")["bias"], 0.16)
ci("S3 recurrent", "recurrent gamma=0 bias", lambda: gamma_row(R, 0.0, "market-dotpfn")["bias_ci"], 0.09, 0.23)
C("S3 recurrent", "recurrent gamma=2 bias +0.52", lambda: gamma_row(R, 2.0, "market-dotpfn")["bias"], 0.52)
ci("S3 recurrent", "recurrent gamma=2 bias", lambda: gamma_row(R, 2.0, "market-dotpfn")["bias_ci"], 0.38, 0.66)
C("S3 recurrent", "transformer gamma=2 bias +0.16", lambda: gamma_row("methods_vs_gamma_v4_full.json", 2.0, "market-dotpfn")["bias"], 0.16)
ci("S3 recurrent", "transformer gamma=2 bias", lambda: gamma_row("methods_vs_gamma_v4_full.json", 2.0, "market-dotpfn")["bias_ci"], 0.05, 0.28)

# ----------------------------------------- Sec. 4.2 metaorder benchmark tier
C("S4.2 benchmark", "12 duration x magnitude cells", lambda: len(J("benchmark_prototype.json")["classes"]), 12, 0)
C("S4.2 benchmark", "2,158 BTC episodes", lambda: J("benchmark_corr_ci.json")["n_episodes"], 2158, 0)
C("S4.2 benchmark", "transformer curve correlation +0.68", lambda: J("benchmark_corr_ci.json")["corr"], 0.68)
ci("S4.2 benchmark", "transformer curve correlation", lambda: J("benchmark_corr_ci.json")["ci"], 0.60, 0.75)
C("S4.2 benchmark", "measured per-unit-flow impact rises ~3.6x over durations", lambda: J("benchmark_corr_ci_v10s.json")["measured_shape_ratio"], 3.6, 1)
DD = "duration_response_diag_v10s.json"
C("S4.2 benchmark", "recurrent response accrues 3.1x over 2-31 bars",
  lambda: J(DD)["models"]["v10s"]["end_pinned"][-1]["response"] / J(DD)["models"]["v10s"]["end_pinned"][0]["response"], 3.1, 1)
C("S4.2 benchmark", "duration diagnostic spans 2..31 bars (1=true)",
  lambda: float(J(DD)["models"]["v10s"]["end_pinned"][0]["dur"] == 2 and J(DD)["models"]["v10s"]["end_pinned"][-1]["dur"] == 31), 1, 0)
C("S4.2 benchmark", "recurrent hard-trained correlation +0.94", lambda: J("benchmark_corr_ci_v10.json")["corr"], 0.94)
ci("S4.2 benchmark", "recurrent hard-trained correlation", lambda: J("benchmark_corr_ci_v10.json")["ci"], 0.90, 0.96)
C("S4.2 benchmark", "recurrent soft-mix correlation +0.96", lambda: J("benchmark_corr_ci_v10s.json")["corr"], 0.96)
ci("S4.2 benchmark", "recurrent soft-mix correlation", lambda: J("benchmark_corr_ci_v10s.json")["ci"], 0.93, 0.98)
SEEDS = ["benchmark_v10s_softmix.json", "benchmark_v10s_s43.json", "benchmark_v10s_s44.json"]
C("S4.2 benchmark", "three-seed correlation range, min +0.92", lambda: min(J(f)["curve_correlation"] for f in SEEDS), 0.92)
C("S4.2 benchmark", "three-seed correlation range, max +0.96", lambda: max(J(f)["curve_correlation"] for f in SEEDS), 0.96)
HO = "benchmark_v10s_heldout_jun.json"
C("S4.2 benchmark", "held-out replication +0.97", lambda: J("benchmark_corr_ci_v10s_heldout.json")["corr"], 0.97)
ci("S4.2 benchmark", "held-out replication", lambda: J("benchmark_corr_ci_v10s_heldout.json")["ci"], 0.93, 0.98)
C("S4.2 benchmark", "held-out window is 40 days", lambda: J(HO)["args"]["n_days"], 40, 0)
C("S4.2 benchmark", "held-out window starts May 11 (1=true)", lambda: float(J(HO)["args"]["dates_from"] == "2026-05-11"), 1, 0)
KV = "k_per_venue_v10s.json"
C("S4.2 scale k", "crypto k BTC 0.49", lambda: J(KV)["venues"]["BTCUSDT"]["k"], 0.49)
C("S4.2 scale k", "crypto k ETH 0.51", lambda: J(KV)["venues"]["ETHUSDT"]["k"], 0.51)
C("S4.2 scale k", "crypto k SOL 0.53", lambda: J(KV)["venues"]["SOLUSDT"]["k"], 0.53)
C("S4.2 scale k", "k transfer to 20 disjoint days: calibrated level 1.19x", lambda: J("affine_calibration.json")["level_cal_B"], 1.19)
C("S4.2 scale k", "k transfer to 20 disjoint days: corr +0.98", lambda: J("affine_calibration.json")["corr_B"], 0.98)
RS = [f"crypto_{s}_v10s{x}.json" for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT") for x in ("", "_s43", "_s44")]
C("S4.2 recurrent seeds", "causal bias across seeds and assets, min -1.2", lambda: min(J(f)["methods"]["market-dotpfn"]["bias"] for f in RS), -1.2, 1)
C("S4.2 recurrent seeds", "causal bias across seeds and assets, max +0.6", lambda: max(J(f)["methods"]["market-dotpfn"]["bias"] for f in RS), 0.6, 1)
C("S4.2 recurrent seeds", "per-seed do-head delta, min +1.2", lambda: min(J(f)["methods"]["market-dotpfn"]["delta_vs_ablated"] for f in RS), 1.2, 1)
C("S4.2 recurrent seeds", "per-seed do-head delta, max +3.0", lambda: max(J(f)["methods"]["market-dotpfn"]["delta_vs_ablated"] for f in RS), 3.0, 1)
EN = "crypto_v10s_seed_ensemble.json"
C("S4.2 recurrent seeds", "ensemble bias BTC +0.02", lambda: J(EN)["symbols"]["BTCUSDT"]["ens_bias"][0], 0.02)
C("S4.2 recurrent seeds", "ensemble bias ETH -0.22", lambda: J(EN)["symbols"]["ETHUSDT"]["ens_bias"][0], -0.22)
C("S4.2 recurrent seeds", "ensemble bias SOL +0.13", lambda: J(EN)["symbols"]["SOLUSDT"]["ens_bias"][0], 0.13)
C("S4.2 recurrent seeds", "ensemble ETH upper bound -0.00", lambda: J(EN)["symbols"]["ETHUSDT"]["ens_bias"][2], -0.00)
C("S4.2 recurrent seeds", "BTC/SOL ensemble intervals span zero (1=true)",
  lambda: float(all(J(EN)["symbols"][s]["ens_bias"][1] < 0 < J(EN)["symbols"][s]["ens_bias"][2] for s in ("BTCUSDT", "SOLUSDT"))), 1, 0)

# ---------------------------------------------------------------- Table 2
CAL = "calibration_n500agents.json"
TABLE2 = {"synthetic g=0": (0.77, 0.76, 1.68, 2.44), "synthetic g=1": (0.82, 0.80, 2.32, 3.47),
          "synthetic g=2": (0.79, 0.85, 3.17, 4.75), "ABIDES": (0.32, 0.39, 3.20, 3.73),
          "BTC bursts": (0.47, 0.35, 3.06, 3.46), "ETH bursts": (0.51, 0.32, 3.05, 3.44)}
for tier, (cc, ca, wc, wa) in TABLE2.items():
    C(f"Tab.2 {tier}", "causal coverage", lambda t=tier: J(CAL)["tiers"][t]["causal"]["coverage80"], cc)
    C(f"Tab.2 {tier}", "ablated coverage", lambda t=tier: J(CAL)["tiers"][t]["ablated"]["coverage80"], ca)
    C(f"Tab.2 {tier}", "causal median width", lambda t=tier: J(CAL)["tiers"][t]["causal"]["median_width80"], wc)
    C(f"Tab.2 {tier}", "ablated median width", lambda t=tier: J(CAL)["tiers"][t]["ablated"]["median_width80"], wa)
CV = "conformal_v10s.json"
C("Tab.2 ABIDES (recurrent)", "causal coverage 0.49", lambda: J(CV)["models"]["causal"]["raw"]["coverage80"], 0.49)
C("Tab.2 ABIDES (recurrent)", "ablated coverage 0.76", lambda: J(CV)["models"]["ablated"]["raw"]["coverage80"], 0.76)
C("Tab.2 ABIDES (recurrent)", "causal width 10.49", lambda: J(CV)["models"]["causal"]["raw"]["median_width80"], 10.49)
C("Tab.2 ABIDES (recurrent)", "ablated width 10.50", lambda: J(CV)["models"]["ablated"]["raw"]["median_width80"], 10.50)

# ------------------------------------------------------ Sec. 4.3 conformal
CA = "conformal_abides_n500.json"
C("S4.3 conformal", "raw coverage on evaluation half 0.31", lambda: J(CA)["models"]["causal"]["raw"]["coverage80"], 0.31)
C("S4.3 conformal", "conformalized coverage 0.82", lambda: J(CA)["models"]["causal"]["conformal"]["coverage80"], 0.82)
C("S4.3 conformal", "interval widening ~4x",
  lambda: J(CA)["models"]["causal"]["conformal"]["median_width80"] / J(CA)["models"]["causal"]["raw"]["median_width80"], 4, 0)
C("S4.3 conformal", "per-side correction +5.0", lambda: J(CA)["models"]["causal"]["correction"], 5.0, 1)
C("S4.3 conformal", "transformer conformalized width 13.1", lambda: J(CA)["models"]["causal"]["conformal"]["median_width80"], 13.1, 1)
C("S4.3 conformal", "recurrent conformalized coverage 0.81", lambda: J(CV)["models"]["causal"]["conformal"]["coverage80"], 0.81)
C("S4.3 conformal", "recurrent widening 1.6x",
  lambda: J(CV)["models"]["causal"]["conformal"]["median_width80"] / J(CV)["models"]["causal"]["raw"]["median_width80"], 1.6, 1)
C("S4.3 conformal", "recurrent conformalized width 16.9", lambda: J(CV)["models"]["causal"]["conformal"]["median_width80"], 16.9, 1)
C("S4.3 conformal", "recurrent base ~3x wider than transformer",
  lambda: J(CV)["models"]["causal"]["raw"]["median_width80"] / J(CA)["models"]["causal"]["raw"]["median_width80"], 3, 0)
AS = "abides_v10s_calibrated_seeds.json"
C("S4.3 scale k", "ABIDES k across seeds, min 0.049", lambda: min(v["k"] for v in J(AS).values()), 0.049, 3)
C("S4.3 scale k", "ABIDES k across seeds, max 0.096", lambda: max(v["k"] for v in J(AS).values()), 0.096, 3)
C("S4.3 scale k", "calibrated bias seed-invariant at -0.12 (three-seed mean)",
  lambda: sum(J(AS)[s]["cal_bias"][0] for s in ("s42", "s43", "s44")) / 3, -0.12)
C("S4.3 scale k", "calibrated bias seed spread below print resolution (max-min, 2 dp)",
  lambda: max(J(AS)[s]["cal_bias"][0] for s in ("s42", "s43", "s44")) - min(J(AS)[s]["cal_bias"][0] for s in ("s42", "s43", "s44")), 0.0)
for s in ("s42", "s43", "s44"):
    C("S4.3 scale k", f"calibrated bias CI spans zero, seed {s} (1=true)",
      lambda s=s: float(J(AS)[s]["cal_bias"][1] < 0 < J(AS)[s]["cal_bias"][2]), 1, 0)
C("S4.3 scale k", "k fitted on n=500 calibration episodes", lambda: J("abides_v10s_calibrated_full.json")["n_cal"], 500, 0)
C("S4.3 scale k", "k applied to n=1000 held-out counterfactuals", lambda: J("abides_v10s_calibrated_full.json")["n_holdout"], 1000, 0)
KS = "conformal_v10s_kscaled.json"
C("S4.3 k+conformal", "k-recentered conformal coverage 0.783", lambda: J(KS)["models"]["causal"]["conformal"]["coverage80"], 0.783, 3)
C("S4.3 k+conformal", "k-recentered conformal width 13.4", lambda: J(KS)["models"]["causal"]["conformal"]["median_width80"], 13.4, 1)
C("S4.3 k+conformal", "unscaled pipeline coverage 0.812", lambda: J(CV)["models"]["causal"]["conformal"]["coverage80"], 0.812, 3)
C("S4.3 k+conformal", "k-recentered run n=300", lambda: J(KS)["args"]["n_eval"], 300, 0)
C("S4.3 k+conformal", "unscaled run n=500", lambda: J(CV)["args"]["n_eval"], 500, 0)

# --------------------------------------------- Sec. 4.1 off-prior stress table
ST = "offprior_stress.json"
STRESS = {"control": (-0.05, -0.13, 0.03, 0.60, 0.52, 0.68),
          "feedback_weak": (-0.02, -0.10, 0.06, 0.64, 0.56, 0.72),
          "feedback_strong": (0.03, -0.06, 0.12, 0.69, 0.60, 0.79),
          "jumps": (0.08, -0.02, 0.18, 0.73, 0.64, 0.83),
          "saturation": (0.11, 0.04, 0.18, 0.57, 0.50, 0.64)}
for cfg, (b, blo, bhi, d, dlo, dhi) in STRESS.items():
    C(f"Stress {cfg}", "M-DoT-PFN bias", lambda c=cfg: J(ST)["configs"][c]["market-dotpfn"]["bias"], b)
    ci(f"Stress {cfg}", "M-DoT-PFN bias", lambda c=cfg: J(ST)["configs"][c]["market-dotpfn"]["bias_ci"], blo, bhi)
    C(f"Stress {cfg}", "do-head effect", lambda c=cfg: J(ST)["configs"][c]["delta_vs_ablated"]["mean"], d)
    ci(f"Stress {cfg}", "do-head effect", lambda c=cfg: J(ST)["configs"][c]["delta_vs_ablated"]["ci"], dlo, dhi)
C("Stress", "n=500 episodes per generator", lambda: J(ST)["args"]["n_episodes"], 500, 0)

# Claims the paper makes for which no summary artifact is stored. Listed so
# the gap is visible; several are post-hoc summaries of the per-episode dumps
# in abides_heldout_v4_full.json and could be recomputed from them.
UNCHECKED = [
    ("S4.2", "calm/stressed volatility split (BTC +1.49 / +1.36; six CIs overlap) — no stored summary"),
    ("S4.1", "ABIDES post-hoc correlations r=0.09 / 0.11, constant-shift RMSE CI [-0.56, 0.14], response SD 0.60, "
             "seed-replay per-episode SD 3.6 — derivable from the per-episode dump, no stored summary"),
    ("S4.1", "raw ABM intervention values ~19 sigma-hat out of distribution — pre-standardization diagnostic, no artifact"),
    ("S4.2", "recurrent zero-shot ETH/SOL correlations are stored (benchmark_v10s_{ETH,SOL}USDT.json) but not quoted numerically"),
]


# Claims kept at the paper's literal wording although the stored value
# disagrees at print resolution. They FAIL above by design; this block says why.
KNOWN_DISCREPANCIES: list[tuple[str, str, str]] = []  # none after the 2026-08-12 paper fixes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=None, help="directory holding the result JSONs (default: this file's directory)")
    args = ap.parse_args()
    global RESULTS
    if args.results_dir:
        RESULTS = Path(args.results_dir).resolve()
        J.cache_clear()

    failures = 0
    print(f"{'status':6s}  {'paper location':26s}  {'claim':62s}  {'stored':>10s}  {'printed':>8s}")
    for where, what, fn, printed, dec in CLAIMS:
        try:
            actual = float(fn())
            ok = abs(actual - printed) <= 0.5 * 10 ** (-dec) + 1e-9
        except Exception as err:  # missing file/key is a failure, not a crash
            actual, ok = float("nan"), False
            what = f"{what} [{type(err).__name__}: {err}]"
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL':6s}  {where:26s}  {what[:62]:62s}  {actual:10.4f}  {printed:8g}")
    print(f"\n{len(CLAIMS) - failures}/{len(CLAIMS)} claims pass; {failures} fail "
          f"({len(KNOWN_DISCREPANCIES)} known, explained below); {len(UNCHECKED)} unchecked.")
    for where, what, why in KNOWN_DISCREPANCIES:
        print(f"  KNOWN      {where:12s}  {what}: {why}")
    for where, note in UNCHECKED:
        print(f"  UNCHECKED  {where:12s}  {note}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
