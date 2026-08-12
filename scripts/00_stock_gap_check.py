#!/usr/bin/env python
"""Phase 0 — does stock dotime's `unobserved_confounder` open a measurable gap
between E[Y|A] and E[Y|do(A)]?

Per SCM:
  - beta_naive: OLS slope of Y_t on A_t on a long observational trajectory
    (the naive predictive/conditional estimate of the effect).
  - beta_do:    empirical interventional slope from hard do(A=+2) vs do(A=-2)
    runs (true effect; analytically 0 here — the structure U->A, U->Y has no
    A->Y edge, so any nonzero naive slope is pure confounding bias).

Gap metric: |beta_naive - beta_do| averaged over SCMs, swept over weight_scale
(the global parental-weight prior std — the only stock "confounder knob").

Reference output (2026-07-02, seeds as below):
  weight_scale=0.25  mean|beta_naive|=0.0597  mean|beta_do|=0.0046  mean gap=0.0592
  weight_scale=0.50  mean|beta_naive|=0.0972  mean|beta_do|=0.0051  mean gap=0.0966
  weight_scale=1.00  mean|beta_naive|=0.1958  mean|beta_do|=0.0065  mean gap=0.1955
"""

from __future__ import annotations

import torch

from dotime.continuous import (
    ContinuousIntervention,
    ContinuousTSCMSampler,
    InterventionKind,
    regular_schedule,
)
from dotime.tscm_sampler import TSCMStructure

N_SCMS = 60
T_OBS = 500          # observational trajectory length for the naive fit
T_INT = 300          # interventional trajectory length
BURN = 50
NUM_SUBSTEPS = 2
N_REPS = 6           # noise realisations per intervention arm
A_VAL = 2.0
WIN = (150.0, 300.0)  # hard intervention window (absolute time)


def naive_slope(x: torch.Tensor, a_idx: int, y_idx: int) -> float:
    a = x[BURN:, a_idx]
    y = x[BURN:, y_idx]
    a = a - a.mean()
    y = y - y.mean()
    return float((a * y).sum() / (a * a).sum().clamp_min(1e-8))


def do_slope(scm, a_idx: int, y_idx: int, gen: torch.Generator) -> float:
    times, dts = regular_schedule(T=T_INT, dt=1.0)
    post = (times >= WIN[0] + 20.0)  # let the effect settle past onset
    means = {}
    for sign in (+1.0, -1.0):
        vals = []
        for _ in range(N_REPS):
            noise = scm._draw_noise((T_INT - 1) * NUM_SUBSTEPS, generator=gen)
            iv = ContinuousIntervention(
                target=a_idx, t_start=WIN[0], t_end=WIN[1],
                kind=InterventionKind.HARD, value=sign * A_VAL,
            )
            _, traj = scm.simulate(times, dts, intervention=iv, noise=noise,
                                   num_substeps=NUM_SUBSTEPS)
            vals.append(float(traj[post, y_idx].mean()))
        means[sign] = sum(vals) / len(vals)
    return (means[1.0] - means[-1.0]) / (2 * A_VAL)


def run_sweep(weight_scales=(0.25, 0.5, 1.0), n_scms: int = N_SCMS) -> dict[float, dict]:
    results: dict[float, dict] = {}
    for ws in weight_scales:
        gen = torch.Generator().manual_seed(123)
        sampler = ContinuousTSCMSampler(
            structure=TSCMStructure.UNOBSERVED_CONFOUNDER, weight_scale=ws
        )
        a_idx = sampler.get_intervention_target()
        y_idx = sampler.get_outcome_var()
        naive, do, gaps = [], [], []
        for _ in range(n_scms):
            scm = sampler.sample(generator=gen)
            times, dts = regular_schedule(T=T_OBS, dt=1.0)
            _, x = scm.simulate(times, dts, generator=gen, num_substeps=NUM_SUBSTEPS)
            bn = naive_slope(x, a_idx, y_idx)
            bd = do_slope(scm, a_idx, y_idx, gen)
            naive.append(abs(bn))
            do.append(abs(bd))
            gaps.append(abs(bn - bd))
        t = torch.tensor
        results[ws] = {
            "mean_abs_naive": float(t(naive).mean()),
            "mean_abs_do": float(t(do).mean()),
            "mean_gap": float(t(gaps).mean()),
            "median_gap": float(t(gaps).median()),
            "frac_gap_gt_005": float((t(gaps) > 0.05).float().mean()),
        }
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    for ws, r in run_sweep().items():
        print(
            f"weight_scale={ws:4.2f}  "
            f"mean|beta_naive|={r['mean_abs_naive']:.4f}  "
            f"mean|beta_do|={r['mean_abs_do']:.4f}  "
            f"mean gap={r['mean_gap']:.4f}  "
            f"median gap={r['median_gap']:.4f}  "
            f"frac(gap>0.05)={r['frac_gap_gt_005']:.2f}"
        )
