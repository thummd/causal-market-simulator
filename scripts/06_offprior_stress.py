#!/usr/bin/env python
"""Off-prior misspecification stress test .

The synthetic evaluation of scripts/01b is on-prior by construction: the PFN
is evaluated on the same OU family it was trained on. This script generates
episodes from three generators OUTSIDE the training prior's model class ---
exact counterfactuals still exist because we control the simulator (shared
Wiener noise and jump draws between the factual and interventional runs):

- control:    linear OU, inside the prior family (validates this standalone
              simulator against the 01b gamma=1 grid numbers)
- feedback:   dA += kappa * Y dt --- price->flow feedback, a cyclic graph the
              prior's DAG cannot express (kappa in {0.1, 0.2})
- jumps:      compound-Poisson jumps in the hidden core U (heavy tails)
- saturation: impact enters the price as 2*tanh(A/2) --- nonlinear response

Everything else matches the 01b evaluation conventions: gamma=1, s=0.7,
theta ~ U(0.5,1), sigma ~ U(0.2,0.6), lambda ~ U(0.25,1), T=120, dt=1 with
2 Euler substeps, hard clamp interventions, one post-onset price query.

Usage:
    python scripts/06_offprior_stress.py --n-episodes 500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime import InterventionSpec, InterventionType
from dotime.baselines import get
from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata
from dotime.evaluation import bootstrap_ci
from dotime_market.prior.confounding import CoreReactionConfounder

GAMMA = 1.0
CORE_SHARE = 0.7
T, SUB, BURN = 120, 2, 60

CONFIGS = {
    "control": {},
    "feedback_weak": {"kappa": 0.1},
    "feedback_strong": {"kappa": 0.2},
    "jumps": {"jump_rate": 0.1, "jump_scale_mult": 3.0},
    "saturation": {"saturate": True},
}


def simulate_episode(rng: np.random.RandomState, kappa: float = 0.0,
                     jump_rate: float = 0.0, jump_scale_mult: float = 0.0,
                     saturate: bool = False, scm_id: int = 0) -> Episode:
    th_u, th_a, th_y = rng.uniform(0.5, 1.0, 3)
    sg_u, sg_a, sg_y = rng.uniform(0.2, 0.6, 3)
    lam = rng.uniform(0.25, 1.0)
    w = CoreReactionConfounder(core_share=CORE_SHARE, theta_core=th_u,
                               sigma_core=sg_u, theta_flow=th_a,
                               sigma_flow=sg_a).core_weight

    h = 1.0 / SUB
    n_steps = (T + BURN) * SUB
    onset = int(rng.randint(40, 80))
    end = min(onset + int(rng.randint(20, 36)), T - 1)
    c = float(np.clip(rng.randn() * 2.0, -4.0, 4.0))

    dW = rng.randn(n_steps, 3) * math.sqrt(h)
    jump = (rng.rand(n_steps) < jump_rate * h) * rng.randn(n_steps) * (jump_scale_mult * sg_u)

    def run(intervene: bool) -> np.ndarray:
        u = a = y = 0.0
        out = np.zeros((T, 3), dtype=np.float32)
        for i in range(n_steps):
            t_obs = i / SUB - BURN  # observation-clock time of the *next* state
            in_window = intervene and (onset <= t_obs + h < end + 1e-9)
            a_eff = c if in_window else a
            g = 2.0 * math.tanh(a_eff / 2.0) if saturate else a_eff
            y_new = y + (-th_y * y + lam * g + GAMMA * u) * h + sg_y * dW[i, 2]
            a_new = a_eff + (-th_a * a_eff + w * u + kappa * y) * h + sg_a * dW[i, 1]
            u_new = u + (-th_u * u) * h + sg_u * dW[i, 0] + jump[i]
            u, y = u_new, y_new
            a = c if in_window else a_new
            k = i + 1  # state after this substep
            if k % SUB == 0:
                t_idx = k // SUB - BURN - 1
                if 0 <= t_idx < T:
                    out[t_idx] = (a, 0.0, y)  # canonical (A, U hidden->0, Y)
        return out

    x_obs = torch.from_numpy(run(False))
    x_int = torch.from_numpy(run(True))
    q = int(rng.randint(onset, T - 1))
    intervention = InterventionSpec(
        targets=[0], times=list(range(onset, max(end, onset + 1))),
        intervention_type=InterventionType.HARD, values=c,
    )
    return Episode(
        x_obs=x_obs, x_int=x_int, intervention=intervention,
        y_true=x_int[[q], 2].to(torch.float32),
        query_target=torch.tensor([2], dtype=torch.long),
        query_time=torch.tensor([q / (T - 1)], dtype=torch.float32),
        structure="market_core_reaction",
        scm_id=scm_id,
        metadata={"query_idx": [q]},
    )


def score(model, suite, n_boot: int) -> dict:
    signed = []
    for ep in suite:
        p = model.predict(ep).reshape(-1)
        y = ep.y_true.reshape(-1)
        sgn = math.copysign(1.0, float(ep.intervention.values))
        signed.extend(sgn * float(a - b) for a, b in zip(p, y))
    m, _, lo, hi = bootstrap_ci(signed, n=n_boot)
    sq = np.square(signed)
    rng = np.random.RandomState(0)
    boots = np.sqrt(sq[rng.randint(0, len(sq), size=(n_boot, len(sq)))].mean(axis=1))
    return {
        "bias": m, "bias_ci": [lo, hi],
        "rmse": float(np.sqrt(sq.mean())),
        "rmse_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "signed_errors": [float(v) for v in signed],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--checkpoint",
                    default="checkpoints/market_v4_soft/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--ablated-checkpoint",
                    default="checkpoints/market_v4_ablated_s42/continuous_do_over_time_pfn_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="output/offprior_stress.json")
    args = ap.parse_args()

    models = {
        "Mean": get("Mean"),
        "AlmgrenChriss": get("AlmgrenChriss"),
        "OWPropagator": get("OWPropagator"),
        "SquareRootLaw": get("SquareRootLaw"),
        "market-dotpfn": get("market-dotpfn", checkpoint=args.checkpoint, device=args.device),
        "ablated-dotpfn": get("ablated-dotpfn", checkpoint=args.ablated_checkpoint,
                              device=args.device),
    }
    results = {"args": vars(args), "configs": {}}
    for cname, kw in CONFIGS.items():
        rng = np.random.RandomState(args.seed)
        eps = [simulate_episode(rng, scm_id=i, **kw) for i in range(args.n_episodes)]
        meta = SuiteMetadata(
            name=f"offprior-{cname}", version="0.0.1", zenodo_record_id="",
            doi="", description="Off-prior misspecification stress suite.",
            n_episodes=len(eps), structures=("market_core_reaction",),
        )
        suite = BenchmarkSuite(meta, eps)
        row = {}
        for mname, model in models.items():
            row[mname] = score(model, suite, args.bootstrap)
        # paired do-head delta on identical episodes
        d = [a - b for a, b in zip(row["market-dotpfn"]["signed_errors"],
                                   row["ablated-dotpfn"]["signed_errors"])]
        dm, _, dlo, dhi = bootstrap_ci(d, n=args.bootstrap)
        row["delta_vs_ablated"] = {"mean": dm, "ci": [dlo, dhi]}
        results["configs"][cname] = row
        print(f"{cname:16s} " + "  ".join(
            f"{m}: b{row[m]['bias']:+.3f}/r{row[m]['rmse']:.2f}"
            for m in ("market-dotpfn", "ablated-dotpfn", "SquareRootLaw", "OWPropagator")),
            flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # signed_errors are bulky; keep only aggregates in the artifact
    for row in results["configs"].values():
        for m in list(row):
            if isinstance(row[m], dict) and "signed_errors" in row[m]:
                del row[m]["signed_errors"]
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
