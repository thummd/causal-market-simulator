#!/usr/bin/env python
"""Schedule-robustness: Gate-A hardening for the irregular-observation claim.

Both backbones trained on regular dt=1 schedules. This evaluates each
checkpoint zero-shot on on-prior batches drawn under regular / jittered /
exponential observation schedules and reports the eval-loss degradation per
architecture. The continuous-time claim predicts the Δt-gated recurrent
model degrades less: its gates are exponentiated by the actual gap, while
the transformer must extrapolate its (regular-trained) time embeddings.

Usage:
    python scripts/16_schedule_robustness.py --n-batches 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.models.dataloader import ContinuousTemporalInterventionDataLoader

MODELS = {
    "transformer_v8": "checkpoints/market_v8_geom/continuous_do_over_time_pfn_best.pt",
    "recurrent_v10s": "checkpoints/market_v10s_softmix/recurrent_dotpfn_best.pt",
}
SCHEDULES = ["regular", "jittered", "exponential"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_market.yaml")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="output/schedule_robustness.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m, p, t = cfg["model"], cfg["prior"], cfg["training"]
    nets = {name: get("market-dotpfn", checkpoint=ck).model
            for name, ck in MODELS.items()}

    results = {"args": vars(args), "eval_loss": {}}
    print(f"{'schedule':>12}  " + "  ".join(f"{n:>16}" for n in nets))
    for schedule in SCHEDULES:
        # Fresh loader per schedule with a fixed seed: every model sees the
        # SAME episodes within a schedule, so rows are paired comparisons.
        loader = ContinuousTemporalInterventionDataLoader(
            num_steps=args.n_batches, seed=args.seed, prefetch=0,
            batch_size=t["batch_size"], prior_mode="market",
            tscm_structure=p["tscm_structure"], n_min_prior=p["n_min_prior"],
            n_max_prior=p["n_max_prior"], edge_prob=p["edge_prob"],
            hidden_prob=p["hidden_prob"], regime_prob=p["regime_prob"],
            regime_count_range=tuple(p["regime_count_range"]),
            mechanism_kind=p["mechanism_kind"], p_neural=p["p_neural"],
            neural_hidden_dim=p["neural_hidden_dim"],
            neural_out_scale_range=tuple(p["neural_out_scale_range"]),
            num_substeps=p["num_substeps"], p_no_context=p["p_no_context"],
            schedule=schedule, dt=p["dt"], jitter=p["jitter"],
            exp_rate=p["exp_rate"], pair_mode=p["pair_mode"],
            t_range=tuple(p["t_range"]), n_max=m["n_max"], normalize=True,
            target_key=t["target_key"], n_queries=t["n_queries"],
            query_mode=t["query_mode"], theta_range=tuple(p["theta_range"]),
            sigma_range=tuple(p["sigma_range"]), weight_scale=p["weight_scale"],
            intervention_value_scale=p["intervention_value_scale"],
            intervention_window_frac=(0.02, 0.30),
            intervention_kind_probs=(0.5, 0.5, 0.0),
            market_coupling_gamma=tuple(p["market_coupling_gamma"]),
            market_core_share=tuple(p["market_core_share"]),
            market_impact_lambda=tuple(p["market_impact_lambda"]),
            market_coupling_gamma_power=3.0, market_perm_prob=0.3)
        batches = list(loader)
        row = {}
        for name, net in nets.items():
            net.eval()
            tot = 0.0
            with torch.no_grad():
                for b in batches:
                    tot += float(net.head.loss(net(b), b["Y_true_norm"]))
            row[name] = tot / max(len(batches), 1)
        results["eval_loss"][schedule] = row
        print(f"{schedule:>12}  " + "  ".join(f"{row[n]:>16.4f}" for n in nets))

    reg = results["eval_loss"]["regular"]
    print("\ndegradation vs regular (relative):")
    for schedule in SCHEDULES[1:]:
        row = results["eval_loss"][schedule]
        print(f"{schedule:>12}  " + "  ".join(
            f"{(row[n] - reg[n]) / reg[n]:>+16.1%}" for n in nets))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
