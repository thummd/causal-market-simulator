#!/usr/bin/env python
"""Gate A: train the recurrent (GatedDeltaProduct) causal PFN backbone.

A lean training loop for :class:`dotime_market.models.recurrent_model.
RecurrentDoTPFN` that reuses the market dataloader and the do-ablation
protocol but leaves the vendored transformer trainer untouched. The matched
comparison holds the training distribution fixed at v8's EFFECTIVE recipe
(hard-only interventions — see the kind-probs wiring fix, commit 150ea5b;
pass --intervention-kind-probs explicitly to change this consciously).

Usage:
    python scripts/15_train_recurrent.py --config configs/train_market.yaml \
        --perm-prob 0.3 --window-frac 0.02 0.30 --seed 42 \
        --save-dir checkpoints/market_v10_rnn
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from dotime_market.models.dataloader import ContinuousTemporalInterventionDataLoader
from dotime_market.models.recurrent_model import RecurrentDoTPFN
from dotime_market.models.train import _ablate_batch, _cosine_with_warmup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_market.yaml")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="checkpoints/market_v10_rnn")
    ap.add_argument("--ablate-intervention", action="store_true")
    ap.add_argument("--coupling-gamma-power", type=float, default=3.0)
    ap.add_argument("--perm-prob", type=float, default=0.0)
    ap.add_argument("--lambda-log-range", type=float, nargs=2, default=None)
    ap.add_argument("--window-frac", type=float, nargs=2, default=None)
    ap.add_argument("--intervention-kind-probs", type=float, nargs=3,
                    default=[1.0, 0.0, 0.0])
    ap.add_argument("--embed-size", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m, p, t = cfg["model"], cfg["prior"], cfg["training"]
    total_steps = args.total_steps or t["total_steps"]
    eval_every = args.eval_every or t["eval_every"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    loader_kwargs = dict(
        batch_size=t["batch_size"],
        prior_mode="market",
        tscm_structure=p["tscm_structure"],
        n_min_prior=p["n_min_prior"], n_max_prior=p["n_max_prior"],
        edge_prob=p["edge_prob"], hidden_prob=p["hidden_prob"],
        regime_prob=p["regime_prob"],
        regime_count_range=tuple(p["regime_count_range"]),
        mechanism_kind=p["mechanism_kind"], p_neural=p["p_neural"],
        neural_hidden_dim=p["neural_hidden_dim"],
        neural_out_scale_range=tuple(p["neural_out_scale_range"]),
        num_substeps=p["num_substeps"], p_no_context=p["p_no_context"],
        schedule=p["schedule"], dt=p["dt"], jitter=p["jitter"],
        exp_rate=p["exp_rate"], pair_mode=p["pair_mode"],
        t_range=tuple(p["t_range"]), n_max=m["n_max"], normalize=True,
        target_key=t["target_key"], n_queries=t["n_queries"],
        query_mode=t["query_mode"], theta_range=tuple(p["theta_range"]),
        sigma_range=tuple(p["sigma_range"]), weight_scale=p["weight_scale"],
        intervention_value_scale=p["intervention_value_scale"],
        intervention_window_frac=(
            tuple(args.window_frac) if args.window_frac
            else tuple(p["intervention_window_frac"])),
        intervention_kind_probs=tuple(args.intervention_kind_probs),
        market_coupling_gamma=tuple(p["market_coupling_gamma"]),
        market_core_share=tuple(p["market_core_share"]),
        market_impact_lambda=tuple(p["market_impact_lambda"]),
        market_coupling_gamma_power=args.coupling_gamma_power,
        market_perm_prob=args.perm_prob,
        market_impact_lambda_log_range=(tuple(args.lambda_log_range) if args.lambda_log_range else None),
    )
    train_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=total_steps, seed=args.seed, device=device, prefetch=2,
        **loader_kwargs)
    eval_loader = ContinuousTemporalInterventionDataLoader(
        num_steps=t["eval_num_steps"], seed=args.seed + 10_000, device=device,
        prefetch=0, **loader_kwargs)

    model = RecurrentDoTPFN(
        n_max=m["n_max"], embed_size=args.embed_size, n_layers=args.n_layers,
        tau_levels=m["tau_levels"]).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"   Arch: recurrent GatedDeltaProduct  |  Parameters: {n_params:,}"
          f"  |  ablate: {args.ablate_intervention}")
    print(f"   Kind probs (EFFECTIVE): {tuple(args.intervention_kind_probs)}")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"],
                            betas=(0.9, 0.98), eps=1e-6)
    sched = _cosine_with_warmup(opt, t["warmup_steps"], total_steps)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = {"arch": "recurrent", "n_max": m["n_max"],
                "embed_size": args.embed_size, "n_layers": args.n_layers,
                "tau_levels": m["tau_levels"],
                "intervention_kind_probs": list(args.intervention_kind_probs),
                "perm_prob": args.perm_prob,
                "window_frac": args.window_frac, "seed": args.seed}

    best = float("inf")
    t0 = time.time()
    model.train()
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
        if args.ablate_intervention:
            batch = _ablate_batch(batch)
        if float(batch["Y_true_norm"].abs().max()) >= 49.5:
            continue
        loss = model.head.loss(model(batch), batch["Y_true_norm"])
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        sched.step()

        if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for eb in eval_loader:
                    if args.ablate_intervention:
                        eb = _ablate_batch(eb)
                    tot += float(model.head.loss(model(eb), eb["Y_true_norm"]))
                    n += 1
            ev = tot / max(n, 1)
            print(f"   Step {step + 1}/{total_steps} ({time.time() - t0:.0f}s)"
                  f"  |  train {float(loss):.4f}  eval {ev:.4f}"
                  f"  lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if ev < best:
                best = ev
                torch.save({"config": ckpt_cfg,
                            "model_state_dict": model.state_dict()},
                           save_dir / "recurrent_dotpfn_best.pt")
            model.train()

    print(f"Training complete in {time.time() - t0:.0f}s.  Best eval loss: {best:.4f}")


if __name__ == "__main__":
    main()
