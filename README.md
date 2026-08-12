# Causal Market Simulator (`dotime-market`)

ICAIF '26 submission repo: a continuous-time causal PFN for market impact,
built as an extension of the released `dotime` package (the continuous-time causal PFN construction cited in the paper)
package. Market impact is treated as what it is — a counterfactual question:
*what would the price have been had I not traded?*

**Private until the ICAIF '26 camera-ready (double-blind).**

## Layout

- `src/dotime_market/prior/` — market prior mechanisms (core/reaction confounding,
  informational coupling γ, jump-diffusion, factor DAGs, Hawkes timing)
- `src/dotime_market/baselines/` — oracle adjustment estimators + naive impact models
- `src/dotime_market/evaluation/` — diffusivity, square-root-law, calibration diagnostics
- `src/dotime_market/data/` — ABIDES / Binance → `dotime.benchmarks.Episode` adapters
- `src/dotime_market/models/` — continuous-time causal PFN + trainer (adapted from the
  cited construction's source; the released `dotime` ships only the discrete model, no trainer)
- `scripts/` — numbered phase entry points; `configs/` — YAML configs

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[train,dev]"          # pulls dotime[models,baselines] from PyPI
# dev-against-local-dotime alternative: pip install -e "~/repos/dotime[models,baselines]" first
```

## Phase 0 — the PoC precondition

```bash
python scripts/00_stock_gap_check.py   # |E[Y|A] - E[Y|do(A)]| on stock dotime
pytest tests/test_confounding_gap.py   # the same check as a CI gate
```

## Smoke-train the vendored pipeline (stock prior)

```bash
python scripts/02_train_market_pfn.py --config configs/train_dotpfn.yaml \
    --total-steps 50 --eval-every 25 --save-dir checkpoints/smoke
```

See `ARTIFACT.md` for the reviewer-facing overview and reproduction order.

## Reproducing the results

Start with `notebooks/reproduction_walkthrough.ipynb` — it reproduces each
evidence pillar at smoke scale (CPU, minutes) and maps every figure/table to
its generating script and runtime. Full-scale protocol invariants (matched
causal/ablated twin, exploratory-vs-held-out ABIDES seed ranges, B=10^4
episode bootstrap, per-layer reduction tests) are stated in the notebook's
final section; numeric provenance for every paper claim is mapped
script-by-script in `ARTIFACT.md`.
