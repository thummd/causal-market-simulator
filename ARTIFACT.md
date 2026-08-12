# Anonymized artifact — Market Impact Estimation via Causal Foundation Models (ICAIF '26 submission)

This repository contains the full experimental pipeline for the paper:
the confounded market prior (`src/dotime_market/prior/`, including the
jump-diffusion and Hawkes extensions used in companion robustness
experiments), baselines including the square-root law and
matched-horizon return regression (`src/dotime_market/baselines/`),
episode builders for the synthetic, ABIDES seed-replay, and Binance
flow-burst tiers (`src/dotime_market/data/`), and every evaluation
script behind the paper's figures and tables. `tests/` contains the
unit tests referenced in the paper (pre-onset path identity,
confounder calibration, reduction tests, oracle sanity, intervention
kind-mix wiring).

Script map: `scripts/01*`–`09*` are the transformer pipeline (`06`–`08`:
off-prior stress, prior-attribution control, functional-form studies;
`04d`: the sign-shuffle momentum control). `scripts/11`–`13` build the
metaorder benchmark of §5.2 (synthetic-metaorder episodes, the
duration-response diagnostic, and the recursive-readout utility it
imports). The recurrent state-space variant of §3 is
`src/dotime_market/models/recurrent_model.py`, trained by
`scripts/15_train_recurrent.py`; `scripts/16_schedule_robustness.py` is
the schedule-robustness check behind the §3 deployment-license remark,
and `scripts/05b_conformal.py --k-scale` composes per-domain scale
calibration with split-conformal intervals (§5.3).

Start with `notebooks/reproduction_walkthrough.ipynb`: it reproduces
each evidence pillar at smoke scale on CPU in minutes and maps every
figure and table to its generating script, runtime, and the protocol
invariants (matched causal/ablated twin, exploratory-vs-held-out seed
ranges, B=10^4 episode bootstrap).

`results/` holds the material the paper points to but does not inline:

- `offprior_stress_table.md` — the per-generator off-prior stress table
  with all CIs (n=500 each), summarized in §4.1; regenerate with
  `render_offprior_table.py` from `offprior_stress.json`.
- `fig3_forest.{pdf,png}` — the forest-plot rendering of Table 1
  (paired do-head effects with bootstrap CIs).
- `magnitude_scaling_v4s.json` — the soft/hard magnitude-sweep
  (`scripts/19_magnitude_scaling.py`): the soft-trained response tracks
  the hard-trained curve at a roughly 2--3x compressed input scale.
- Result JSONs backing the recurrent-variant claims: benchmark curve
  correlations and episode-bootstrap CIs (`benchmark_corr_ci_v10*.json`,
  `benchmark_v10*.json`, incl. the s43/s44 seed pairs and the zero-shot
  ETH/SOL runs), crypto burst-tier per-seed and seed-ensemble numbers
  (`crypto_*_v10s*.json`, `crypto_v10s_seed_ensemble.json`), per-venue
  and per-domain scale constants (`k_per_venue_v10s.json`,
  `affine_calibration.json`, `abides_v10s_calibrated_*.json`), interval
  calibration (`conformal_v10s*.json`), the duration-response
  diagnostic (`duration_response_diag_v10s.json`), the schedule
  check (`schedule_robustness.json`), and the synthetic-tier grids for
  the recurrent pair (`methods_vs_gamma_v10s.json`) and the transformer
  (`methods_vs_gamma_v4_full.json`; source of the paper's
  gamma=2 bias +0.16 [0.05, 0.28]).

Checkpoints for the recurrent state-space pairs (1.94M parameters,
about 7.5 MB each) are included under `checkpoints/`: the hard-trained pair
(`market_v10_rnn{,_ablated}`) and the soft-mix pairs for all three
training seeds (`market_v10s_softmix{,_ablated}`,
`market_v10s_s43{,_ablated}`, `market_v10s_s44{,_ablated}`). The
transformer checkpoints (about 46 MB each) and the ABIDES/Binance episode
caches remain too large for this mirror; they will be released with a
DOI upon acceptance. All data sources are public (the Binance archive;
ABIDES from the public jpmorganchase/abides-jpmc-public repository).
