# Artifact — Market Impact Estimation via Causal Foundation Models (ICAIF '26)

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

- Transformer-tier result JSONs backing Table 1, §4.1–4.3 and Table 2:
  the held-out ABIDES seed-replay evaluation (`abides_heldout_v4_full.json`,
  seeds 1000–1999, per-episode dump included; hard-clamp encoding control
  `abides_hard_v4pair_n300.json`), the three burst-tier runs
  (`crypto_{BTC,ETH,SOL}USDT_q2_full.json`), the placebo controls
  (`placebo_test.json` = BTC, `placebo_{ETH,SOL}USDT.json`), the sign-shuffle
  null (`sign_shuffle.json`), the square-root-law prefactor sweep
  (`sqrt_sensitivity.json`), the generic-prior controls
  (`generic_baseline_full.json`, `generic_abides_n500.json`), the
  calibration table (`calibration_n500agents.json`) and its split-conformal
  repair (`conformal_abides_n500.json`), the transformer's metaorder
  benchmark (`benchmark_prototype.json`, `benchmark_corr_ci.json`), and the
  recipe-ablation chain on the synthetic grid (`methods_vs_gamma_v1_ci.json`
  unskewed → `_v2` 4x longer → `_v3` gamma-skewed → `_v4_full` re-run;
  soft-mix seeds `methods_vs_gamma_v4s{,_s43,_s44}.json` with their BTC
  transfer runs `crypto_BTCUSDT_v4s.json`, `crypto_BTC_v4s_s{43,44}.json`).

**Every number the paper prints is asserted against these files by
`results/check_paper_numbers.py`** (standard library only, seconds, no GPU):

```bash
python results/check_paper_numbers.py
```

Each claim names its paper location, the JSON it comes from, the stored
value and the printed one; a claim passes when the stored value rounds to
the printed one. Claims with no stored artifact are listed as UNCHECKED
rather than omitted.

Checkpoints for the recurrent state-space pairs (1.94M parameters,
about 7.5 MB each) are included under `checkpoints/`: the hard-trained pair
(`market_v10_rnn{,_ablated}`) and the soft-mix pairs for all three
training seeds (`market_v10s_softmix{,_ablated}`,
`market_v10s_s43{,_ablated}`, `market_v10s_s44{,_ablated}`). The
transformer checkpoints (about 46 MB each) and the ABIDES/Binance episode
caches remain too large for this mirror; they will be released with a
DOI upon acceptance. All data sources are public (the Binance archive;
ABIDES from the public jpmorganchase/abides-jpmc-public repository).

## Camera-ready additions (October 2026)

The camera-ready text cites four post-submission runs. Their result files are
in `results/` and are asserted by `results/check_paper_numbers.py` like every
other printed number:

- `methods_vs_gamma_v8_geom.json`, `methods_vs_gamma_v13_perm.json`,
  `methods_vs_gamma_v12_perstep.json` — one-shot and per-step transformers
  trained with the recurrent variant's recipe (gamma=2 bias +0.80 [0.67, 0.93];
  permanent-impact knob alone +1.04 [0.90, 1.18]); recipe attribution in §3.
- `paired_curve_corr.json`, `benchmark_corr_ci_v12_perstep.json` — paired
  curve-correlation differences on the shared 2,158 benchmark episodes
  (recurrent vs one-shot +0.28 [0.21, 0.34]; per-step injection +0.22
  [0.16, 0.27]; recurrence +0.06 [0.03, 0.10]); §4.2.
- `schedule_robustness_v10mix.json`, `benchmark_v10mix.json` — the
  schedule-mixture replicate of the recurrent variant (exponential-schedule
  degradation +7.8% -> +0.7%); §3.
- `seed_ensemble_coverage.json` — seed-mixture interval coverage on the ABIDES
  seed-replay tier (mixture 0.35 / 0.50 against single seeds); §4.3.
- `equity_transfer_eq50.json` — the burst tier on 50 Nasdaq names with the paper's
  transformer pair (n=3,802; paired do-head effect +1.14 [1.09, 1.19]; calm/stressed
  +1.17/+1.10); §4.2.
- `scale_calibration_cross_eq50.json` — few-shot fit of the per-domain scale
  constant k on a Nasdaq closing-auction tier (five labeled auctions: held-out
  RMSE 0.240 vs last-value 0.280); §4.3. The underlying Nasdaq trades are
  licensed from Databento and are not redistributed. `scripts/31_databento_pull.py`
  reproduces the pull from a Databento API key (`DATABENTO_API_KEY` in `.env`) and
  `data/raw/databento/manifest.json` lists every dataset, symbol, and date range it
  fetched. `scripts/32_equity_transfer.py` is the Nasdaq burst tier of §4.2,
  `scripts/33_closing_auction.py` and `scripts/36_scale_calibration.py` the
  closing-auction tier and the few-shot fit of the scale constant k of §4.3
  (modules `data/databento.py`, `data/auction.py`, `data/fomc.py`,
  `evaluation/transfer.py`; tests in `tests/test_databento.py`).
