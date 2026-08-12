"""Phase 2: ABIDES metaorder episodes with seed-replay counterfactuals.

Skipped entirely unless the ``abides`` extra is installed (abides-core /
abides-markets from jpmorganchase/abides-jpmc-public; not on PyPI).

The tiny-simulation integration test asserts the two properties the whole
ground-truth evaluation rests on:

1. **Seed-replay identity** — with the metaorder agent present-but-silent in
   the observational run, both runs are bit-identical before the execution
   window (same background seeds, same latency model, same kernel RNG).
2. **Causal impact** — a buy metaorder moves the with-metaorder mid price up
   relative to the counterfactual during/after the window (checked in
   aggregate over the window; single-seed noise is real, so the test uses a
   deliberately market-moving order size).
"""

import numpy as np
import pytest
import torch

pytest.importorskip("abides_markets")

from dotime_market.data.abides_builder import (  # noqa: E402
    abides_episode,
    build_abides_suite,
    run_metaorder_pair,
)

# Tiny market: 15 simulated minutes, ~100 agents — seconds per run, CI-safe.
TINY = dict(
    end_time="09:45:00",
    exec_window=("09:35:00", "09:40:00"),
    num_noise_agents=80,
    num_value_agents=10,
    num_momentum_agents=2,
    num_mm_agents=1,
)


@pytest.fixture(scope="module")
def pair():
    return run_metaorder_pair(seed=7, quantity_per_wake=300, bar="30s", **TINY)


def test_pre_onset_paths_identical(pair):
    onset = pair["int"]["onset_bar"]
    assert onset > 2
    np.testing.assert_array_equal(pair["obs"]["mid"][:onset], pair["int"]["mid"][:onset])
    np.testing.assert_array_equal(pair["obs"]["flow"][:onset], pair["int"]["flow"][:onset])


def test_metaorder_only_in_interventional_run(pair):
    assert pair["obs"]["submitted"] == 0
    assert pair["int"]["submitted"] > 0


def test_buy_metaorder_lifts_price_vs_counterfactual(pair):
    onset, end = pair["int"]["onset_bar"], pair["int"]["end_bar"]
    lift = (pair["int"]["mid"][onset:end] - pair["obs"]["mid"][onset:end]).mean()
    assert lift > 0.0, f"buy metaorder should lift the mid vs counterfactual, got {lift}"


def test_episode_shape_and_semantics(pair):
    rng = np.random.RandomState(0)
    ep = abides_episode(pair, rng=rng, n_queries=2)
    T = len(pair["obs"]["mid"])
    assert ep.x_obs.shape == (T, 2)
    assert ep.x_int.shape == (T, 2)
    assert ep.structure == "abides_metaorder"
    assert ep.intervention.targets == [0]
    onset = pair["int"]["onset_bar"]
    for q in range(2):
        q_idx = ep.metadata["query_idx"][q]
        assert q_idx >= onset
        assert int(ep.query_target[q]) == 1
        assert float(ep.y_true[q]) == pytest.approx(float(ep.x_int[q_idx, 1]))
    # Flow column diverges only from the window on.
    assert torch.equal(ep.x_obs[:onset], ep.x_int[:onset])
    # Standardized units: pre-onset stds ~1, do-value in rescaled flow units.
    assert float(ep.x_obs[:onset, 0].std()) == pytest.approx(1.0, rel=1e-4)
    assert float(ep.x_obs[:onset, 1].std()) == pytest.approx(1.0, rel=1e-4)
    shares_per_bar = pair["shares_per_bar"]
    assert float(ep.intervention.values) == pytest.approx(
        shares_per_bar / ep.metadata["flow_scale"], rel=1e-5
    )


def test_episode_standardize_off_keeps_physical_units(pair):
    rng = np.random.RandomState(0)
    ep = abides_episode(pair, rng=rng, standardize=False)
    assert ep.metadata["flow_scale"] == 1.0
    assert float(ep.intervention.values) == pytest.approx(pair["shares_per_bar"], rel=1e-6)


def test_episode_soft_intervention_encoding(pair):
    from dotime import InterventionType

    rng = np.random.RandomState(0)
    ep = abides_episode(pair, rng=rng, intervention_kind="soft")
    assert ep.intervention.intervention_type == InterventionType.SOFT
    hard = abides_episode(pair, rng=np.random.RandomState(0), intervention_kind="hard")
    # Same value either way — only the semantic label differs.
    assert float(ep.intervention.values) == pytest.approx(float(hard.intervention.values))
    with pytest.raises(ValueError, match="intervention_kind"):
        abides_episode(pair, rng=rng, intervention_kind="ramp")


def test_suite_runs_stock_baselines():
    from dotime.baselines import get
    from dotime.evaluation import evaluate

    import dotime_market.baselines  # noqa: F401

    suite = build_abides_suite(n_episodes=2, seed0=11, quantity_per_wake=300,
                               bar="30s", **TINY)
    assert len(suite) == 2
    for name in ("Mean", "AlmgrenChriss", "OWPropagator"):
        results = evaluate(get(name), suite)
        assert np.isfinite(results.pooled["rmse"])
