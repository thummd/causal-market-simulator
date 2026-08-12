"""Phase 2: market suites feed the stock evaluation harness unchanged, and
the naive impact-model baselines (AlmgrenChriss, OWPropagator) conform to the
Baseline protocol — fitting well without confounding and degrading with gamma.
"""

import numpy as np
import pytest
import torch

import dotime_market.baselines  # noqa: F401  (self-registration side effect)
from dotime import InterventionSpec, InterventionType
from dotime.baselines import available, get
from dotime.benchmarks import Episode
from dotime.evaluation import evaluate
from dotime_market.data.synthetic import build_market_suite
from dotime_market.prior.market_scm import MarketDoTime


def _suite(gamma, core_share, n_episodes=20, seed=0, n_queries=1):
    # num_substeps=4: at the hyperprior's theta=2 upper end, the default
    # single EM step of dt=1 sits at the stability edge (AR coef -1).
    prior = MarketDoTime(
        coupling_gamma=gamma, core_share=core_share, impact_lambda=0.75, seed=seed,
        num_substeps=4,
    )
    return build_market_suite(prior, n_episodes=n_episodes, T=120, n_queries=n_queries, seed=seed)


# ------------------------------------------------------------ suite builder


def test_market_suite_episodes_are_valid():
    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=5, n_queries=2)
    assert len(suite) == 5
    for ep in suite:
        assert ep.x_obs.shape == (120, 3)
        assert ep.x_int.shape == (120, 3)
        assert ep.structure == "market_core_reaction"
        assert ep.intervention.targets == [0]
        assert ep.intervention.intervention_type == InterventionType.HARD
        # Hidden core column is zeroed in both trajectories.
        assert torch.all(ep.x_obs[:, 1] == 0.0)
        assert torch.all(ep.x_int[:, 1] == 0.0)
        # Queries: outcome variable, post-onset, y_true from X_int.
        onset = min(ep.intervention.times)
        for q in range(2):
            assert int(ep.query_target[q]) == 2
            q_idx = ep.metadata["query_idx"][q]
            assert q_idx >= onset
            assert float(ep.y_true[q]) == pytest.approx(float(ep.x_int[q_idx, 2]))
            assert 0.0 <= float(ep.query_time[q]) <= 1.0
        for key in ("coupling_gamma", "core_share", "confounding_bias_ana", "do_slope_ana"):
            assert key in ep.metadata


def test_market_suite_requires_regular_dt1_schedule():
    prior = MarketDoTime(seed=0, schedule="exponential")
    with pytest.raises(ValueError, match="regular"):
        build_market_suite(prior, n_episodes=1)


# ------------------------------------------------------- baseline protocol


def test_impact_models_registered():
    names = available()
    assert "AlmgrenChriss" in names
    assert "OWPropagator" in names


@pytest.mark.parametrize("name", ["AlmgrenChriss", "OWPropagator"])
def test_predict_shape_and_finiteness(name):
    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=3, n_queries=3)
    model = get(name)
    for ep in suite:
        pred = model.predict(ep)
        assert pred.shape == (3,)
        assert torch.isfinite(pred).all()


@pytest.mark.parametrize("name", ["AlmgrenChriss", "OWPropagator"])
def test_treatment_query_inside_window_returns_clamp(name):
    x = torch.zeros(50, 3)
    x[:, 0] = torch.linspace(-1, 1, 50)
    ep = Episode(
        x_obs=x,
        x_int=x.clone(),
        intervention=InterventionSpec(
            targets=[0], times=list(range(30, 40)),
            intervention_type=InterventionType.HARD, values=1.5,
        ),
        y_true=torch.tensor([1.5]),
        query_target=torch.tensor([0]),
        query_time=torch.tensor([35.0 / 49.0]),
        structure="market_core_reaction",
    )
    pred = get(name).predict(ep)
    assert float(pred[0]) == pytest.approx(1.5)


# ------------------------------------------------------------- integration


def test_evaluate_runs_stock_and_market_baselines():
    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=8)
    for name in ("Mean", "AlmgrenChriss", "OWPropagator"):
        results = evaluate(get(name), suite)
        assert "rmse" in results.pooled
        assert np.isfinite(results.pooled["rmse"])


def test_ow_beats_mean_without_confounding():
    """gamma=0, core_share=0: OW is (approximately) the right model class, so
    modelling the intervention must beat ignoring it."""
    suite = _suite(gamma=0.0, core_share=0.0, n_episodes=30, seed=3)
    rmse_ow = evaluate(get("OWPropagator"), suite).pooled["rmse"]
    rmse_mean = evaluate(get("Mean"), suite).pooled["rmse"]
    assert rmse_ow < rmse_mean


def _ac_implied_slopes(suite) -> list[float]:
    """Back out AC's fitted impact coefficient per episode:
    pred = mean(Y_pre) + beta * c  =>  beta = (pred - mean(Y_pre)) / c."""
    model = get("AlmgrenChriss")
    betas = []
    for ep in suite:
        onset = min(ep.intervention.times)
        pred = float(model.predict(ep)[0])
        c = float(ep.intervention.values)
        base = float(ep.x_obs[:onset, 2].mean())
        betas.append((pred - base) / c)
    return betas


def test_ac_implied_slope_tracks_analytic_naive_slope():
    """AC *is* the naive impact regression — its fitted coefficient must
    agree with the closed-form stationary naive slope on average."""
    suite = _suite(gamma=2.0, core_share=0.7, n_episodes=30, seed=5)
    betas = _ac_implied_slopes(suite)
    ana = [ep.metadata["naive_slope_ana"] for ep in suite]
    # ~0.13 systematic attenuation: the fit window is short (30-90 obs),
    # includes the from-zero transient, and EM discretization shaves a bit.
    # The paired-suite shift test below carries the precision claim.
    assert float(np.mean(betas)) == pytest.approx(float(np.mean(ana)), abs=0.2)


def test_ac_bias_shift_matches_analytic_confounding_bias():
    """The point of the PoC, at baseline granularity: opening the
    informational channel shifts AC's fitted coefficient by exactly the
    analytic confounding bias (paired suites share all other draws)."""
    clean = _suite(gamma=0.0, core_share=0.7, n_episodes=30, seed=5)
    confounded = _suite(gamma=2.0, core_share=0.7, n_episodes=30, seed=5)
    shift = float(np.mean(_ac_implied_slopes(confounded)) - np.mean(_ac_implied_slopes(clean)))
    bias_ana = float(np.mean([ep.metadata["confounding_bias_ana"] for ep in confounded]))
    assert bias_ana > 0.2  # the knob is wide open
    assert shift == pytest.approx(bias_ana, rel=0.35)
