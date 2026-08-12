"""Phase 2: the param-oracle is unbiased where the naive foils are biased,
and the PFN checkpoint wrappers load/predict/ablate correctly."""

import numpy as np
import pytest
import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import available, get
from dotime.evaluation import evaluate
from dotime_market.data.synthetic import build_market_suite
from dotime_market.prior.market_scm import MarketDoTime


def _suite(gamma, core_share, n_episodes=20, seed=0):
    prior = MarketDoTime(
        coupling_gamma=gamma, core_share=core_share, impact_lambda=0.75, seed=seed,
        num_substeps=4,
    )
    return build_market_suite(prior, n_episodes=n_episodes, T=120, seed=seed)


# ------------------------------------------------------------- param-oracle


def test_param_oracle_registered():
    assert "param-oracle" in available()


def test_param_oracle_beats_naive_foils_under_confounding():
    suite = _suite(gamma=2.0, core_share=0.7, n_episodes=30, seed=11)
    rmse = {
        name: evaluate(get(name), suite).pooled["rmse"]
        for name in ("param-oracle", "AlmgrenChriss", "OWPropagator")
    }
    assert rmse["param-oracle"] < rmse["AlmgrenChriss"]
    assert rmse["param-oracle"] < rmse["OWPropagator"]


def test_param_oracle_is_unbiased_in_impact_direction():
    """Signed error projected on the clamp sign: the oracle centres on zero,
    the naive foils shift with the confounding bias."""
    suite = _suite(gamma=2.0, core_share=0.7, n_episodes=40, seed=11)

    def signed_error(name):
        model = get(name)
        errs = []
        for ep in suite:
            c = float(ep.intervention.values)
            pred = float(model.predict(ep)[0])
            errs.append(np.sign(c) * (pred - float(ep.y_true[0])))
        return float(np.mean(errs))

    assert abs(signed_error("param-oracle")) < 0.25
    assert signed_error("AlmgrenChriss") > 2 * abs(signed_error("param-oracle"))


def test_param_oracle_raises_without_market_metadata():
    suite = _suite(gamma=1.0, core_share=0.5, n_episodes=1)
    ep = suite[0]
    ep.metadata = {}
    with pytest.raises(RuntimeError, match="param-oracle requires"):
        get("param-oracle").predict(ep)


# ------------------------------------------------------- checkpoint wrappers


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """An untrained but loadable checkpoint in the trainer's save format."""
    from dotime_market.models.continuous_model import ContinuousDoOverTimePFN

    cfg = dict(
        n_max=16, embed_size=32, n_encoder_layers=1, n_cross_attn_heads=2,
        encoder_backend="transformer", context_window=64, n_mixer_layers=1,
        num_time_frequencies=8, time_min_freq=0.01, time_max_freq=10.0,
        tau_levels=[0.1, 0.5, 0.9], head_type="quantile", n_buckets=100,
    )
    torch.manual_seed(0)
    model = ContinuousDoOverTimePFN(n_heads=2, **cfg)
    path = tmp_path_factory.mktemp("ckpt") / "continuous_do_over_time_pfn_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, path)
    return str(path)


def test_pfn_wrappers_registered():
    names = available()
    assert "market-dotpfn" in names
    assert "ablated-dotpfn" in names


def test_pfn_wrapper_predicts_on_market_suite(tiny_checkpoint):
    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=2)
    model = get("market-dotpfn", checkpoint=tiny_checkpoint)
    for ep in suite:
        pred = model.predict(ep)
        assert pred.shape == (1,)
        assert torch.isfinite(pred).all()


def test_wrapper_passes_intervention_kind(tiny_checkpoint):
    """The batch's intervention_type index must follow the episode's kind
    (HARD -> 0, SOFT -> 1), and the causal model's output must depend on it."""
    from dotime import InterventionType

    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=1)
    ep = suite[0]
    model = get("market-dotpfn", checkpoint=tiny_checkpoint)

    batch_hard = model._episode_to_batch(ep, 0)
    assert int(batch_hard["intervention_type"][0]) == 0
    p_hard = float(model.predict(ep)[0])
    ep.intervention.intervention_type = InterventionType.SOFT
    batch_soft = model._episode_to_batch(ep, 0)
    assert int(batch_soft["intervention_type"][0]) == 1
    p_soft = float(model.predict(ep)[0])
    assert p_hard != pytest.approx(p_soft, abs=1e-7)


def test_predict_quantiles_shape_and_scale(tiny_checkpoint):
    """(n_queries, n_tau) raw-scale quantiles; the mean prediction lies
    inside the de-normalized quantile range (same stats, same batch path)."""
    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=1)
    ep = suite[0]
    model = get("market-dotpfn", checkpoint=tiny_checkpoint)
    q = model.predict_quantiles(ep)
    assert q.shape == (1, len(model.tau_levels))
    assert torch.isfinite(q).all()
    pred = float(model.predict(ep)[0])
    assert float(q.min()) <= pred <= float(q.max())


def test_ablated_wrapper_ignores_intervention_value(tiny_checkpoint):
    """The ablated twin must be invariant to the do-value; the causal wrapper
    must not be (even untrained, the value feeds the mixer)."""
    suite = _suite(gamma=1.0, core_share=0.7, n_episodes=1)
    ep = suite[0]
    causal = get("market-dotpfn", checkpoint=tiny_checkpoint)
    ablated = get("ablated-dotpfn", checkpoint=tiny_checkpoint)

    p_causal_1 = float(causal.predict(ep)[0])
    p_ablated_1 = float(ablated.predict(ep)[0])
    ep.intervention.values = float(ep.intervention.values) + 5.0
    p_causal_2 = float(causal.predict(ep)[0])
    p_ablated_2 = float(ablated.predict(ep)[0])

    assert p_ablated_1 == pytest.approx(p_ablated_2, abs=1e-6)
    assert p_causal_1 != pytest.approx(p_causal_2, abs=1e-6)
