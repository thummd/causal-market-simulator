"""Phase 2: prior_mode="market" drops MarketDoTime into the vendored
training dataloader — model-ready batches, no other trainer changes."""

import torch

from dotime_market.models.dataloader import ContinuousTemporalInterventionDataLoader
from dotime_market.prior.market_scm import MarketDoTime


def test_market_prior_mode_yields_model_ready_batches():
    loader = ContinuousTemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        prior_mode="market",
        market_coupling_gamma=1.0,
        market_core_share=0.7,
        market_impact_lambda=0.5,
        n_max=16,
        t_range=(50, 60),
        num_substeps=2,
        seed=0,
        prefetch=0,
    )
    assert isinstance(loader.prior, MarketDoTime)

    batch = next(iter(loader))
    B = 2
    assert batch["X_obs"].shape[0] == B
    assert batch["X_obs"].shape[2] == 16
    assert batch["X_int"].shape == batch["X_obs"].shape
    assert batch["Y_true"].shape == (B,)
    assert batch["Y_true_norm"].shape == (B,)  # normalize=True default
    assert batch["intervention_target"].dtype == torch.long
    # Hidden core (canonical index 1) masked in every sample of the batch.
    assert torch.all(batch["variable_mask"][:, 1] == 0.0)
    assert torch.all(batch["variable_mask"][:, 0] == 1.0)
    assert torch.all(batch["variable_mask"][:, 2] == 1.0)
    assert torch.all(batch["X_obs"][:, :, 1] == 0.0)


def test_invalid_prior_mode_still_raises():
    try:
        ContinuousTemporalInterventionDataLoader(
            num_steps=1, batch_size=1, prior_mode="bogus"
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "prior_mode" in str(e)
