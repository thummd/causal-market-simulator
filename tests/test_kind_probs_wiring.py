"""Regression: intervention-distribution knobs must reach the prior.

The original wiring accepted --intervention-kind-probs in the CLI and
train_continuous, printed and wandb-logged it, but never forwarded it to the
dataloader -- every market checkpoint through v9 trained with the hard-only
default (1, 0, 0) regardless of the flag. This test pins the fix.
"""

from dotime_market.models.dataloader import ContinuousTemporalInterventionDataLoader


def test_kind_probs_reach_the_prior():
    dl = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2, prior_mode="market",
        intervention_kind_probs=(0.5, 0.5, 0.0), soft_shift_scale=2.0)
    assert tuple(dl.prior.intervention_kind_probs) == (0.5, 0.5, 0.0)


def test_default_stays_hard_only():
    dl = ContinuousTemporalInterventionDataLoader(
        num_steps=1, batch_size=2, prior_mode="market")
    assert tuple(dl.prior.intervention_kind_probs) == (1.0, 0.0, 0.0)
