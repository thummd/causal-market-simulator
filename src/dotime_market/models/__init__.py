"""Continuous-time causal PFN model + trainer, adapted from the cited construction's source.

Vendored files (import-rewritten from the old ``dotime`` 0.1.0 layout to the
released ``dotime`` >= 0.1.1 layout; do not edit the source repo):

- ``time_embedding.py``   <- dotime/model/continuous/time_embedding.py
- ``encoder.py``          <- dotime/model/continuous/encoder.py
- ``continuous_model.py`` <- dotime/model/continuous/model.py
- ``dataloader.py``       <- dotime/data/continuous_dataloader.py
- ``train.py``            <- dotime/training/continuous_trainer.py

Public API
----------
:class:`ContinuousDoOverTimePFN`, :class:`ContinuousTemporalEncoder`,
:class:`FourierTimeEmbedding`, :class:`DeltaTEmbedding`,
:func:`relative_to_intervention`,
:class:`ContinuousTemporalInterventionDataLoader`, :func:`train_continuous`.
"""

from .continuous_model import ContinuousDoOverTimePFN
from .dataloader import ContinuousTemporalInterventionDataLoader
from .encoder import ContinuousTemporalEncoder
from .time_embedding import (
    DeltaTEmbedding,
    FourierTimeEmbedding,
    relative_to_intervention,
)
from .train import train_continuous

__all__ = [
    "ContinuousDoOverTimePFN",
    "ContinuousTemporalEncoder",
    "ContinuousTemporalInterventionDataLoader",
    "DeltaTEmbedding",
    "FourierTimeEmbedding",
    "relative_to_intervention",
    "train_continuous",
]
