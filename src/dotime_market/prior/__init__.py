"""LAYER 2 — market prior mechanisms. The only code touching dotime internals.

Implemented (Phase 1): confounding.py (CoreReactionConfounder — load-bearing),
coupling.py (InformationalCoupling — the swept gamma knob), market_scm.py
(MarketContinuousSCMSampler + MarketDoTime via the _sample_scm_context hook).
Planned (docs/market_extension_architecture.md): interfaces.py, drift.py,
graph.py, timing.py.

Integration seams in released dotime (verified 2026-07-02):
- ContinuousSCM(mechanisms): duck-typed — `.parents`, `.sigma`, `.drift(x_self, x_parents)`.
- ContinuousExtendedPrior._sample_scm_context(): documented subclass hook for the sampler.
- _sample_intervention_kind / _sample_intervention_value: overridable.
- Intervention *timing* has no hook — override generate_sample() (Phase 3, Hawkes).
- Jumps: subclass ContinuousSCM, extend _draw_noise (pre-draw jump randomness to keep
  shared-noise counterfactual pairs) and _step (add the jump increment).
"""

from .confounding import CoreReactionConfounder, sample_core_reaction_confounder
from .coupling import InformationalCoupling, sample_informational_coupling
from .interfaces import (
    EpisodeSCMSampler,
    InterventionTiming,
    MechanismFactory,
    UniformWindowTiming,
)
from .market_scm import MarketContinuousSCMSampler, MarketDoTime

__all__ = [
    "CoreReactionConfounder",
    "EpisodeSCMSampler",
    "InformationalCoupling",
    "InterventionTiming",
    "MarketContinuousSCMSampler",
    "MarketDoTime",
    "MechanismFactory",
    "UniformWindowTiming",
    "sample_core_reaction_confounder",
    "sample_informational_coupling",
]
