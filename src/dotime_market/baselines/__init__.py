"""LAYER 1 — market baselines, self-registering into dotime.baselines on import.

Implemented: impact_models.py (AlmgrenChriss, OWPropagator — the naive
predictive foils, biased under confounding by construction); adjustment.py
(param-oracle — true-parameter conditional-mean rollout; no observable
adjustment set exists, so the oracle is handed the mechanism parameters);
market_pfn.py (market-dotpfn / ablated-dotpfn — trained checkpoint wrappers,
the ablated twin zeroes the do-information mixer inputs).

Contract (verified against dotime 0.1.1): `Baseline` is a runtime-checkable
Protocol — `name: str` + `predict(episode) -> 1-D torch.Tensor` (one raw-scale
value per query, aligned with episode.query_target/query_time). Register with
`@register("name")` on the class; duplicate names raise. Taken names include
BackDoorOLS, IV2SLS, Oracle, Zero, Mean, AR1, VAR-OLS, DoOverTimePFN.
"""

from .adjustment import ParamOracleBaseline
from .impact_models import AlmgrenChrissBaseline, OWPropagatorBaseline
from .market_pfn import AblatedDoTPFNBaseline, MarketDoTPFNBaseline

__all__ = [
    "AblatedDoTPFNBaseline",
    "AlmgrenChrissBaseline",
    "MarketDoTPFNBaseline",
    "OWPropagatorBaseline",
    "ParamOracleBaseline",
]
