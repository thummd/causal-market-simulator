"""The pluggable prior interfaces — the upstream candidate for dotime.

These are extracted from the *working* Phase-1/2 mechanisms rather than
designed a priori (the original architecture sketch guessed step-wise ABCs;
what two shipped implementations actually converged on is calibrated
**mechanism factories** plus the sampler duck-type that
``ContinuousExtendedPrior._sample_scm_context`` already relies on).

Upstreaming plan (docs/market_extension_architecture.md): move this module
into ``dotime`` as public API; the market mechanisms then become registered
plugins instead of an internals-touching extension.

- :class:`MechanismFactory` — anything that contributes calibrated
  per-variable mechanisms to a :class:`ContinuousSCM`. Implemented by
  :class:`~dotime_market.prior.confounding.CoreReactionConfounder` and
  :class:`~dotime_market.prior.coupling.InformationalCoupling`.
- :class:`EpisodeSCMSampler` — the duck-type behind the
  ``_sample_scm_context()`` hook (both dotime's ``ContinuousTSCMSampler``
  and :class:`~dotime_market.prior.market_scm.MarketContinuousSCMSampler`
  satisfy it structurally).
- :class:`InterventionTiming` + :class:`UniformWindowTiming` — the missing
  seam: intervention-window timing is inlined in ``generate_sample`` today
  (the one place Phase-3 Hawkes timing must override). ``UniformWindowTiming``
  reproduces the stock uniform-window behaviour as the reference
  implementation; the proposal for dotime 0.1.2 is a
  ``_sample_intervention_window(times, x_obs)`` hook accepting this
  interface (state-dependent timing needs the window sampled *after*
  ``X_obs``, a safe reorder since ``X_obs`` never depends on the window).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class MechanismFactory(Protocol):
    """Contributes calibrated per-variable mechanisms to a ContinuousSCM.

    ``mechanisms(*var_indices)`` returns one mechanism per variable this
    factory owns, in the order of the passed topological indices. Mechanisms
    must satisfy dotime's duck-typed contract (``.parents``, ``.sigma``,
    ``.drift(x_self, x_parents)``); parent indices may only reference
    strictly earlier topological positions.
    """

    def mechanisms(self, *var_indices: int) -> tuple: ...


@runtime_checkable
class EpisodeSCMSampler(Protocol):
    """The sampler duck-type behind ``_sample_scm_context()``.

    ``sample()`` returns a tuple whose first element is the SCM; trailing
    elements are implementation-specific per-episode metadata (the market
    sampler returns its parameter objects for diagnostics).
    """

    @property
    def n_vars(self) -> int: ...

    def get_intervention_target(self) -> int: ...

    def get_outcome_var(self) -> int: ...

    def get_hidden_vars(self) -> list: ...

    def sample(self, generator: torch.Generator | None = None) -> tuple: ...


class InterventionTiming(ABC):
    """Samples the intervention window — the seam dotime does not expose yet.

    ``sample_window`` may condition on the realized observational trajectory
    (state-dependent timing, e.g. Hawkes-triggered metaorders); stock
    behaviour ignores it.
    """

    @abstractmethod
    def sample_window(
        self,
        times: torch.Tensor,
        x_obs: torch.Tensor | None,
        rng: np.random.RandomState,
    ) -> tuple:
        """Return ``(t_start, t_end)`` in absolute time units."""


class UniformWindowTiming(InterventionTiming):
    """Reference implementation: dotime's stock uniform window.

    Mirrors ``ContinuousExtendedPrior.generate_sample`` step 3: window length
    is a uniform fraction of the span, start uniform in
    ``[t0 + 0.3 * span, t_end - length]`` (with the same fallback when the
    window does not fit).
    """

    def __init__(self, window_frac: tuple = (0.1, 0.3), min_len: float = 2.0):
        if not 0 < window_frac[0] <= window_frac[1]:
            raise ValueError(f"invalid window_frac: {window_frac}")
        self.window_frac = tuple(window_frac)
        self.min_len = float(min_len)

    def sample_window(
        self,
        times: torch.Tensor,
        x_obs: torch.Tensor | None,
        rng: np.random.RandomState,
    ) -> tuple:
        t0, t1 = float(times[0]), float(times[-1])
        span = t1 - t0
        win_len = max(self.min_len, float(rng.uniform(*self.window_frac)) * span)
        earliest = t0 + 0.3 * span
        latest = t1 - win_len
        if latest <= earliest:
            earliest = t0 + 0.1 * span
            latest = t1 - self.min_len / 2.0
        t_start = float(rng.uniform(earliest, latest))
        return t_start, t_start + win_len
