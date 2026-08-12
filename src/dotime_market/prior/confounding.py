"""Core/reaction order-flow confounder — the load-bearing Phase-1 mechanism.

Market story
------------
Observable order flow ``A`` mixes two sources:

- **Core flow** ``U``: exogenous, informational trading (participants acting
  on a private signal about fundamentals). It is *hidden* from the model and
  moves the price twice — through its trading (``U -> A``) and directly
  through information revelation (``U -> Y``, the coupling ``gamma`` owned by
  :mod:`.coupling`). ``U`` is the confounder that opens the
  ``E[Y | do(A)]`` vs ``E[Y | A]`` gap.
- **Reaction flow**: endogenous, uninformed response — market making,
  inventory control, noise trading. Modelled as the flow variable's own
  mean-reversion ``theta_flow`` plus idiosyncratic diffusion ``sigma_flow``.

Dynamics (both stock :class:`dotime.continuous.OUMechanism` instances)::

    dU = -theta_core * U dt + sigma_core dW_U            (hidden core flow)
    dA = (-theta_flow * A + w * U) dt + sigma_flow dW_A   (observable flow)

Why a *variance-share* parameterization
---------------------------------------
Stock ``dotime`` draws parental weights ``N(0, weight_scale^2)`` with a single
global scale, so per-episode confounding strength is an uncontrolled draw —
the Phase-0 sweep found roughly half of episodes at the default setting have a
near-zero gap (median gap far below the mean). The paper's headline figure
sweeps confounding strength explicitly, so the knob here is

    ``core_share`` = the fraction of ``A``'s *stationary variance* driven by
    the hidden core flow,

and the edge weight ``w`` is solved for exactly. From the stationary Lyapunov
equation of the linear system above (``Var`` and ``Cov`` at stationarity)::

    Var(U)    = sigma_core^2 / (2 theta_core)
    Cov(A, U) = w * Var(U) / (theta_flow + theta_core)
    Var(A)    = sigma_flow^2 / (2 theta_flow)                          [reaction]
              + w^2 * Var(U) / (theta_flow * (theta_flow + theta_core)) [core-driven]

so ``core_share = core_driven / (core_driven + reaction)`` inverts to::

    w = sqrt( core_share / (1 - core_share)
              * reaction_var * theta_flow * (theta_flow + theta_core) / Var(U) )

``core_share = 0`` gives ``w = 0`` and :meth:`CoreReactionConfounder.mechanisms`
returns a parentless flow mechanism — bit-identical to a stock root OU
variable (the reduction test).

The moments are exact for the continuous-time SDE; Euler–Maruyama
discretization inflates them by ``O(theta * dt / num_substeps)``, which is an
integrator property, not a prior property (tests use fine substeps).

Assembly (Phase 1, :mod:`.market_scm`): the core index goes into
``_SampledSCMContext.hidden_vars_topo`` so the existing dotime machinery
simulates it but masks it from the model's tokens; :mod:`.coupling` adds the
``U -> Y`` / ``A -> Y`` outcome mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dotime.continuous import OUMechanism

__all__ = ["CoreReactionConfounder", "sample_core_reaction_confounder"]


@dataclass(frozen=True)
class CoreReactionConfounder:
    """Calibrated hidden-core / observable-flow block of the market prior.

    Parameters
    ----------
    core_share : float
        Target fraction of the flow variable's stationary variance driven by
        the hidden core process. Must lie in ``[0, 1)``; ``0`` disconnects the
        core (stock-dotime reduction), values approaching ``1`` require an
        unbounded edge weight.
    theta_core, sigma_core : float
        Mean-reversion rate and diffusion of the hidden core flow ``U``
        (both positive). ``1 / theta_core`` is the persistence timescale of
        the informational signal.
    theta_flow, sigma_flow : float
        Mean-reversion rate and diffusion of the observable flow ``A``
        (both positive) — the *reaction* component: inventory control pulls
        flow back to zero, uninformed trading supplies idiosyncratic noise.
    """

    core_share: float
    theta_core: float = 1.0
    sigma_core: float = 0.4
    theta_flow: float = 1.0
    sigma_flow: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.core_share < 1.0:
            raise ValueError(f"core_share must be in [0, 1), got {self.core_share}")
        for name in ("theta_core", "sigma_core", "theta_flow", "sigma_flow"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    # ------------------------------------------------------------ calibration

    @property
    def core_weight(self) -> float:
        """The ``U -> A`` edge weight ``w`` realizing ``core_share`` exactly."""
        if self.core_share == 0.0:
            return 0.0
        ratio = self.core_share / (1.0 - self.core_share)
        return math.sqrt(ratio * self.reaction_var() / self._unit_core_gain())

    def _unit_core_gain(self) -> float:
        """Core-driven stationary variance of ``A`` per unit ``core_weight**2``."""
        return self.stationary_core_var() / (self.theta_flow * (self.theta_flow + self.theta_core))

    # ----------------------------------------- stationary moments (SDE-exact)

    def stationary_core_var(self) -> float:
        """``Var(U) = sigma_core^2 / (2 theta_core)``."""
        return self.sigma_core**2 / (2.0 * self.theta_core)

    def reaction_var(self) -> float:
        """Reaction (idiosyncratic) component of ``Var(A)``."""
        return self.sigma_flow**2 / (2.0 * self.theta_flow)

    def core_driven_var(self) -> float:
        """Core-driven component of ``Var(A)``; equals
        ``core_share / (1 - core_share) * reaction_var()`` by construction."""
        return self.core_weight**2 * self._unit_core_gain()

    def stationary_flow_var(self) -> float:
        """``Var(A)`` — reaction plus core-driven components."""
        return self.reaction_var() + self.core_driven_var()

    def stationary_flow_core_cov(self) -> float:
        """``Cov(A, U)`` — the moment that produces naive-regression bias
        once :mod:`.coupling` adds the ``U -> Y`` edge."""
        return self.core_weight * self.stationary_core_var() / (self.theta_flow + self.theta_core)

    # ------------------------------------------------------------- mechanisms

    def mechanisms(self, core_idx: int, flow_idx: int) -> tuple[OUMechanism, OUMechanism]:
        """Build the ``(core, flow)`` mechanisms for a :class:`ContinuousSCM`.

        Parameters
        ----------
        core_idx, flow_idx : int
            Topological indices the caller assigns to ``U`` and ``A``. The
            core must precede the flow (dotime's samplers draw parents from
            earlier topological positions; keeping that invariant here means
            downstream graph logic never needs a special case).

        Returns
        -------
        tuple of OUMechanism
            ``(core_mechanism, flow_mechanism)``. At ``core_share = 0`` the
            flow mechanism is parentless — identical to a stock root OU.
        """
        if not 0 <= core_idx < flow_idx:
            raise ValueError(
                f"need 0 <= core_idx < flow_idx (topological order), "
                f"got core_idx={core_idx}, flow_idx={flow_idx}"
            )
        core = OUMechanism(theta=self.theta_core, sigma=self.sigma_core)
        w = self.core_weight
        if w == 0.0:
            flow = OUMechanism(theta=self.theta_flow, sigma=self.sigma_flow)
        else:
            flow = OUMechanism(
                theta=self.theta_flow,
                sigma=self.sigma_flow,
                parent_weights=torch.tensor([w], dtype=torch.float32),
                parents=(core_idx,),
            )
        return core, flow


def sample_core_reaction_confounder(
    core_share_range: tuple = (0.3, 0.9),
    theta_range: tuple = (0.5, 2.0),
    sigma_range: tuple = (0.2, 0.6),
    generator: torch.Generator | None = None,
) -> CoreReactionConfounder:
    """Draw a :class:`CoreReactionConfounder` from priors on its parameters.

    - ``core_share ~ Uniform(core_share_range)`` — degenerate ranges
      (``lo == hi``) are allowed so sweep configs can pin the knob.
    - ``theta_core, theta_flow ~ Uniform(theta_range)`` and
      ``sigma_core, sigma_flow ~ Uniform(sigma_range)`` independently,
      matching :func:`dotime.continuous.sample_ou_mechanism`'s hyperpriors.

    Parameters
    ----------
    core_share_range : tuple of float
        Uniform prior bounds on the core variance share, within ``[0, 1)``.
    theta_range, sigma_range : tuple of float
        Uniform prior bounds (positive) shared by the core and flow processes.
    generator : torch.Generator, optional
        RNG for reproducibility.
    """
    if not 0.0 <= core_share_range[0] <= core_share_range[1] < 1.0:
        raise ValueError(f"invalid core_share_range: {core_share_range}")
    if theta_range[0] <= 0 or theta_range[1] < theta_range[0]:
        raise ValueError(f"invalid theta_range: {theta_range}")
    if sigma_range[0] <= 0 or sigma_range[1] < sigma_range[0]:
        raise ValueError(f"invalid sigma_range: {sigma_range}")

    u = torch.empty(5)
    u.uniform_(0.0, 1.0, generator=generator)

    def _unif(i: int, lo: float, hi: float) -> float:
        return float(lo + u[i] * (hi - lo))

    return CoreReactionConfounder(
        core_share=_unif(0, *core_share_range),
        theta_core=_unif(1, *theta_range),
        sigma_core=_unif(2, *sigma_range),
        theta_flow=_unif(3, *theta_range),
        sigma_flow=_unif(4, *sigma_range),
    )
