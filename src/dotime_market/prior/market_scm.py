"""Assemble the Phase-1 market prior: MarketContinuousSCMSampler + MarketDoTime.

Topology (fixed, minimal — realism layers arrive in Phase 3)::

    U (topo 0, HIDDEN)  --w-->  A (topo 1, intervention target)
     \\--gamma-->  Y (topo 2, outcome)  <--impact_lambda--  A

``U`` is the hidden informational core flow (:mod:`.confounding`), ``A`` the
observable order flow (a hard ``do(A = c)`` is the metaorder), ``Y`` the
price/return (:mod:`.coupling`). The core index goes into
``hidden_vars_topo``, so stock dotime machinery simulates ``U`` but masks it
from the model's tokens and ``variable_mask`` — the same semantics as the
hidden ``U`` in the named ``unobserved_confounder`` structure.

:class:`MarketDoTime` extends :class:`dotime.continuous.ContinuousExtendedPrior`
via the documented ``_sample_scm_context()`` hook only (the exact pattern of
``RandomContinuousExtendedPrior``); schedule sampling, intervention kind/value,
counterfactual pairing, canonical permutation, and query sampling are all
inherited unchanged. The knobs ``coupling_gamma`` / ``core_share`` /
``impact_lambda`` accept either a scalar (pinned — sweep grid points, CI
gates) or a ``(lo, hi)`` tuple (sampled per episode — training).
"""

from __future__ import annotations

import torch

from dotime.continuous import ContinuousSCM
from dotime.continuous.extended_prior import (
    ContinuousExtendedPrior,
    _SampledSCMContext,
)

from .confounding import CoreReactionConfounder, sample_core_reaction_confounder
from .coupling import InformationalCoupling, sample_informational_coupling
from .hawkes import HawkesFlowSCM
from .jumps import JumpDiffusionSCM

__all__ = ["MarketContinuousSCMSampler", "MarketDoTime", "CORE_TOPO", "FLOW_TOPO", "PRICE_TOPO"]

CORE_TOPO, FLOW_TOPO, PRICE_TOPO = 0, 1, 2
N_VARS = 3


def _as_range(value, name: str) -> tuple[float, float]:
    """Normalise a scalar-or-``(lo, hi)`` knob to a ``(lo, hi)`` tuple."""
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    lo, hi = value
    if lo > hi:
        raise ValueError(f"{name} range must have lo <= hi, got {value}")
    return (float(lo), float(hi))


class MarketContinuousSCMSampler:
    """Sample a fresh 3-variable market SCM per call.

    Parameters
    ----------
    core_share, coupling_gamma, impact_lambda : float or (lo, hi)
        The market knobs (see :mod:`.confounding` / :mod:`.coupling`).
        Scalars pin the value; tuples sample uniformly per episode.
    theta_range, sigma_range : tuple of float
        OU hyperpriors shared by all three processes, matching
        :func:`dotime.continuous.sample_ou_mechanism` defaults.
    coupling_gamma_power : float
        Skews the per-episode gamma draw toward the top of its range:
        ``gamma = lo + (hi - lo) * u**(1/power)`` with ``u ~ U(0, 1)``.
        ``1.0`` (default) is uniform; ``power = 3`` puts ~58% of the mass in
        the upper third of the range (mean at 3/4 of the span). Training
        lever for the residual high-gamma bias — evaluation grids keep
        pinned gammas, where the power is irrelevant.
    perm_prob : float
        Probability of the unit-root price carve-out (``theta_Y = 0``:
        permanent, duration-accruing mechanical impact — see
        :class:`.coupling.IntegratorMechanism`). ``0.0`` (default) draws
        nothing and keeps seeded streams bit-identical to the legacy prior.
    seed : int
        Seeds the internal :class:`torch.Generator`; the sampler advances
        it on every call.
    """

    def __init__(
        self,
        core_share=(0.3, 0.9),
        coupling_gamma=(0.0, 2.0),
        impact_lambda=(0.25, 1.0),
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        coupling_gamma_power: float = 1.0,
        jump_rate=(0.0, 0.0),
        jump_scale_mult: float = 3.0,
        theta_price_range=None,
        impact_lambda_log_range=None,
        perm_prob: float = 0.0,
        hawkes_rate=(0.0, 0.0),
        hawkes_alpha: float = 0.5,
        hawkes_beta: float = 1.0,
        hawkes_scale_mult: float = 1.5,
        hawkes_persist: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.theta_price_range = tuple(theta_price_range) if theta_price_range else None
        self.perm_prob = float(perm_prob)
        self.impact_lambda_log_range = tuple(impact_lambda_log_range) if impact_lambda_log_range else None
        if coupling_gamma_power <= 0:
            raise ValueError(f"coupling_gamma_power must be positive, got {coupling_gamma_power}")
        self.jump_rate_range = _as_range(jump_rate, "jump_rate")
        self.hawkes_rate_range = _as_range(hawkes_rate, "hawkes_rate")
        self.hawkes_alpha = float(hawkes_alpha)
        self.hawkes_beta = float(hawkes_beta)
        self.hawkes_scale_mult = float(hawkes_scale_mult)
        self.hawkes_persist = float(hawkes_persist)
        self.jump_scale_mult = float(jump_scale_mult)
        self.core_share_range = _as_range(core_share, "core_share")
        self.coupling_gamma_range = _as_range(coupling_gamma, "coupling_gamma")
        self.impact_lambda_range = _as_range(impact_lambda, "impact_lambda")
        self.theta_range = tuple(theta_range)
        self.sigma_range = tuple(sigma_range)
        self.coupling_gamma_power = float(coupling_gamma_power)
        self._torch_gen = torch.Generator().manual_seed(int(seed))

    @property
    def n_vars(self) -> int:
        return N_VARS

    def get_intervention_target(self) -> int:
        return FLOW_TOPO

    def get_outcome_var(self) -> int:
        return PRICE_TOPO

    def get_hidden_vars(self) -> list[int]:
        return [CORE_TOPO]

    def sample(
        self,
        generator: torch.Generator | None = None,
    ) -> tuple[ContinuousSCM, CoreReactionConfounder, InformationalCoupling]:
        """Return ``(scm, confounder, coupling)``.

        The parameter objects are returned alongside the SCM so callers
        (sweep script, diagnostics, tests) can read the episode's analytic
        slopes without reverse-engineering mechanism weights. The
        ``generator`` arg is kept for interface parity with the stock
        samplers; the internal generator is used throughout.
        """
        conf = sample_core_reaction_confounder(
            core_share_range=self.core_share_range,
            theta_range=self.theta_range,
            sigma_range=self.sigma_range,
            generator=self._torch_gen,
        )
        gamma_range = self.coupling_gamma_range
        if self.coupling_gamma_power != 1.0:
            lo, hi = gamma_range
            u = float(torch.rand((), generator=self._torch_gen))
            g = lo + (hi - lo) * u ** (1.0 / self.coupling_gamma_power)
            gamma_range = (g, g)
        coup = sample_informational_coupling(
            gamma_range=gamma_range,
            impact_lambda_range=self.impact_lambda_range,
            theta_range=self.theta_range,
            sigma_range=self.sigma_range,
            theta_price_range=self.theta_price_range,
            impact_lambda_log_range=self.impact_lambda_log_range,
            perm_prob=self.perm_prob,
            generator=self._torch_gen,
        )
        core, flow = conf.mechanisms(CORE_TOPO, FLOW_TOPO)
        price = coup.mechanism(CORE_TOPO, FLOW_TOPO, PRICE_TOPO)
        lo, hi = self.jump_rate_range
        rate = 0.0
        if hi > 0.0:
            # Draw only when the feature is on: the off-path must consume no
            # randomness, so seeded episode streams stay bit-identical to the
            # stock prior (reduction guarantee + artifact reproducibility).
            rate = lo + (hi - lo) * float(torch.rand((), generator=self._torch_gen))
        h_lo, h_hi = self.hawkes_rate_range
        h_rate = 0.0
        if h_hi > 0.0:
            # Enabled-path-only draw, same discipline as the jump knob.
            h_rate = h_lo + (h_hi - h_lo) * float(torch.rand((), generator=self._torch_gen))
        if h_rate > 0.0:
            # Rung 3: self-exciting sign-persistent flow arrivals (count-driven
            # metaorder timing per Gate C); mutually exclusive with the core
            # jump layer to keep attribution single-mechanism per episode.
            scm = HawkesFlowSCM(
                [core, flow, price], flow_var=FLOW_TOPO, mu=h_rate,
                alpha=self.hawkes_alpha, beta=self.hawkes_beta,
                jump_scale=self.hawkes_scale_mult * conf.sigma_flow,
                sign_persist=self.hawkes_persist,
            )
        elif rate > 0.0:
            # Compound-Poisson news arrivals in the hidden core (Phase-3 drift
            # layer); rate == 0 keeps the exact stock construction.
            scm = JumpDiffusionSCM(
                [core, flow, price], jump_var=CORE_TOPO, jump_rate=rate,
                jump_scale=self.jump_scale_mult * conf.sigma_core,
            )
        else:
            scm = ContinuousSCM([core, flow, price])
        return scm, conf, coup


class MarketDoTime(ContinuousExtendedPrior):
    """The market prior: model-ready episode generator over market SCMs.

    Drops into everything that consumes a :class:`ContinuousExtendedPrior`
    (the vendored trainer's dataloader included). Only the SCM-sampling hook
    is overridden; the hidden core flow rides the existing
    ``hidden_vars_topo`` masking machinery.

    Examples
    --------
    >>> prior = MarketDoTime(coupling_gamma=1.0, core_share=0.7, seed=0)
    >>> sample = prior.generate_sample()          # model-ready dict
    >>> prior.last_confounder.core_share
    0.7
    """

    def __init__(
        self,
        *,
        coupling_gamma=(0.0, 2.0),
        core_share=(0.3, 0.9),
        impact_lambda=(0.25, 1.0),
        coupling_gamma_power: float = 1.0,
        jump_rate=(0.0, 0.0),
        jump_scale_mult: float = 3.0,
        theta_price_range=None,
        impact_lambda_log_range=None,
        perm_prob: float = 0.0,
        hawkes_rate=(0.0, 0.0),
        seed: int = 0,
        **kwargs,
    ) -> None:
        # The parent builds a fixed-TSCM sampler cache from `tscm_structure`;
        # like RandomContinuousExtendedPrior we pass a harmless placeholder —
        # the market sampler below is what _sample_scm_context actually uses.
        kwargs.setdefault("tscm_structure", "rct_no_confounding")
        super().__init__(seed=seed, **kwargs)

        self.market_sampler = MarketContinuousSCMSampler(
            core_share=core_share,
            coupling_gamma=coupling_gamma,
            impact_lambda=impact_lambda,
            theta_range=kwargs.get("theta_range", (0.5, 2.0)),
            sigma_range=kwargs.get("sigma_range", (0.2, 0.6)),
            coupling_gamma_power=coupling_gamma_power,
            jump_rate=jump_rate,
            jump_scale_mult=jump_scale_mult,
            theta_price_range=theta_price_range,
            impact_lambda_log_range=impact_lambda_log_range,
            perm_prob=perm_prob,
            hawkes_rate=hawkes_rate,
            seed=seed,
        )
        # Parameter objects of the most recently sampled episode (diagnostics).
        self.last_confounder: CoreReactionConfounder | None = None
        self.last_coupling: InformationalCoupling | None = None

    @property
    def n_vars(self) -> int:
        return N_VARS

    @property
    def hidden_vars(self) -> list[int]:
        return [CORE_TOPO]

    # ------------------------------------------------------------------ hook

    def _sample_scm_context(self) -> _SampledSCMContext:
        scm, conf, coup = self.market_sampler.sample(generator=self._torch_gen)
        self.last_confounder = conf
        self.last_coupling = coup
        return _SampledSCMContext(
            scm=scm,
            n_vars=N_VARS,
            intervention_target_topo=FLOW_TOPO,
            outcome_var_topo=PRICE_TOPO,
            hidden_vars_topo=[CORE_TOPO],
        )
