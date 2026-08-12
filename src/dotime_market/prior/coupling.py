"""Informational coupling gamma — the swept knob of the Phase-1 headline figure.

Market story
------------
The price (return) variable ``Y`` responds to two channels::

    dY = ( -theta_price * Y + impact_lambda * A + gamma * U ) dt + sigma_price dW_Y

- ``impact_lambda`` (``A -> Y``): *mechanical* impact of observable order flow —
  the true causal effect the PFN must estimate under ``do(A)``.
- ``gamma`` (``U -> Y``): *informational* coupling — the hidden core flow's
  signal reaches the price directly (information revelation), not only through
  its trading. Because ``U`` also drives ``A`` (see :mod:`.confounding`),
  ``gamma > 0`` confounds the naive flow-price regression.

``gamma = 0`` (or ``core_share = 0``) closes the back-door path and the naive
slope's confounding bias vanishes — both reductions are exact and tested.

Analytic slopes
---------------
For the joint linear system ``(U, A, Y)`` with the :class:`.confounding`
block's stationary moments ``Var(U)``, ``Cov(A, U)``, ``Var(A)``, the
stationary Lyapunov equation gives::

    Cov(Y, U) = (gamma * Var(U) + lambda * Cov(A, U)) / (theta_price + theta_core)
    Cov(Y, A) = (gamma * Cov(A, U) + lambda * Var(A) + w * Cov(Y, U))
                / (theta_price + theta_flow)

so the *naive* contemporaneous OLS slope of ``Y`` on ``A`` is
``Cov(Y, A) / Var(A)``, while the *interventional* steady-state slope under a
hard clamp ``do(A = a)`` is ``lambda / theta_price`` (``E[U] = 0``, so the
information channel averages out under the clamp).

Note the two slopes differ even without confounding: at ``gamma = 0`` the
naive slope is ``~ lambda / (theta_price + theta_flow)`` because ``Y``
low-pass-filters a fluctuating ``A`` — a filtering artifact, not bias. The
clean confounding quantity is therefore
:meth:`InformationalCoupling.confounding_bias`, the *excess* naive slope over
its ``gamma = 0`` value::

    bias(gamma) = gamma * (Cov(A, U) + w * Var(U) / (theta_price + theta_core))
                  / ((theta_price + theta_flow) * Var(A))

— exactly linear in ``gamma``, exactly zero when ``core_share = 0``
(``w = Cov(A, U) = 0``). This is the analytic curve behind the bias-vs-gamma
figure (``scripts/01_bias_vs_gamma.py``).
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import torch

from dotime.continuous import OUMechanism

from .confounding import CoreReactionConfounder

__all__ = ["InformationalCoupling", "IntegratorMechanism", "sample_informational_coupling"]


class IntegratorMechanism(OUMechanism):
    """An :class:`OUMechanism` that permits ``theta == 0`` (unit root).

    The stock mechanism rejects ``theta <= 0`` because a generic OU variable
    needs a stationary distribution. The unit-root price carve-out is exactly
    the case where we *want* no mean reversion: ``dY = (lambda A + gamma U) dt
    + sigma dW`` makes mechanical impact accrue linearly with metaorder
    duration instead of saturating at ``lambda / theta_Y`` — the failure mode
    the Gate-B benchmark localized. Simulation is safe: both the scalar and
    vectorized simulators use ``-theta * x`` linearly and never divide by
    ``theta`` (plain Euler-Maruyama, no exact-OU transition).

    Raises
    ------
    ValueError
        If ``theta`` is negative, ``sigma`` is not positive, or
        ``parent_weights`` does not match ``parents``.
    """

    def __post_init__(self) -> None:
        if self.theta < 0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if len(self.parents) != self.parent_weights.numel():
            raise ValueError(
                f"parent_weights length {self.parent_weights.numel()} does not match "
                f"number of parents {len(self.parents)}"
            )


@dataclass(frozen=True)
class InformationalCoupling:
    """Price mechanism parameters: mechanical impact + informational coupling.

    Parameters
    ----------
    gamma : float
        Informational coupling ``U -> Y``. The swept confounding knob;
        ``0`` closes the back-door path.
    impact_lambda : float
        Mechanical impact ``A -> Y`` — the true causal effect.
    theta_price, sigma_price : float
        Mean-reversion rate and diffusion of the price variable (both
        positive). ``impact_lambda / theta_price`` is the steady-state
        interventional response.
    """

    gamma: float
    impact_lambda: float
    theta_price: float = 1.0
    sigma_price: float = 0.4

    def __post_init__(self) -> None:
        # theta_price == 0 is the unit-root (permanent-impact) carve-out:
        # impact accrues without saturating. Stationary analytics below
        # remain finite (they divide by theta_price + theta_core/flow) but
        # describe a nonstationary Y only formally — use them for theta > 0.
        if self.theta_price < 0:
            raise ValueError(f"theta_price must be non-negative, got {self.theta_price}")
        if self.sigma_price <= 0:
            raise ValueError(f"sigma_price must be positive, got {self.sigma_price}")

    # ------------------------------------------------------------- mechanism

    def mechanism(self, core_idx: int, flow_idx: int, price_idx: int) -> OUMechanism:
        """Build the price :class:`OUMechanism` for a :class:`ContinuousSCM`.

        Zero-weight parents are dropped, so ``gamma = 0`` yields a price
        with no ``U`` parent at all (exact graph-level reduction) and
        ``gamma = impact_lambda = 0`` yields a parentless stock root OU.
        """
        if not 0 <= core_idx < flow_idx < price_idx:
            raise ValueError(
                f"need 0 <= core_idx < flow_idx < price_idx (topological order), got "
                f"core_idx={core_idx}, flow_idx={flow_idx}, price_idx={price_idx}"
            )
        parents: list[int] = []
        weights: list[float] = []
        if self.gamma != 0.0:
            parents.append(core_idx)
            weights.append(self.gamma)
        if self.impact_lambda != 0.0:
            parents.append(flow_idx)
            weights.append(self.impact_lambda)
        # IntegratorMechanism is required only for the theta == 0 carve-out;
        # using the stock class otherwise keeps the reduction surface minimal.
        cls = IntegratorMechanism if self.theta_price == 0.0 else OUMechanism
        if not parents:
            return cls(theta=self.theta_price, sigma=self.sigma_price)
        return cls(
            theta=self.theta_price,
            sigma=self.sigma_price,
            parent_weights=torch.tensor(weights, dtype=torch.float32),
            parents=tuple(parents),
        )

    def mechanisms(self, core_idx: int, flow_idx: int, price_idx: int) -> tuple:
        """:class:`~dotime_market.prior.interfaces.MechanismFactory`
        conformance — the price mechanism as a 1-tuple."""
        return (self.mechanism(core_idx, flow_idx, price_idx),)

    # ------------------------------------------ analytic slopes (SDE-exact)

    def stationary_price_core_cov(self, conf: CoreReactionConfounder) -> float:
        """``Cov(Y, U)`` at stationarity."""
        return (self.gamma * conf.stationary_core_var()
                + self.impact_lambda * conf.stationary_flow_core_cov()) / (
            self.theta_price + conf.theta_core
        )

    def stationary_price_flow_cov(self, conf: CoreReactionConfounder) -> float:
        """``Cov(Y, A)`` at stationarity."""
        return (
            self.gamma * conf.stationary_flow_core_cov()
            + self.impact_lambda * conf.stationary_flow_var()
            + conf.core_weight * self.stationary_price_core_cov(conf)
        ) / (self.theta_price + conf.theta_flow)

    def naive_slope(self, conf: CoreReactionConfounder) -> float:
        """Contemporaneous OLS slope of ``Y`` on ``A`` — what a predictive
        model fits on observational data."""
        return self.stationary_price_flow_cov(conf) / conf.stationary_flow_var()

    def do_slope(self) -> float:
        """Steady-state interventional slope ``dE[Y | do(A = a)] / da``.

        Raises
        ------
        ValueError
            If ``theta_price == 0``: a unit-root price has no steady state —
            the interventional response is ``impact_lambda * a * T``,
            accruing with duration ``T`` instead of converging.
        """
        if self.theta_price == 0.0:
            raise ValueError(
                "do_slope is undefined for a unit-root price (theta_price == 0): "
                "the interventional response accrues as impact_lambda * a * T"
            )
        return self.impact_lambda / self.theta_price

    def confounding_bias(self, conf: CoreReactionConfounder) -> float:
        """Excess naive slope over its ``gamma = 0`` value — the pure
        confounding contribution (linear in ``gamma``, zero at
        ``core_share = 0``)."""
        numer = self.gamma * (
            conf.stationary_flow_core_cov()
            + conf.core_weight * conf.stationary_core_var() / (self.theta_price + conf.theta_core)
        )
        return numer / ((self.theta_price + conf.theta_flow) * conf.stationary_flow_var())


def sample_informational_coupling(
    gamma_range: tuple = (0.0, 2.0),
    impact_lambda_range: tuple = (0.25, 1.0),
    theta_range: tuple = (0.5, 2.0),
    sigma_range: tuple = (0.2, 0.6),
    theta_price_range: tuple | None = None,
    impact_lambda_log_range: tuple | None = None,
    perm_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> InformationalCoupling:
    """Draw an :class:`InformationalCoupling` from priors on its parameters.

    All draws are uniform over their ranges; degenerate ranges (``lo == hi``)
    are allowed so sweep configs can pin ``gamma`` per grid point.
    ``theta_range`` / ``sigma_range`` match the hyperpriors used for the
    core/flow processes in :func:`.confounding.sample_core_reaction_confounder`.

    Args
    ----
    perm_prob : float
        Probability of the unit-root carve-out: with this probability the
        episode's ``theta_price`` is set to exactly ``0`` (permanent,
        duration-accruing impact; see :class:`IntegratorMechanism`). The
        extra Bernoulli draw is consumed ONLY when ``perm_prob > 0``, so the
        default keeps seeded episode streams bit-identical to the legacy
        prior (reduction guarantee).

    Raises
    ------
    ValueError
        If any range is invalid or ``perm_prob`` is outside ``[0, 1]``.
    """
    if gamma_range[0] > gamma_range[1]:
        raise ValueError(f"invalid gamma_range: {gamma_range}")
    if impact_lambda_range[0] > impact_lambda_range[1]:
        raise ValueError(f"invalid impact_lambda_range: {impact_lambda_range}")
    if theta_range[0] <= 0 or theta_range[1] < theta_range[0]:
        raise ValueError(f"invalid theta_range: {theta_range}")
    if sigma_range[0] <= 0 or sigma_range[1] < sigma_range[0]:
        raise ValueError(f"invalid sigma_range: {sigma_range}")
    if not 0.0 <= perm_prob <= 1.0:
        raise ValueError(f"perm_prob must be in [0, 1], got {perm_prob}")

    u = torch.empty(4)
    u.uniform_(0.0, 1.0, generator=generator)

    def _unif(i: int, lo: float, hi: float) -> float:
        return float(lo + u[i] * (hi - lo))

    if impact_lambda_log_range is None:
        lam = _unif(1, *impact_lambda_range)
    else:
        # Scale-augmentation rung (i): log-uniform lambda over a wide range,
        # reusing draw slot u[1] (the rung-1 pattern) so None is bit-exact.
        # Rationale: per-domain k mismatch is an impact-to-noise-ratio error;
        # a 10-50x lambda span forces the posterior to infer the episode's
        # response scale from pre-window flow-price co-movement.
        llo, lhi = impact_lambda_log_range
        if llo <= 0 or lhi < llo:
            raise ValueError(f"invalid impact_lambda_log_range: {impact_lambda_log_range}")
        lam = float(llo * math.exp(float(u[1]) * math.log(lhi / llo)))

    if theta_price_range is None:
        theta_price = _unif(2, *theta_range)
    else:
        # Slow-price-mean-reversion regime (duration-accruing impact): a
        # log-uniform draw over the given range reuses the SAME u[2], so the
        # RNG stream and every other parameter are unchanged for a given
        # seed; theta_price_range=None reproduces the legacy prior exactly.
        lo, hi = theta_price_range
        if lo <= 0 or hi < lo:
            raise ValueError(f"invalid theta_price_range: {theta_price_range}")
        theta_price = float(lo * math.exp(float(u[2]) * math.log(hi / lo)))
    if perm_prob > 0.0:
        # Unit-root carve-out. The Bernoulli draw sits AFTER the u-vector so
        # every continuous parameter reuses its usual slot; the draw itself
        # happens only on the enabled path (perm_prob == 0 consumes nothing).
        v = torch.empty(1)
        v.uniform_(0.0, 1.0, generator=generator)
        if float(v[0]) < perm_prob:
            theta_price = 0.0
    return InformationalCoupling(
        gamma=_unif(0, *gamma_range),
        impact_lambda=lam,
        theta_price=theta_price,
        sigma_price=_unif(3, *sigma_range),
    )
