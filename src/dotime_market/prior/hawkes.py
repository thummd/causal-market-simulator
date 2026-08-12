"""Hawkes flow-timing extension — rung 3 of the Gate-B prior ladder.

``HawkesFlowSCM`` adds *self-exciting, sign-persistent* jumps to the
observable order flow ``A``: each jump raises the arrival intensity (which
then decays), and consecutive jumps keep their sign with high probability.
This produces the clustered, same-sign burst structure of real metaorder
child-order arrivals — the count-driven regime Gate C measured on
BTC/ETH/SOL — as a *timing* property of the flow itself, rather than a new
price mechanism.

Integration seam (mirrors :class:`.jumps.JumpDiffusionSCM`):

- ``_draw_noise`` pre-draws all jump randomness (thinning uniforms,
  sign-persistence uniforms, size normals) with the shared generator, so
  counterfactual pairs replay identical arrival sequences. The intensity
  state depends only on the jump history — never on the simulated variables —
  so the thinning decisions are identical in both arms and counterfactual
  exactness is preserved. (Do NOT make the intensity depend on ``A`` or
  ``Y`` without revisiting this argument.)
- ``_step`` advances the intensity decay, thins one pre-drawn uniform
  against ``intensity * dt``, and on arrival adds the signed jump to the
  flow variable.

``hawkes_rate = 0`` is never constructed: the sampler returns the plain
class in that case, so reduction tests compare identical classes.
"""

from __future__ import annotations

import math

import torch

from dotime.continuous.continuous_scm import ContinuousSCM

__all__ = ["HawkesFlowSCM"]


class HawkesFlowSCM(ContinuousSCM):
    """A :class:`ContinuousSCM` whose flow variable receives Hawkes jumps.

    Parameters
    ----------
    mechanisms : list
        As for :class:`ContinuousSCM`.
    flow_var : int
        Topological index of the observable flow variable.
    mu : float
        Baseline arrival intensity (jumps per unit time, must be positive).
    alpha : float
        Intensity increment per arrival (self-excitation).
    beta : float
        Exponential decay rate of the excitation. Stability requires a
        branching ratio ``alpha / beta < 1``.
    jump_scale : float
        Scale of the half-normal jump magnitudes, in flow units.
    sign_persist : float
        Probability that an arrival keeps the previous arrival's sign
        (in ``[0.5, 1)``); the complement flips it. High values give the
        same-sign run structure of metaorder child orders.

    Raises
    ------
    ValueError
        If ``mu`` or ``jump_scale`` is not positive, the branching ratio
        ``alpha / beta`` is not in ``[0, 1)``, or ``sign_persist`` is
        outside ``[0.5, 1)``.
    """

    def __init__(self, mechanisms, flow_var: int, mu: float, alpha: float,
                 beta: float, jump_scale: float, sign_persist: float = 0.9):
        super().__init__(mechanisms)
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if jump_scale <= 0:
            raise ValueError(f"jump_scale must be positive, got {jump_scale}")
        if not 0.0 <= alpha / beta < 1.0:
            raise ValueError(
                f"branching ratio alpha/beta must be in [0, 1), got {alpha / beta}"
            )
        if not 0.5 <= sign_persist < 1.0:
            raise ValueError(f"sign_persist must be in [0.5, 1), got {sign_persist}")
        self.flow_var = int(flow_var)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jump_scale = float(jump_scale)
        self.sign_persist = float(sign_persist)
        self._u_thin: torch.Tensor | None = None
        self._u_sign: torch.Tensor | None = None
        self._z_size: torch.Tensor | None = None
        self._cursor = 0
        self._excitation = 0.0
        self._sign = 1.0

    def _draw_noise(self, num_steps, generator):
        noise = super()._draw_noise(num_steps, generator)
        self._u_thin = torch.rand(num_steps, device=noise.device, generator=generator)
        self._u_sign = torch.rand(num_steps, device=noise.device, generator=generator)
        # Half-normal magnitudes: metaorder child orders push one way; the
        # direction comes from the persistent sign chain, not the size draw.
        self._z_size = torch.randn(
            num_steps, device=noise.device, generator=generator
        ).abs() * self.jump_scale
        self._cursor = 0
        return noise

    def simulate(self, *args, **kwargs):
        # Both passes of a counterfactual pair replay the same pre-drawn
        # sequence from the start, with the intensity/sign state reset.
        self._cursor = 0
        self._excitation = 0.0
        self._sign = 1.0
        return super().simulate(*args, **kwargs)

    def _step(self, x, dt, noise_row, intervention, t_next):
        x_new = super()._step(x, dt, noise_row, intervention, t_next)
        if self._u_thin is not None and self._cursor < self._u_thin.shape[0]:
            i = self._cursor
            self._excitation *= math.exp(-self.beta * float(dt))
            intensity = self.mu + self._excitation
            if float(self._u_thin[i]) < intensity * float(dt):
                if float(self._u_sign[i]) > self.sign_persist:
                    self._sign = -self._sign
                x_new[self.flow_var] = x_new[self.flow_var] + self._sign * float(
                    self._z_size[i]
                )
                self._excitation += self.alpha
            self._cursor = i + 1
        return x_new
