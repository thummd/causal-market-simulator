"""Jump-diffusion extension of the market SCM (the prepared Phase-3 drift layer).

``JumpDiffusionSCM`` adds compound-Poisson jumps to one variable — the hidden
informational core ``U`` in the market instantiation, where jumps model news
arrivals. It follows the integration seam documented in ``prior/__init__``:

- ``_draw_noise`` pre-draws the jump randomness (per-step uniforms and jump
  sizes) alongside the Gaussian increments, so
  ``sample_counterfactual_pair`` — which draws noise once and simulates
  twice — shares identical jump realisations between the observational and
  interventional runs. Counterfactual exactness is preserved.
- ``_step`` thins the pre-drawn uniforms against ``rate * dt`` (the standard
  small-``dt`` Poisson approximation, exact in the fine-grid limit) and adds
  the pre-drawn jump size to the jump variable.

``jump_rate = 0`` is never constructed: the sampler returns a plain
``ContinuousSCM`` in that case, so the reduction test compares identical
classes (feature off == stock market prior).
"""

from __future__ import annotations

import torch

from dotime.continuous.continuous_scm import ContinuousSCM

__all__ = ["JumpDiffusionSCM"]


class JumpDiffusionSCM(ContinuousSCM):
    """A :class:`ContinuousSCM` whose ``jump_var`` receives compound-Poisson jumps.

    Parameters
    ----------
    mechanisms : list
        As for :class:`ContinuousSCM`.
    jump_var : int
        Topological index of the variable receiving jumps.
    jump_rate : float
        Expected number of jumps per unit time (must be positive; use the
        plain class for the no-jump case).
    jump_scale : float
        Standard deviation of the (zero-mean Gaussian) jump sizes, in the
        jump variable's state units.
    """

    def __init__(self, mechanisms, jump_var: int, jump_rate: float, jump_scale: float):
        super().__init__(mechanisms)
        if jump_rate <= 0:
            raise ValueError(f"jump_rate must be positive, got {jump_rate}")
        self.jump_var = int(jump_var)
        self.jump_rate = float(jump_rate)
        self.jump_scale = float(jump_scale)
        self._jump_u: torch.Tensor | None = None
        self._jump_z: torch.Tensor | None = None
        self._jump_cursor = 0

    def _draw_noise(self, num_steps, generator):
        noise = super()._draw_noise(num_steps, generator)
        # Pre-draw jump randomness with the same generator so the whole
        # trajectory randomness (Gaussian + jumps) is one shared draw.
        self._jump_u = torch.rand(num_steps, device=noise.device, generator=generator)
        self._jump_z = (
            torch.randn(num_steps, device=noise.device, generator=generator) * self.jump_scale
        )
        self._jump_cursor = 0
        return noise

    def simulate(self, *args, **kwargs):
        # Each simulate() pass (observational, then interventional, with the
        # same pre-drawn noise) must consume the jump sequence from the start.
        self._jump_cursor = 0
        return super().simulate(*args, **kwargs)

    def _step(self, x, dt, noise_row, intervention, t_next):
        x_new = super()._step(x, dt, noise_row, intervention, t_next)
        if self._jump_u is not None and self._jump_cursor < self._jump_u.shape[0]:
            i = self._jump_cursor
            if float(self._jump_u[i]) < self.jump_rate * float(dt):
                x_new[self.jump_var] = x_new[self.jump_var] + self._jump_z[i]
            self._jump_cursor = i + 1
        return x_new
