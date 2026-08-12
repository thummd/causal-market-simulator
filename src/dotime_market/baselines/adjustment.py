"""Oracle causal estimators for market suites.

The market structure has *no observable adjustment set* — the confounder U is
hidden, so back-door/IV estimators are infeasible from the data alone (that
is the paper's premise). The oracle here is therefore a *parameter* oracle:
it is handed the true mechanism parameters (impact_lambda, theta_price,
theta_flow — from ``Episode.metadata``, where the suite builder records them)
but NOT the realized noise, and predicts the conditional mean of the
counterfactual price by exact ODE rollout from the last observed pre-onset
state:

- inside the clamp window (``do(A = c)`` on ``[t0, t1)``)::

      E[Y(t)] = y0 * exp(-theta_Y * (t - t0)) +
                (lambda * c / theta_Y) * (1 - exp(-theta_Y * (t - t0)))

- after the window, the flow relaxes (``E[A(t)] = c * exp(-theta_A * (t - t1))``,
  the hidden core contributes zero mean) and the price integrates it::

      E[Y(t)] = y(t1) * e^{-theta_Y * d} +
                lambda * c * (e^{-theta_A * d} - e^{-theta_Y * d}) / (theta_Y - theta_A)

  (with the ``theta_Y == theta_A`` limit ``lambda * c * d * e^{-theta * d}``).

This is the best *feasible* causal estimator given the identification
knowledge the PFN is supposed to infer: unbiased for E[Y | do], with only the
irreducible post-onset noise as error. Its gap to the naive impact models is
the confounding bias; its gap to zero is the noise floor.

Registered as ``"param-oracle"``. Raises on episodes without market metadata
rather than guessing (mirrors the stock ``Oracle`` contract).
"""

from __future__ import annotations

import math

import torch

from dotime.baselines import register
from dotime.benchmarks import Episode

__all__ = ["ParamOracleBaseline"]

_REQUIRED_KEYS = ("impact_lambda", "theta_price", "theta_flow")


def _relaxation_integral(lam: float, c: float, th_y: float, th_a: float, d: float) -> float:
    """lambda * c * int_0^d e^{-theta_A s} e^{-theta_Y (d - s)} ds."""
    if abs(th_y - th_a) < 1e-8:
        return lam * c * d * math.exp(-th_y * d)
    return lam * c * (math.exp(-th_a * d) - math.exp(-th_y * d)) / (th_y - th_a)


@register("param-oracle")
class ParamOracleBaseline:
    """True-parameter conditional-mean rollout (see module docstring)."""

    name = "param-oracle"

    def predict(self, episode: Episode) -> torch.Tensor:
        md = episode.metadata
        missing = [k for k in _REQUIRED_KEYS if k not in md]
        if missing:
            raise RuntimeError(
                f"param-oracle requires market metadata {missing} "
                "(build suites with dotime_market.data.synthetic.build_market_suite)"
            )
        lam = float(md["impact_lambda"])
        th_y = float(md["theta_price"])
        th_a = float(md["theta_flow"])

        x = episode.x_obs
        T = x.shape[0]
        onset, end = min(episode.intervention.times), max(episode.intervention.times) + 1
        a_idx = episode.intervention.targets[0] if episode.intervention.targets else 0
        c = float(episode.intervention.values)
        win_len = float(end - onset)

        preds = []
        qt = episode.query_time.reshape(-1)
        for var, t_q in zip(episode.query_target.reshape(-1).tolist(), qt.tolist()):
            var = int(var)
            q_idx = int(round(t_q * (T - 1)))
            tau = float(max(q_idx - onset, 0))

            if var == a_idx:
                # E[A]: clamped inside the window, exponential relaxation after.
                if q_idx < end:
                    preds.append(c)
                else:
                    preds.append(c * math.exp(-th_a * (q_idx - end + 1)))
                continue

            y0 = float(x[onset - 1, var]) if onset > 0 else 0.0
            if tau <= win_len:
                y = y0 * math.exp(-th_y * tau) + (lam * c / th_y) * (1.0 - math.exp(-th_y * tau))
            else:
                y_end = (
                    y0 * math.exp(-th_y * win_len)
                    + (lam * c / th_y) * (1.0 - math.exp(-th_y * win_len))
                )
                d = tau - win_len
                y = y_end * math.exp(-th_y * d) + _relaxation_integral(lam, c, th_y, th_a, d)
            preds.append(y)
        return torch.tensor(preds, dtype=torch.float32)
