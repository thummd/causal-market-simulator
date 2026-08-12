"""Naive predictive impact models: Almgren-Chriss and an OW-style propagator.

These are the *foils* of the PoC: standard market-impact models fitted on
observational data. Both estimate the flow->price response by regression on
the pre-intervention trajectory, which conflates the mechanical impact
(lambda) with the informational channel (gamma) — under core/reaction
confounding their fitted coefficients absorb the back-door bias, and their
counterfactual predictions are wrong *by construction*. That bias, growing
with gamma, is the headline figure.

Both conform to the ``dotime.baselines.Baseline`` protocol and self-register;
importing :mod:`dotime_market.baselines` is enough to make them available to
``dotime.evaluation.evaluate``.

Conventions (match ``data/synthetic.py`` suites): regular ``dt = 1`` schedule
(query index = ``round(query_time * (T - 1))``), hard interventions with a
scalar clamp value, treatment index from ``intervention.targets[0]``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from dotime.baselines import register
from dotime.benchmarks import Episode

__all__ = [
    "AlmgrenChrissBaseline",
    "OWPropagatorBaseline",
    "HingeImpactBaseline",
    "ReturnRegressionBaseline",
    "SquareRootLawBaseline",
]

_MIN_FIT = 8  # minimum pre-onset observations for a regression fit


def _onset_end(episode: Episode) -> tuple[int, int]:
    times = episode.intervention.times
    if not times:
        t = episode.x_obs.shape[0]
        return t, t
    return min(times), max(times) + 1


def _query_indices(episode: Episode) -> list[int]:
    T = episode.x_obs.shape[0]
    qt = episode.query_time.reshape(-1)
    return [int(round(float(t) * (T - 1))) for t in qt]


def _clamp_value(episode: Episode) -> float:
    v = episode.intervention.values
    return float(v) if isinstance(v, (int, float)) else float(torch.as_tensor(v).mean())


@register("AlmgrenChriss")
class AlmgrenChrissBaseline:
    """Almgren-Chriss permanent impact, naively fitted (steady-state form).

    Fit: contemporaneous OLS of price level on flow over the pre-onset
    observational data, ``Y_t ~ beta * A_t`` — the standard empirical
    impact regression (price response per unit of flow). This is exactly
    the analytic ``naive_slope`` of :mod:`dotime_market.prior.coupling`,
    so under confounding ``beta`` absorbs the informational channel and
    the prediction is biased by precisely ``confounding_bias * c``.

    Predict: the permanent-impact shift applies from onset onward and
    persists after the window (the AC permanent component has no
    resilience): ``Y(q) = mean(Y_pre) + beta * c`` for ``q >= onset``.

    (The dynamic accumulate-per-executed-step AC form explodes on
    mean-reverting prices over benchmark-length windows — model-class
    error would swamp the confounding signal this foil exists to show.)
    """

    name = "AlmgrenChriss"

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()
        onset, end = _onset_end(episode)
        a_idx = episode.intervention.targets[0] if episode.intervention.targets else 0
        c = _clamp_value(episode)

        preds = []
        for var, q_idx in zip(episode.query_target.reshape(-1).tolist(), _query_indices(episode)):
            var = int(var)
            if var == a_idx:
                # The treatment itself: clamped inside the window, else naive mean.
                preds.append(c if onset <= q_idx < end else float(x[:onset, var].mean()))
                continue
            pre_y = x[:onset, var]
            pre_a = x[:onset, a_idx]
            if onset < _MIN_FIT:
                preds.append(float(x[:, var].mean()))
                continue
            a_c = pre_a - pre_a.mean()
            beta = float((a_c * (pre_y - pre_y.mean())).sum() / max((a_c * a_c).sum(), 1e-8))
            preds.append(float(pre_y.mean()) + beta * c)
        return torch.tensor(preds, dtype=torch.float32)


@register("OWPropagator")
class OWPropagatorBaseline:
    """Obizhaeva-Wang-style resilient propagator, naively fitted.

    Fit: OLS of the one-step price change on price level and flow over the
    pre-onset data, ``dY_t ~ a0 + a * Y_t + b * A_t`` (``a < 0`` is the
    fitted resilience / mean-reversion; ``b`` the impact coefficient —
    confounded like AC's).

    Predict: deterministic recursion ``Y_{t+1} = Y_t + a0 + a Y_t + b A_t``
    from the last pre-onset price, with ``A_t = c`` inside the window and
    ``A_t = 0`` after it — impact builds toward ``-(a0 + b c) / a`` and
    decays post-window (the resilient-impact shape AC lacks).
    """

    name = "OWPropagator"

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()
        onset, end = _onset_end(episode)
        a_idx = episode.intervention.targets[0] if episode.intervention.targets else 0
        c = _clamp_value(episode)

        preds = []
        for var, q_idx in zip(episode.query_target.reshape(-1).tolist(), _query_indices(episode)):
            var = int(var)
            if var == a_idx:
                preds.append(c if onset <= q_idx < end else float(x[:onset, var].mean()))
                continue
            pre_y = x[:onset, var]
            pre_a = x[:onset, a_idx]
            if onset < _MIN_FIT:
                preds.append(float(x[:, var].mean()))
                continue
            dy = pre_y[1:] - pre_y[:-1]
            design = np.column_stack([np.ones(onset - 1), pre_y[:-1], pre_a[:-1]])
            gram = design.T @ design + 1e-6 * np.eye(3)
            a0, a, b = np.linalg.solve(gram, design.T @ dy)
            a = float(np.clip(a, -1.9, -1e-3))  # keep the recursion stable & mean-reverting
            y = float(pre_y[-1])
            for t in range(onset, q_idx + 1):
                flow = c if t < end else 0.0
                y = y + a0 + a * y + b * flow
            preds.append(float(y))
        return torch.tensor(preds, dtype=torch.float32)


@register("ReturnRegression")
class ReturnRegressionBaseline:
    """Return-space impact regression at matched horizon.

    Fit: OLS of the ``h``-step price return on total flow over the same
    ``h`` steps, using overlapping pre-onset windows with ``h`` matched to
    the onset-to-query horizon — the practitioner-standard
    returns-on-signed-flow regression at the horizon actually queried.
    Mean reversion over the horizon is absorbed into the fitted
    coefficient (no per-step accumulation blow-up), and working in return
    space removes the stationary-levels filtering offset that the AC
    levels regression carries at gamma = 0; the informational back-door
    bias remains (returns still co-move with informed flow), so this foil
    isolates the confounding channel from the filtering artifact.

    Predict: ``Y(q) = Y(onset-1) + beta_h * c * n_exec`` with
    ``n_exec = min(q, end) - onset + 1`` executed steps adding ``c`` flow
    each. Falls back to the one-step fit when the pre-onset span is too
    short for the matched horizon.
    """

    name = "ReturnRegression"

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()
        onset, end = _onset_end(episode)
        a_idx = episode.intervention.targets[0] if episode.intervention.targets else 0
        c = _clamp_value(episode)

        preds = []
        for var, q_idx in zip(episode.query_target.reshape(-1).tolist(), _query_indices(episode)):
            var = int(var)
            if var == a_idx:
                preds.append(c if onset <= q_idx < end else float(x[:onset, var].mean()))
                continue
            if onset < _MIN_FIT:
                preds.append(float(x[:, var].mean()))
                continue
            pre_y = x[:onset, var]
            pre_a = x[:onset, a_idx]
            h = max(q_idx - onset + 1, 1)
            while h > 1 and onset - h < _MIN_FIT:
                h = h // 2  # shrink toward the one-step fit if pre-onset span is short
            starts = np.arange(0, onset - h)
            ret_h = pre_y[starts + h] - pre_y[starts]
            flow_h = np.array([pre_a[t : t + h].sum() for t in starts])
            beta = float((flow_h * ret_h).sum() / max((flow_h * flow_h).sum(), 1e-8))
            n_exec = max(min(q_idx, end - 1) - onset + 1, 0)
            preds.append(float(pre_y[-1]) + beta * c * n_exec)
        return torch.tensor(preds, dtype=torch.float32)


@register("SquareRootLaw")
class SquareRootLawBaseline:
    """Square-root impact law, the empirical metaorder benchmark.

    ``dY = k * sigma_pre * sign(c) * sqrt(|Q| / V)`` with metaorder size
    ``|Q| = |c| * executed steps``, volume proxy ``V = mean |A_pre|`` per
    step times the same duration (so the ratio reduces to participation
    ``|c| / mean|A_pre|``), pre-onset return volatility ``sigma_pre``,
    and the literature constant ``k = 0.7`` (the 0.5–1 range of the
    empirical metaorder studies). Zero fitted parameters — a pure
    reference point rather than an episode-fitted model; it carries no
    confounding bias by construction, but also no episode adaptivity.
    """

    name = "SquareRootLaw"
    k = 0.7

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()
        onset, end = _onset_end(episode)
        a_idx = episode.intervention.targets[0] if episode.intervention.targets else 0
        c = _clamp_value(episode)

        preds = []
        for var, q_idx in zip(episode.query_target.reshape(-1).tolist(), _query_indices(episode)):
            var = int(var)
            if var == a_idx:
                preds.append(c if onset <= q_idx < end else float(x[:onset, var].mean()))
                continue
            if onset < _MIN_FIT:
                preds.append(float(x[:, var].mean()))
                continue
            pre_y = x[:onset, var]
            sigma_pre = float(np.std(pre_y[1:] - pre_y[:-1]))
            v_per_step = float(np.abs(x[:onset, a_idx]).mean())
            participation = abs(c) / max(v_per_step, 1e-8)
            impact = self.k * sigma_pre * math.copysign(1.0, c) * math.sqrt(participation)
            preds.append(float(pre_y[-1]) + impact)
        return torch.tensor(preds, dtype=torch.float32)


@register("HingeImpact")
class HingeImpactBaseline:
    """Zero-then-linear ("hinge") impact response, after the auction evidence
    of Salek, Challet & Muni Toke (impact zero below a size threshold, then
    linear). This is a per-episode fitted *functional form* transplanted to
    continuous trading --- NOT their auction framework, which requires
    order-book state our episodes do not carry.

    Fit: on pre-onset matched-horizon windows (as ReturnRegression), regress
    ``ret_h ~ beta * s_tau(flow_h)`` with the hinge
    ``s_tau(f) = sign(f) * max(|f| - tau, 0)``, grid-searching ``tau`` over
    quantiles of ``|flow_h|`` (including 0, the pure-linear case) and solving
    ``beta`` by OLS at each ``tau``; keep the SSE-minimizing pair.

    Predict: ``Y(q) = Y(onset-1) + beta * s_tau(c * n_exec)`` --- zero
    predicted impact for interventions below the fitted threshold, linear
    beyond it.
    """

    name = "HingeImpact"
    _TAU_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9)

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()
        onset, end = _onset_end(episode)
        a_idx = episode.intervention.targets[0] if episode.intervention.targets else 0
        c = _clamp_value(episode)

        preds = []
        for var, q_idx in zip(episode.query_target.reshape(-1).tolist(), _query_indices(episode)):
            var = int(var)
            if var == a_idx:
                preds.append(c if onset <= q_idx < end else float(x[:onset, var].mean()))
                continue
            if onset < _MIN_FIT:
                preds.append(float(x[:, var].mean()))
                continue
            pre_y = x[:onset, var]
            pre_a = x[:onset, a_idx]
            h = max(q_idx - onset + 1, 1)
            while h > 1 and onset - h < _MIN_FIT:
                h = h // 2
            starts = np.arange(0, onset - h)
            ret_h = pre_y[starts + h] - pre_y[starts]
            flow_h = np.array([pre_a[t : t + h].sum() for t in starts])

            best = (0.0, 0.0, np.inf)  # (tau, beta, sse)
            abs_f = np.abs(flow_h)
            for q in self._TAU_QUANTILES:
                tau = 0.0 if q == 0.0 else float(np.quantile(abs_f, q))
                s = np.sign(flow_h) * np.maximum(abs_f - tau, 0.0)
                denom = float((s * s).sum())
                beta = float((s * ret_h).sum() / denom) if denom > 1e-12 else 0.0
                sse = float(np.square(ret_h - beta * s).sum())
                if sse < best[2]:
                    best = (tau, beta, sse)
            tau, beta, _ = best
            n_exec = max(min(q_idx, end - 1) - onset + 1, 0)
            f_int = c * n_exec
            s_int = math.copysign(1.0, f_int) * max(abs(f_int) - tau, 0.0) if f_int else 0.0
            preds.append(float(pre_y[-1]) + beta * s_int)
        return torch.tensor(preds, dtype=torch.float32)
