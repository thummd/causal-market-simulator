"""Shared real-data transfer evaluation: paired do-head deltas on factual suites.

Every real-data tier (Binance bursts, Nasdaq bursts, closing auctions, FOMC
windows) asks the same question: does the do-information improve factual
prediction? The readout is the paired delta between the causal PFN and its
do-ablated twin on identical episodes, with bootstrap CIs, plus every
registered baseline scored on the same episodes. This module holds the
scoring and table code so each script is only data assembly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime.evaluation import bootstrap_ci

__all__ = ["score", "build_models", "paired_results", "vol_split"]

_DEFAULT_BASELINES = ("Mean", "AR1", "AlmgrenChriss", "OWPropagator",
                      "ReturnRegression", "SquareRootLaw")


def score(model, suite) -> dict:
    """Score one model on a factual suite.

    Args:
        model: Object with ``predict(episode) -> tensor``.
        suite: Iterable of dotime Episodes with ``y_true`` and a scalar
            soft-intervention value whose sign orients the error.

    Returns:
        Dict with ``rmse``, ``mae``, ``bias`` (error signed by the
        intervention direction, so under-reaction to a buy and to a sell
        both count as negative), ``n``, and the per-query ``signed_errors``
        needed for paired bootstraps.
    """
    preds, targets, signs = [], [], []
    for ep in suite:
        p = model.predict(ep).reshape(-1)
        y = ep.y_true.reshape(-1)
        c = float(ep.intervention.values)
        preds.append(p)
        targets.append(y)
        signs.extend([math.copysign(1.0, c)] * y.numel())
    pred = torch.cat(preds).double()
    target = torch.cat(targets).double()
    sign = torch.tensor(signs, dtype=torch.float64)
    err = pred - target
    signed = sign * err
    return {
        "rmse": float(torch.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(signed.mean()),
        "n": int(err.numel()),
        "signed_errors": [float(v) for v in signed],
    }


def build_models(checkpoint: str, ablated_checkpoint: str, device: str = "cpu",
                 extra: tuple[str, ...] = ()) -> dict:
    """Instantiate the standard method set for a transfer table.

    Args:
        checkpoint: Path to the causal market PFN checkpoint.
        ablated_checkpoint: Path to the do-ablated twin checkpoint.
        device: Torch device for the PFNs.
        extra: Additional registered baseline names scored on the same episodes.

    Returns:
        Ordered dict name -> model; the ablated twin is last.
    """
    models = {name: get(name) for name in _DEFAULT_BASELINES}
    models["market-dotpfn"] = get("market-dotpfn", checkpoint=checkpoint, device=device)
    for name in extra:
        models[name] = get(name)
    models["ablated-dotpfn"] = get("ablated-dotpfn", checkpoint=ablated_checkpoint,
                                   device=device)
    return models


def paired_results(models: dict, suite, ref: str = "ablated-dotpfn",
                   bootstrap: int = 10000, verbose: bool = True) -> dict:
    """Score every model and attach the paired delta vs ``ref`` with a CI.

    Args:
        models: name -> model, as from :func:`build_models`.
        suite: Factual suite (all models see identical episodes).
        ref: Name of the reference method the deltas are taken against.
        bootstrap: Bootstrap resamples for the delta CI.
        verbose: Print the table.

    Returns:
        Dict name -> score dict (with ``delta_vs_ablated`` / ``delta_ci`` for
        every non-reference method).
    """
    results = {name: score(model, suite) for name, model in models.items()}
    if verbose:
        print(f"{'method':>18s}  {'rmse':>8s}  {'mae':>8s}  {'bias':>8s}  "
              f"{'d_vs_ablated':>12s}  {'95% CI':>20s}")
    for name, s in results.items():
        if name != ref:
            deltas = [a - b for a, b in zip(s["signed_errors"], results[ref]["signed_errors"])]
            dm, _, dlo, dhi = bootstrap_ci(deltas, n=bootstrap)
            s.update({"delta_vs_ablated": dm, "delta_ci": [dlo, dhi]})
            d_str = f"{dm:12.4f}  [{dlo:8.4f}, {dhi:8.4f}]"
        else:
            d_str = f"{'--':>12s}"
        if verbose:
            print(f"{name:>18s}  {s['rmse']:8.4f}  {s['mae']:8.4f}  {s['bias']:8.4f}  {d_str}",
                  flush=True)
    return results


def vol_split(results: dict, suite, ref: str = "ablated-dotpfn", method: str = "market-dotpfn",
              bootstrap: int = 10000, verbose: bool = True) -> dict:
    """Calm-vs-stressed robustness split of the paired delta.

    Splits episodes at the median pre-onset price std (the episode's own
    standardization scale) and reports the do-head delta in each half.

    Args:
        results: Output of :func:`paired_results`.
        suite: The same suite the results were computed on.
        ref: Reference method name.
        method: Method whose delta is split.
        bootstrap: Bootstrap resamples.
        verbose: Print the split.

    Returns:
        Dict with the median scale and per-regime ``n``, delta, and CI.
    """
    scales = np.array([float(ep.metadata["price_scale"]) for ep in suite])
    n_q = [int(ep.y_true.numel()) for ep in suite]
    scales = np.repeat(scales, n_q)  # one entry per query, aligned with signed_errors
    median = float(np.median(scales))
    stressed = scales > median
    deltas = np.array(results[method]["signed_errors"]) - np.array(results[ref]["signed_errors"])
    out = {"median_price_scale_bps": median, "regimes": {}}
    if verbose:
        print(f"\nvol split (median pre-onset sigma = {median:.2f} bps):")
    for regime, mask in (("calm", ~stressed), ("stressed", stressed)):
        if mask.sum() == 0:
            continue
        dm, _, dlo, dhi = bootstrap_ci([float(d) for d in deltas[mask]], n=bootstrap)
        out["regimes"][regime] = {"n": int(mask.sum()), "delta_vs_ablated": dm,
                                  "delta_ci": [dlo, dhi]}
        if verbose:
            print(f"  {regime:9s} n={int(mask.sum()):4d}  do-head delta {dm:+.3f}  "
                  f"[{dlo:+.3f}, {dhi:+.3f}]", flush=True)
    return out
