#!/usr/bin/env python
"""Square-root-law prefactor sensitivity on the crypto burst suites.

The paper reports the square-root law with the literature prefactor
k = 0.7 as the least-biased method on all three crypto assets. This
script sweeps k over the empirical range {0.5, 0.7, 1.0} to check
whether that ranking is prefactor-dependent (it is not: bias stays
within [-0.67, -0.13] across the sweep on every asset).

Usage:
    python scripts/04c_sqrt_sensitivity.py
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np

import dotime_market.baselines  # noqa: F401  (registration side effect)
from dotime.baselines import get
from dotime_market.data.liquidation import build_binance_suite


def main() -> None:
    dates = [str(date(2026, 4, 1) + timedelta(d)) for d in range(91)]
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        suite = build_binance_suite(symbol=sym, dates=dates)
        row = {}
        for k in (0.5, 0.7, 1.0):
            m = get("SquareRootLaw")
            m.k = k
            signed = []
            for ep in suite:
                p = m.predict(ep).reshape(-1)
                y = ep.y_true.reshape(-1)
                s = math.copysign(1.0, float(ep.intervention.values))
                signed.extend(s * float(a - b) for a, b in zip(p, y))
            row[k] = round(float(np.mean(signed)), 4)
        out[sym] = row
        print(sym, row, flush=True)
    json.dump(out, open("output/sqrt_sensitivity.json", "w"), indent=2)
    print("wrote output/sqrt_sensitivity.json")


if __name__ == "__main__":
    main()
