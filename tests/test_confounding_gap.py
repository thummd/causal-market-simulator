"""Phase 0 CI gate: the PoC precondition.

A measurable |E[Y|A] - E[Y|do(A)]| gap must exist — first on stock dotime
(this test), later on the market prior with an explicit, per-episode-controllable
knob (test to be extended in Phase 1 with MarketDoTime / coupling_gamma).

Kept small so it runs in CI; the reference full-size sweep lives in
scripts/00_stock_gap_check.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

stock_gap_check = __import__("00_stock_gap_check")

# The gate: at the default prior setting the average confounding bias of the
# naive estimator must exceed the interventional slope by a clear margin.
THRESHOLD = 0.03


def test_measurable_confounding_gap_stock_dotime():
    results = stock_gap_check.run_sweep(weight_scales=(0.5,), n_scms=20)
    r = results[0.5]
    assert r["mean_gap"] > THRESHOLD, (
        f"confounding gap {r['mean_gap']:.4f} <= {THRESHOLD}: "
        "the PoC has nothing to show — investigate before building Layer 2"
    )
    # The true effect must stay near zero (no A->Y edge in unobserved_confounder).
    assert r["mean_abs_do"] < 0.02
