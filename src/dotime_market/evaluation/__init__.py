"""LAYER 1 — market diagnostics complementing dotime.evaluation.

Implemented: slopes.py (naive vs interventional slope protocol, shared by the
Phase-0/Phase-1 gap checks and the bias-vs-gamma sweep).
Planned: diffusivity.py (Bonart diffusivity-of-impact), sqrt_law.py
(impact-vs-Q/V log-log slope), calibration.py (counterfactual coverage/width).

Reuse from dotime.evaluation: evaluate(model, suite) — model FIRST — plus
Results, bootstrap_ci, compute_rmse/r2/mae/nmse, direction_accuracy.
BenchmarkSuite(meta, episodes) is constructible in-memory.
"""
