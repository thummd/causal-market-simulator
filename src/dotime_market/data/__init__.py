"""LAYER 3 — real data -> dotime Episode adapters.

Implemented: synthetic.py (MarketDoTime -> in-memory suites),
abides_builder.py (metaorder injection + seed-replay counterfactual),
binance.py (data.binance.vision aggTrades loader; liquidationSnapshot was
REMOVED from the public dumps — see module docstring), liquidation.py
(detected flow-burst episodes, the liquidation-proxy route; factual-outcome
task, no counterfactual on real data).

Pattern to copy: the PK adapters in the cited construction's source repository
(continuous-time-causal-pfn/dotime/data/pk_pd/*_adapter.py).
"""
