"""LAYER 3 — real data -> dotime Episode adapters.

Implemented: synthetic.py (MarketDoTime -> in-memory suites),
abides_builder.py (metaorder injection + seed-replay counterfactual),
binance.py (data.binance.vision aggTrades loader; liquidationSnapshot was
REMOVED from the public dumps — see module docstring), liquidation.py
(detected flow-burst episodes, the liquidation-proxy route; factual-outcome
task, no counterfactual on real data), databento.py (Nasdaq ITCH / CME DBN
files -> the same per-trade frame + parquet caches; closing-imbalance and
statistics loaders), auction.py (Nasdaq closing-cross NOII imbalance as an
announced soft intervention), fomc.py (scheduled 14:00 ET windows on ES,
FOMC days vs control Wednesdays). All real-data tiers share one episode
encoder, liquidation.make_flow_episode, and one bar builder,
binance.bin_signed_trades.

Pattern to copy: the PK adapters in the cited construction's source repository
(continuous-time-causal-pfn/dotime/data/pk_pd/*_adapter.py).
"""
