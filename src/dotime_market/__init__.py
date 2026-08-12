"""dotime-market: causal market simulator built on the `dotime` package.

ICAIF '26 paper repo. Layers (see docs/market_extension_architecture.md):

- ``prior``      — market-specific prior mechanisms (extends dotime.continuous internals)
- ``baselines``  — oracle adjustment + naive impact baselines (dotime.baselines registry)
- ``evaluation`` — market diagnostics complementing dotime.evaluation
- ``data``       — real data (ABIDES, Binance) -> dotime Episode adapters
- ``models``     — adapted continuous-time causal PFN + trainer (from the cited construction's source;
  not shipped in released dotime 0.1.1) and market training entry points
"""

from __future__ import annotations

__version__ = "0.0.1"
