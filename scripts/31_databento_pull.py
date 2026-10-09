"""Cost-capped Databento pull for the traditional-finance transfer tiers.

Downloads the three data tiers priced on 2026-09-15 (see HANDOVER.md) into
``data/raw/databento/`` as DBN files:

* **Equity burst transfer** — XNAS.ITCH ``trades`` (aggressor-signed) and
  ``bbo-1s`` for a 10-name Nasdaq panel, June to August 2026, one file per month.
  Feeds the signed-flow burst detector in :mod:`dotime_market.data.liquidation`.
* **Closing-auction intervention** — XNAS.ITCH ``imbalance`` (closing-cross
  net order imbalance, disseminated 3:50 to 4:00 pm ET) for a 50-name panel over
  12 months, plus ``statistics`` for all symbols over three months.
* **FOMC case study** — GLBX.MDP3 ``trades`` for the ES parent on five 2026
  FOMC announcement days and five control Wednesdays one week earlier.

Every request is priced with the free ``metadata.get_cost`` endpoint and the
sum must stay under ``--max-cost`` before any download starts. Files that
already exist are skipped, so a re-run never bills the same range twice.

Usage::

    .venv/bin/python scripts/31_databento_pull.py --dry-run   # price only
    .venv/bin/python scripts/31_databento_pull.py             # download

The API key is read from ``DATABENTO_API_KEY`` (loaded from ``.env``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import databento as db
from dotenv import load_dotenv

# FOMC statement days and matched control Wednesdays live with the episode
# builder so the pull and the analysis can never drift apart.
from dotime_market.data.fomc import CONTROL_DAYS, FOMC_DAYS

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "raw" / "databento"

# Ten liquid Nasdaq names: enough cross-sectional variety for burst detection
# while keeping the trades pull near 5 GB (the cost driver is bytes, not rate).
EQ10 = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "INTC", "QQQ"]
EQ50 = EQ10 + [
    "SPY", "AVGO", "NFLX", "COST", "PEP", "CSCO", "ADBE", "TXN", "QCOM", "AMGN",
    "CMCSA", "HON", "INTU", "AMAT", "SBUX", "BKNG", "ISRG", "MDLZ", "GILD", "ADI",
    "ADP", "VRTX", "REGN", "LRCX", "PANW", "MU", "KLAC", "SNPS", "CDNS", "MELI",
    "MAR", "CSX", "ORLY", "ASML", "CTAS", "PYPL", "MRVL", "FTNT", "ABNB", "PDD",
]
MONTHS = [("2026-06-01", "2026-07-01"), ("2026-07-01", "2026-08-01"), ("2026-08-01", "2026-09-01")]


@dataclass(frozen=True)
class Pull:
    """One Databento historical request and the file it lands in.

    Args:
        name: Output stem under ``data/raw/databento/``.
        dataset: Databento dataset code (e.g. ``XNAS.ITCH``).
        schema: Record schema (``trades``, ``bbo-1s``, ``imbalance``, ...).
        symbols: Raw symbols, parent symbols, or ``ALL_SYMBOLS``.
        stype_in: Input symbology (``raw_symbol`` or ``parent``).
        start: Inclusive start (ISO date or datetime, UTC).
        end: Exclusive end.
    """

    name: str
    dataset: str
    schema: str
    symbols: tuple[str, ...] | str
    stype_in: str
    start: str
    end: str

    def request(self) -> dict:
        """Keyword arguments shared by ``get_cost`` and ``get_range``."""
        return dict(
            dataset=self.dataset,
            schema=self.schema,
            symbols=list(self.symbols) if isinstance(self.symbols, tuple) else self.symbols,
            stype_in=self.stype_in,
            start=self.start,
            end=self.end,
        )

    @property
    def path(self) -> Path:
        return OUT_DIR / f"{self.name}.dbn.zst"


def _next_day(day: str) -> str:
    """Exclusive end bound for a single calendar day (dates here never cross a month end)."""
    return f"{day[:8]}{int(day[8:]) + 1:02d}"


def build_manifest() -> list[Pull]:
    """Assemble the fixed list of pulls for the bundle.

    Returns:
        Pull objects in download order (cheap tiers first, so a mid-run failure
        leaves the small, self-contained tiers complete).
    """
    pulls: list[Pull] = []
    for day in FOMC_DAYS + CONTROL_DAYS:
        tag = "fomc" if day in FOMC_DAYS else "ctrl"
        pulls.append(Pull(f"glbx_es_trades_{tag}_{day}", "GLBX.MDP3", "trades", ("ES.FUT",),
                          "parent", day, _next_day(day)))
    pulls.append(Pull("xnas_statistics_all_2026q3", "XNAS.ITCH", "statistics", "ALL_SYMBOLS",
                      "raw_symbol", "2026-06-01", "2026-09-01"))
    pulls.append(Pull("xnas_imbalance_eq50_12mo", "XNAS.ITCH", "imbalance", tuple(EQ50),
                      "raw_symbol", "2025-09-01", "2026-09-01"))
    for start, end in MONTHS:
        ym = start[:7]
        pulls.append(Pull(f"xnas_bbo1s_eq10_{ym}", "XNAS.ITCH", "bbo-1s", tuple(EQ10),
                          "raw_symbol", start, end))
    for start, end in MONTHS:
        ym = start[:7]
        pulls.append(Pull(f"xnas_trades_eq10_{ym}", "XNAS.ITCH", "trades", tuple(EQ10),
                          "raw_symbol", start, end))
    # Extension (2026-09-16): trade paths for the other 40 panel names, so the
    # auction encodings can run where the 50-name panel locates the effect.
    # No bbo-1s for these: the mid-price rerun matched the trade-price run.
    for start, end in MONTHS:
        ym = start[:7]
        pulls.append(Pull(f"xnas_trades_eq40_{ym}", "XNAS.ITCH", "trades", tuple(EQ50[10:]),
                          "raw_symbol", start, end))
    return pulls


def main(argv: list[str] | None = None) -> int:
    """Price the bundle, enforce the cap, then download what is missing.

    Args:
        argv: Command-line arguments (``--dry-run``, ``--max-cost``).

    Returns:
        Process exit code (0 on success).

    Raises:
        SystemExit: If the summed estimate exceeds ``--max-cost`` or the key is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="price only, download nothing")
    parser.add_argument("--max-cost", type=float, default=65.0,
                        help="abort if the summed estimate for MISSING files exceeds this (USD)")
    args = parser.parse_args(argv)

    load_dotenv(REPO / ".env")
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("DATABENTO_API_KEY not set (expected in .env)")
    client = db.Historical(key=key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pulls = build_manifest()
    rows = []
    total_missing = 0.0
    for p in pulls:
        cost = client.metadata.get_cost(**p.request())
        present = p.path.exists()
        rows.append({**asdict(p), "cost_usd": cost, "present": present})
        if not present:
            total_missing += cost
        print(f"{'have' if present else 'need':4s} {p.name:38s} {cost:7.2f} USD")
    print(f"\nestimated cost of missing files: {total_missing:.2f} USD (cap {args.max_cost:.2f})")
    # Hard gate: estimates are authoritative for billing, so refusing here is the
    # only point at which the credit balance is actually protected.
    if total_missing > args.max_cost:
        raise SystemExit("estimate exceeds --max-cost; nothing downloaded")
    if args.dry_run:
        return 0

    for p in pulls:
        if p.path.exists():
            continue
        print(f"downloading {p.name} ...", flush=True)
        # Streaming straight to disk keeps the multi-GB trades months out of memory.
        client.timeseries.get_range(**p.request(), path=p.path)
        print(f"  -> {p.path.stat().st_size / 1e6:.1f} MB", flush=True)

    manifest = OUT_DIR / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=1, default=list))
    print(f"manifest written to {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
