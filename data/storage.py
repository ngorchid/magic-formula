"""DuckDB-backed storage placeholder.

Intent: persist OHLCV, fundamentals, and signal panels in a single .duckdb file so
the same dataset can be queried from notebooks, the backtester, and the live trader.
Wire up once we have more than yfinance to ingest.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DB = Path(os.getenv("DB_PATH", "data/market.duckdb"))


def get_connection():
    raise NotImplementedError("DuckDB store not yet implemented")
