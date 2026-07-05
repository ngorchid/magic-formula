"""SimFin daily share prices, market cap, and a broad point-in-time-ish universe.

Used by the broad-universe strategies (e.g. Magic Formula) where pricing thousands of
names via yfinance is impractical and survivorship-biased. SimFin ships one bulk
daily-prices dataset for the whole US market, *including shares outstanding* (so market
cap is trivial) and covering names yfinance would drop. Free-tier prices are a rolling
~5-year window and lag ~1 year — fine for a backtest preview, not for live trading.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# SimFin native column -> canonical wide-panel field name.
_FIELD_MAP = {
    "Adj. Close": "adj_close",
    "Close": "close",
    "Volume": "volume",
    "Shares Outstanding": "shares",
}

# Sectors excluded by the classic Magic Formula (EBIT/EV and ROC are ill-defined here).
EXCLUDED_SECTORS = ("Financial Services", "Utilities")


def _norm(ticker: str) -> str:
    return str(ticker).upper().replace(".", "-").strip()


def _init_simfin():
    import simfin as sf

    key = os.getenv("SIMFIN_API_KEY")
    if not key:
        raise RuntimeError("SIMFIN_API_KEY not set in .env")
    sf.set_api_key(key)
    sf.set_data_dir(os.getenv("SIMFIN_DATA_DIR", str(ROOT / "data" / "simfin_cache")))
    return sf


@lru_cache(maxsize=1)
def _shareprices() -> pd.DataFrame:
    sf = _init_simfin()
    df = sf.load_shareprices(variant="daily", market="us").reset_index()
    df["Ticker"] = df["Ticker"].map(_norm)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@lru_cache(maxsize=1)
def simfin_sector_map() -> pd.Series:
    """ticker -> GICS-style sector (from SimFin industries)."""
    sf = _init_simfin()
    comp = sf.load_companies(market="us")
    ind = sf.load_industries()
    sec = comp["IndustryId"].map(ind["Sector"])
    sec.index = [_norm(t) for t in comp.index]
    return sec.dropna()


def load_simfin_prices(
    tickers: list[str] | None,
    start: str,
    end: str | None = None,
    fields: tuple[str, ...] = ("adj_close", "close", "volume", "shares"),
) -> dict[str, pd.DataFrame]:
    """Return ``{field: [date × ticker]}`` wide panels from SimFin daily prices.

    `tickers=None` returns the whole available universe. Shares outstanding are
    forward-filled per ticker (they only change on corporate actions / filings).
    """
    df = _shareprices()
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    if tickers is not None:
        mask &= df["Ticker"].isin({_norm(t) for t in tickers})
    sub = df.loc[mask]

    out: dict[str, pd.DataFrame] = {}
    for native, field in _FIELD_MAP.items():
        if field not in fields or native not in sub.columns:
            continue
        wide = sub.pivot_table(index="Date", columns="Ticker", values=native).sort_index()
        if field == "shares":
            wide = wide.ffill()
        out[field] = wide
    return out


def broad_universe(
    start: str,
    end: str | None = None,
    min_market_cap: float = 3e8,
    exclude_sectors: tuple[str, ...] = EXCLUDED_SECTORS,
) -> tuple[list[str], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a broad US universe with sector exclusions and a market-cap floor.

    Returns ``(tickers, eligible_mask, price_panels)`` where `eligible_mask` is a
    ``[date × ticker]`` boolean (market cap ≥ floor AND sector allowed) for
    point-in-time membership, and `price_panels` is the SimFin price dict.
    """
    sectors = simfin_sector_map()
    allowed = sectors[~sectors.isin(exclude_sectors)].index.tolist()
    panels = load_simfin_prices(allowed, start, end)
    close, shares = panels["close"], panels["shares"]
    mcap = (close * shares.reindex_like(close)).dropna(how="all", axis=1)
    tickers = list(mcap.columns)
    eligible = mcap >= min_market_cap
    return tickers, eligible, {k: v.reindex(columns=tickers) for k, v in panels.items()}
