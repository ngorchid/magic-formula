"""yfinance OHLCV downloader with parquet caching.

Wide-format DataFrames keyed by date with one column per ticker are returned for the
fields users typically want (`close`, `adj_close`, `volume`, ...). Cache hits avoid
re-downloading; the cache key is field+start+end+ticker-set hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


def _cache_key(tickers: tuple[str, ...], start: str, end: str) -> str:
    h = hashlib.md5("|".join(tickers).encode()).hexdigest()[:10]
    return f"{start}_{end}_{h}"


def _try_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def download_ohlcv(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for `tickers`, return one wide DataFrame per field.

    Cache uses parquet when pyarrow/fastparquet is available; otherwise pickle.
    """
    tickers = tuple(sorted(set(tickers)))
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    key = _cache_key(tickers, start, end)
    ext = "parquet" if _try_parquet() else "pkl"
    cache_path = CACHE_DIR / f"ohlcv_{key}.{ext}"

    if use_cache and cache_path.exists():
        raw = pd.read_parquet(cache_path) if ext == "parquet" else pd.read_pickle(cache_path)
    else:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {len(tickers)} tickers {start}..{end}")
        if ext == "parquet":
            raw.to_parquet(cache_path)
        else:
            raw.to_pickle(cache_path)

    # yfinance returns columns as a 2-level MultiIndex: (field, ticker)
    out: dict[str, pd.DataFrame] = {}
    rename = {"Adj Close": "adj_close"}
    if isinstance(raw.columns, pd.MultiIndex):
        for top in raw.columns.get_level_values(0).unique():
            field = rename.get(top, top.lower().replace(" ", "_"))
            out[field] = raw[top].sort_index()
    else:
        # single ticker case
        df = raw.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        for f in df.columns:
            out[rename.get(f, f)] = df[[f]].rename(columns={f: tickers[0]})
    return out


def load_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    field: str = "adj_close",
) -> pd.DataFrame:
    """Convenience: return one wide DataFrame for a single field."""
    panel = download_ohlcv(tickers, start, end)
    if field not in panel:
        raise KeyError(f"field '{field}' not in {list(panel)}")
    return panel[field]
