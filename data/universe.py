"""Universe definitions.

``sp500_*`` (no ``pit``) return the *current* constituents and are survivorship-biased —
fine for a quick run, wrong for real conclusions. The ``sp500_pit_*`` functions give a
**point-in-time** universe from a historical-membership dataset (fja05680/sp500, 1996→
present): at each date you see only the names that were actually in the index then, so a
backtest can't buy tomorrow's index entrants (e.g. TSLA before 2021) or silently drop
names that later left. Caveat: it fixes index-composition look-ahead, but names that
fully delisted (bankruptcy/acquisition) still lack free prices/fundamentals, so their
returns remain absent — a residual, smaller bias.
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X) algo_trading/0.1"
FALLBACK_PATH = Path(__file__).resolve().parent / "sp500_fallback.csv"

# Point-in-time historical S&P 500 membership (change-dated snapshots, 1996→present).
SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
SP500_HISTORY_CACHE = Path(__file__).resolve().parent / "cache" / "sp500_historical.csv"
_SUFFIX_RE = re.compile(r"-\d{6}$")  # fja05680 tags removed names as TICKER-YYYYMM


def _norm_ticker(t: str) -> str:
    """Membership symbol -> project convention: strip removal tag, BRK.B -> BRK-B."""
    return _SUFFIX_RE.sub("", str(t).strip()).upper().replace(".", "-")


def _fetch_wiki_table() -> pd.DataFrame:
    resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return df


@lru_cache(maxsize=1)
def sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 constituents with sector + sub-industry."""
    try:
        df = _fetch_wiki_table()
        out = df.rename(
            columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
            }
        )[["ticker", "name", "sector", "sub_industry"]]
        return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    except Exception as e:
        if FALLBACK_PATH.exists():
            print(f"[universe] Wikipedia fetch failed ({e!r}); using packaged fallback")
            return pd.read_csv(FALLBACK_PATH)
        raise


@lru_cache(maxsize=1)
def sp500_tickers() -> list[str]:
    """Current S&P 500 tickers, normalised for yfinance (BRK.B -> BRK-B)."""
    return sorted(sp500_constituents()["ticker"].tolist())


@lru_cache(maxsize=1)
def sp500_sectors() -> pd.Series:
    """Series mapping ticker -> GICS sector. Static (current snapshot)."""
    df = sp500_constituents()
    return df.set_index("ticker")["sector"]


# --- Broad current universe (S&P 400 mid + 600 small) ----------------------
# Current constituents only, so survivorship-biased — but they add genuine mid- and
# small-cap names the S&P 500 lacks, which is what a size-effect study needs.
WIKI_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
WIKI_SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


def _fetch_constituents(url: str) -> pd.DataFrame:
    """Fetch a Wikipedia index-constituents table -> [ticker, sector]."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    for tbl in pd.read_html(io.StringIO(resp.text)):
        cols = {str(c): c for c in tbl.columns}
        sym = next((cols[c] for c in cols if "Symbol" in c or "Ticker" in c), None)
        sec = next((cols[c] for c in cols if "Sector" in c), None)
        if sym is None:
            continue
        out = pd.DataFrame({"ticker": tbl[sym].astype(str).str.replace(".", "-", regex=False)})
        out["sector"] = tbl[sec].astype(str) if sec is not None else pd.NA
        return out.drop_duplicates("ticker").reset_index(drop=True)
    raise ValueError(f"no constituents table found at {url}")


@lru_cache(maxsize=1)
def sp1500_constituents() -> pd.DataFrame:
    """Current S&P 1500 (500 large + 400 mid + 600 small) with a size-tier label."""
    parts = [(sp500_constituents()[["ticker", "sector"]], "large")]
    for url, tier in [(WIKI_SP400_URL, "mid"), (WIKI_SP600_URL, "small")]:
        try:
            parts.append((_fetch_constituents(url), tier))
        except Exception as e:  # noqa: BLE001 - fall back to whatever tiers we got
            print(f"[universe] {tier}-cap fetch failed ({e!r}); skipping")
    frames = []
    for df, tier in parts:
        d = df.copy()
        d["tier"] = tier
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    # A name can sit in only one index; keep the first (largest) tier it appears in.
    return out.drop_duplicates("ticker", keep="first").sort_values("ticker").reset_index(drop=True)


@lru_cache(maxsize=1)
def sp1500_tickers() -> list[str]:
    """Current S&P 1500 tickers, normalised for yfinance."""
    return sorted(sp1500_constituents()["ticker"].tolist())


@lru_cache(maxsize=1)
def sp1500_sectors() -> pd.Series:
    """Series mapping S&P 1500 ticker -> GICS sector (current snapshot)."""
    return sp1500_constituents().set_index("ticker")["sector"]


# --- European (eurozone / EUR) universe ------------------------------------
# Current constituents of the major eurozone national indices, whose Wikipedia tables
# already carry yfinance-suffixed tickers (ADS.DE, AC.PA, ABN.AS, ...). All EUR-quoted,
# so cross-sectional signals aren't polluted by FX. Current-only -> survivorship-biased.
EURO_INDEX_PAGES = {
    "DAX": "https://en.wikipedia.org/wiki/DAX",
    "CAC 40": "https://en.wikipedia.org/wiki/CAC_40",
    "AEX": "https://en.wikipedia.org/wiki/AEX_index",
    "IBEX 35": "https://en.wikipedia.org/wiki/IBEX_35",
    "FTSE MIB": "https://en.wikipedia.org/wiki/FTSE_MIB",
}
_EUR_SUFFIXES = (".DE", ".PA", ".AS", ".MC", ".MI")


@lru_cache(maxsize=1)
def european_eur_tickers() -> list[str]:
    """Deduped large-cap eurozone tickers (EUR only) from the major national indices,
    already in yfinance form. Current constituents => survivorship-biased."""
    out: set[str] = set()
    for name, url in EURO_INDEX_PAGES.items():
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            resp.raise_for_status()
            for tbl in pd.read_html(io.StringIO(resp.text)):
                cols = [str(c) for c in tbl.columns]
                tc = next((c for c in cols if "Ticker" in c or "Symbol" in c), None)
                if tc and len(tbl) >= 15:
                    out |= {str(x).strip() for x in tbl[tc] if isinstance(x, str)}
                    break
        except Exception as e:  # noqa: BLE001 - skip an index that fails to fetch
            print(f"[universe] {name} fetch failed ({e!r}); skipping")
    return sorted(t for t in out if t.endswith(_EUR_SUFFIXES))


# --- Broad US universe (all SEC filers) ------------------------------------
def broad_us_tickers() -> list[str]:
    """Every current US SEC filer with a plain common-stock ticker (~10k names).

    Sourced from EDGAR's ticker->CIK master list — the whole listed US market, not just an
    index, so it reaches genuine small/micro caps. Non-operating filers (ETFs, funds, shells)
    have no income-statement fundamentals and drop out naturally downstream. Current filers
    only => survivorship-biased (worst for small caps)."""
    from data.fundamentals import _edgar_cik_map

    return sorted(t for t in _edgar_cik_map() if t.isalpha() and len(t) <= 5)


# --- Point-in-time membership ---------------------------------------------
@lru_cache(maxsize=1)
def _sp500_history() -> pd.Series:
    """Change-dated membership: Series[change_date -> frozenset of normalised tickers].

    Downloads the fja05680 historical-components CSV once and caches it to disk. Each
    row is a date on which membership changed; the value is the full roster as of then.
    """
    SP500_HISTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SP500_HISTORY_CACHE.exists():
        df = pd.read_csv(SP500_HISTORY_CACHE)
    else:
        resp = requests.get(SP500_HISTORY_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(SP500_HISTORY_CACHE, index=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return pd.Series(
        [frozenset(_norm_ticker(t) for t in row.split(",")) for row in df["tickers"]],
        index=df["date"].values,
    )


def sp500_pit_members(date) -> frozenset[str]:
    """Set of S&P 500 tickers that were in the index on ``date`` (as-of, forward-filled)."""
    hist = _sp500_history()
    ts = pd.Timestamp(date)
    prior = hist.index[hist.index <= ts]
    if len(prior) == 0:
        return frozenset()
    return hist.loc[prior[-1]]


def sp500_pit_universe(start: str, end: str | None = None) -> list[str]:
    """Sorted union of every ticker that was an index member at any point in [start, end].

    This is the set of names to fetch prices/fundamentals for; combine with
    ``sp500_pit_eligible`` to restrict each date to that date's actual members.
    """
    hist = _sp500_history()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end or pd.Timestamp.today())
    # Include the roster in force at `start` (last change on/before it) plus all changes
    # within the window, so names present at the window open aren't missed.
    dates = hist.index[(hist.index >= lo) & (hist.index <= hi)]
    members: set[str] = set(sp500_pit_members(lo))
    for dt in dates:
        members |= hist.loc[dt]
    return sorted(members)


def sp500_pit_eligible(calendar: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Boolean ``[date × ticker]`` mask: True where the ticker was an index member.

    Built as a step function over the change dates, forward-filled onto ``calendar``.
    """
    hist = _sp500_history()
    all_tickers = sorted(set().union(*hist.values)) if len(hist) else []
    # Float 0/1 (not bool) so the reindex+ffill never triggers an object downcast.
    mat = pd.DataFrame(0.0, index=hist.index, columns=all_tickers)
    for dt, members in hist.items():
        cols = [t for t in members if t in mat.columns]
        mat.loc[dt, cols] = 1.0
    mat = mat.reindex(mat.index.union(calendar)).ffill().reindex(calendar).fillna(0.0)
    return mat.reindex(columns=tickers, fill_value=0.0) > 0.5
