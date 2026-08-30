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
import logging
import time
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


# --- European universe ------------------------------------------------------
# Current constituents of the major European national indices, scraped from Wikipedia.
# Current-only => SURVIVORSHIP-BIASED; there is no PIT membership data for these venues, so
# any backtest on them overstates. The live sleeve is unaffected (it only needs today's list).
#
# Each entry is (url, suffix). The suffix is NOT cosmetic: only the original big five publish
# yfinance-form tickers (ADS.DE, AC.PA). The venues added 2026-08-28 publish BARE or
# EXCHANGE-PREFIXED symbols -- "Euronext Brussels:\xa0ABI", "OSE: AKRBP", "MAERSK B" -- so the
# earlier "keep anything already ending in a known suffix" filter silently discarded every one
# of them and the scrape returned nothing at all for six of the ten indices. Normalising per
# venue is what makes them usable.
EURO_INDEX_PAGES = {
    "DAX": ("https://en.wikipedia.org/wiki/DAX", ".DE"),
    "CAC 40": ("https://en.wikipedia.org/wiki/CAC_40", ".PA"),
    "AEX": ("https://en.wikipedia.org/wiki/AEX_index", ".AS"),
    "IBEX 35": ("https://en.wikipedia.org/wiki/IBEX_35", ".MC"),
    "FTSE MIB": ("https://en.wikipedia.org/wiki/FTSE_MIB", ".MI"),
    # Added 2026-08-28: eurozone venues the broker already supports, at ZERO incremental FX
    # exposure. Fundamentals coverage was measured first and is not the blocker -- .BR 75%,
    # .HE 75%, .IR 67%, .LS 100% usable against the live yfinance extraction path, versus 83%
    # for the incumbent five indices and 67% for the US control itself.
    "BEL 20": ("https://en.wikipedia.org/wiki/BEL_20", ".BR"),
    "OMX Helsinki 25": ("https://en.wikipedia.org/wiki/OMX_Helsinki_25", ".HE"),
    # ISEQ 20 (.IR) REMOVED 2026-08-30. All 20 Irish names failed to qualify against live IB —
    # not a symbol-format issue (they fail on the correct symbol and on Euronext Dublin's own
    # exchange code ISED alike), so the account simply has no Dublin trading permission. Swept
    # via scripts/ib_symbol_probe.py: 0/20 qualified. Keeping them only fed the "universe
    # shrinking" alerts with 20 permanently-unresolvable names. RESTORE this line once Euronext
    # Dublin market data + trading are enabled on the account, then re-run the probe to confirm.
    # "ISEQ 20": ("https://en.wikipedia.org/wiki/ISEQ_20", ".IR"),
    "PSI": ("https://en.wikipedia.org/wiki/PSI-20", ".LS"),
    # VIENNA (.VI) IS DELIBERATELY ABSENT. Neither the English nor the German ATX article
    # carries a ticker/ISIN column -- the constituent table is Company/Industry/Sector only --
    # so there is nothing to scrape, and mapping ~20 company names to symbols by hand would be
    # a static list masquerading as a feed. Costs ~20 names; revisit with a real index source.
}

# Non-eurozone Europe. Separate because every one of these adds a CURRENCY, and the sleeve
# holds unhedged local-currency equity, so the FX move lands directly in the P&L. Measured
# 2026-08-28 that contribution is 0.02-0.19pp of portfolio vol -- small enough that excluding
# a company purely for being quoted in CHF or SEK is not defensible. The real argument FOR
# them is diversification: these markets correlate 0.60-0.87 with the current EUR book, versus
# 0.96-0.97 among the eurozone indices themselves, which are nearly the same bet.
# WARNING .L is quoted in PENCE. See _MINOR_UNITS in paper/live_data.py -- without that table
# every LSE market cap is 100x too large and the size filter is silently defeated.
NON_EUR_INDEX_PAGES = {
    "FTSE 100": ("https://en.wikipedia.org/wiki/FTSE_100_Index", ".L"),
    "SMI": ("https://en.wikipedia.org/wiki/Swiss_Market_Index", ".SW"),
    "OMX Stockholm 30": ("https://en.wikipedia.org/wiki/OMX_Stockholm_30", ".ST"),
    "OMX Copenhagen 25": ("https://en.wikipedia.org/wiki/OMX_Copenhagen_25", ".CO"),
    "OBX": ("https://en.wikipedia.org/wiki/OBX_Index", ".OL"),
}

_EUR_SUFFIXES = tuple(sfx for _, sfx in EURO_INDEX_PAGES.values())
_NON_EUR_SUFFIXES = tuple(sfx for _, sfx in NON_EUR_INDEX_PAGES.values())
_ALL_SUFFIXES = _EUR_SUFFIXES + _NON_EUR_SUFFIXES


def _normalise_ticker(raw: str, suffix: str) -> str | None:
    """Wikipedia cell -> yfinance ticker, or None if it isn't a symbol at all.

    Handles the three shapes actually observed on these pages (2026-08-28):
      "Euronext Brussels:\xa0ABI" -> ABI.BR     (exchange prefix, non-breaking space)
      "OSE: AKRBP"                -> AKRBP.OL   (prefix with an ordinary space)
      "MAERSK B"                  -> MAERSK-B.CO (Nordic share class; yfinance uses a hyphen)
    A symbol that already carries a known venue suffix is returned untouched, so the original
    five indices are unaffected.
    """
    t = str(raw).replace("\xa0", " ").strip()
    if ":" in t:                      # drop an exchange prefix
        t = t.split(":")[-1].strip()
    if not t or t.lower() == "nan":
        return None
    if t.endswith(_ALL_SUFFIXES):     # already yfinance form
        return t
    # Share-class separators. Wikipedia writes these as a space ("MAERSK B") or a dot
    # ("BT.A"); yfinance uses a hyphen in the root and reserves the dot for the venue suffix,
    # so "BT.A" + ".L" would otherwise yield the unresolvable "BT.A.L" instead of "BT-A.L".
    t = t.replace(" ", "-").replace(".", "-")
    # Reject prose that slipped out of a merged/misaligned cell.
    if not all(c.isalnum() or c in "-." for c in t) or len(t) > 12:
        return None
    return t + suffix


def _scrape_index_tickers(pages: dict[str, tuple[str, str]], min_rows: int = 12) -> list[str]:
    """Deduped, yfinance-form tickers from Wikipedia constituent tables.

    `min_rows` guards against grabbing a sidebar or summary table instead of the constituent
    list. It is looser than the 15 the big five used, because ISEQ 20 / OMXC25 / OBX / BEL 20
    legitimately list only ~20 names.
    """
    out: set[str] = set()
    failed: list[str] = []
    for name, (url, suffix) in pages.items():
        got_any = False
        # RETRY. Wikipedia rate-limits (HTTP 403 "Too many requests") when several indices are
        # fetched back to back, and on 2026-08-30 that silently cost 68 of 461 names -- three
        # indices dropped with only a log line, and nothing downstream could tell a rate-limited
        # run from a genuinely smaller universe. A shrinking universe must never be quiet.
        for attempt in range(3):
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
                resp.raise_for_status()
                for tbl in pd.read_html(io.StringIO(resp.text)):
                    cols = [str(c) for c in tbl.columns]
                    tc = next((c for c in cols if "Ticker" in c or "Symbol" in c
                               or "MNEM" in c), None)   # ISEQ labels its column "MNEM code"
                    if tc and len(tbl) >= min_rows:
                        got = {_normalise_ticker(x, suffix) for x in tbl[tc]}
                        out |= {t for t in got if t}
                        got_any = True
                        break
                if got_any:
                    break
                print(f"[universe] {name}: no constituent table found")
                break                                   # a missing table will not fix itself
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"[universe] {name} fetch FAILED after 3 attempts ({e!r})")
                else:
                    time.sleep(2.0 * (attempt + 1))     # linear backoff; the 403 is transient
        if not got_any:
            failed.append(name)
    if failed:
        # LOUD, because the caller cannot distinguish "index unavailable" from "index is small".
        logging.error("UNIVERSE INCOMPLETE — %d of %d indices did not scrape: %s. The universe "
                      "is smaller than configured and rankings will silently omit those venues.",
                      len(failed), len(pages), ", ".join(failed))
    return sorted(out)


@lru_cache(maxsize=1)
def european_eur_tickers() -> list[str]:
    """Deduped large-cap EUROZONE tickers (EUR-quoted only), in yfinance form."""
    return _scrape_index_tickers(EURO_INDEX_PAGES)


@lru_cache(maxsize=1)
def european_non_eur_tickers() -> list[str]:
    """Deduped large-cap NON-eurozone European tickers (GBp/CHF/SEK/DKK/NOK), yfinance form."""
    return _scrape_index_tickers(NON_EUR_INDEX_PAGES)


@lru_cache(maxsize=1)
def european_tickers() -> list[str]:
    """All European names: eurozone plus non-eurozone."""
    return sorted(set(european_eur_tickers()) | set(european_non_eur_tickers()))


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
