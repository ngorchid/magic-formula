"""Point-in-time fundamentals loader with a multi-vendor backend.

Returns canonical accounting line items as wide ``[date × ticker]`` panels, daily
forward-filled and **point-in-time correct**: each value only becomes visible on the
date it was actually published, never on the (earlier) period-end date. This is the
fundamentals analogue of the 1-day weight lag in the backtester — without it, a
quality/value backtest silently uses numbers the market did not yet have.

Three backends sit behind one interface, preferred in order:

  1. **SEC EDGAR** (free, deep, point-in-time). Pulls the XBRL ``companyfacts`` feed
     straight from ``data.sec.gov`` — every figure a US filer has ever reported, each
     stamped with its real ``filed`` (publication) date, back to the ~2009 XBRL
     mandate. This is the deepest *and* cleanest free source: PIT via filing dates,
     restatements time-indexed, delisted names retained. The cost is engineering —
     income/cash-flow tags arrive as fiscal-year-to-date cumulatives, so we
     reconstruct discrete quarters (YTD differencing) before handing them to the
     shared TTM/PIT machinery. Preferred first.
  2. **SimFin** (free tier). Ships a real ``Publish Date`` per statement, so PIT
     alignment is exact, but the free tier is only ~5 years deep. Patches gaps.
  3. **yfinance** (fallback). Only carries ~4-5 quarters and has *no* filing date,
     so we approximate the publish date as ``report_date + report_lag_days``. Useful
     only to patch very recent gaps / names the others do not cover.

Per (ticker, item) we take the first source that has data; a parallel provenance
mask records which vendor supplied each cell so the cross-vendor disagreement
(and hence "is better data worth paying for?") can be measured downstream.

A Sharadar/CRSP backend would slot in as a further entry behind the same interface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- Canonical vocabulary --------------------------------------------------
# kind="flow"  -> trailing-twelve-month (sum of last 4 quarters) to kill seasonality
# kind="stock" -> balance-sheet level, used as-of (no TTM)
#
# Each entry maps the canonical name onto each vendor's native column/row names.
@dataclass(frozen=True)
class Item:
    kind: str            # "flow" | "stock"
    simfin: tuple[str, str]          # (statement, column)
    yfinance: tuple[str, tuple[str, ...]]  # (statement, candidate row names)
    edgar: tuple[str, ...] = ()      # candidate US-GAAP XBRL tags, tried in order


CANONICAL: dict[str, Item] = {
    "revenue":             Item("flow",  ("income",   "Revenue"),
                                          ("income",   ("Total Revenue", "Operating Revenue")),
                                          edgar=("RevenueFromContractWithCustomerExcludingAssessedTax",
                                                 "Revenues", "SalesRevenueNet",
                                                 "RevenueFromContractWithCustomerIncludingAssessedTax")),
    "cogs":                Item("flow",  ("income",   "Cost of Revenue"),
                                          ("income",   ("Cost Of Revenue", "Reconciled Cost Of Revenue")),
                                          edgar=("CostOfGoodsAndServicesSold", "CostOfRevenue",
                                                 "CostOfGoodsSold")),
    "gross_profit":        Item("flow",  ("income",   "Gross Profit"),
                                          ("income",   ("Gross Profit",)),
                                          edgar=("GrossProfit",)),
    "net_income":          Item("flow",  ("income",   "Net Income"),
                                          ("income",   ("Net Income", "Net Income Common Stockholders")),
                                          edgar=("NetIncomeLoss", "ProfitLoss")),
    "operating_cash_flow": Item("flow",  ("cashflow", "Net Cash from Operating Activities"),
                                          ("cashflow", ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")),
                                          edgar=("NetCashProvidedByUsedInOperatingActivities",
                                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    "total_assets":        Item("stock", ("balance",  "Total Assets"),
                                          ("balance",  ("Total Assets",)),
                                          edgar=("Assets",)),
    "total_equity":        Item("stock", ("balance",  "Total Equity"),
                                          ("balance",  ("Stockholders Equity", "Total Equity Gross Minority Interest")),
                                          edgar=("StockholdersEquity",
                                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    "shares_diluted":      Item("stock", ("income",   "Shares (Diluted)"),
                                          ("income",   ("Diluted Average Shares", "Basic Average Shares")),
                                          edgar=("WeightedAverageNumberOfDilutedSharesOutstanding",
                                                 "WeightedAverageNumberOfSharesOutstandingBasic")),
    # --- Magic-Formula inputs (EBIT/EV yield + return on capital) ---
    "operating_income":    Item("flow",  ("income",   "Operating Income (Loss)"),
                                          ("income",   ("Operating Income", "EBIT", "Total Operating Income As Reported")),
                                          edgar=("OperatingIncomeLoss",)),
    "short_term_debt":     Item("stock", ("balance",  "Short Term Debt"),
                                          ("balance",  ("Current Debt", "Short Long Term Debt")),
                                          edgar=("DebtCurrent", "LongTermDebtCurrent",
                                                 "ShortTermBorrowings")),
    "long_term_debt":      Item("stock", ("balance",  "Long Term Debt"),
                                          ("balance",  ("Long Term Debt",)),
                                          edgar=("LongTermDebtNoncurrent", "LongTermDebt")),
    "cash":                Item("stock", ("balance",  "Cash, Cash Equivalents & Short Term Investments"),
                                          ("balance",  ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")),
                                          edgar=("CashAndCashEquivalentsAtCarryingValue",
                                                 "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")),
    "total_current_assets": Item("stock", ("balance", "Total Current Assets"),
                                          ("balance",  ("Current Assets", "Total Current Assets")),
                                          edgar=("AssetsCurrent",)),
    "total_current_liabilities": Item("stock", ("balance", "Total Current Liabilities"),
                                          ("balance",  ("Current Liabilities", "Total Current Liabilities")),
                                          edgar=("LiabilitiesCurrent",)),
    "ppe_net":             Item("stock", ("balance",  "Property, Plant & Equipment, Net"),
                                          ("balance",  ("Net PPE", "Property Plant And Equipment Net")),
                                          edgar=("PropertyPlantAndEquipmentNet",)),
    # Capital expenditure (cash outflow, reported positive). Free cash flow =
    # operating_cash_flow - capex; feeds the FCF-yield Magic-Formula variant.
    "capex":               Item("flow",  ("cashflow", "Change in Fixed Assets & Intangibles"),
                                          ("cashflow", ("Capital Expenditure", "Purchase Of PPE")),
                                          edgar=("PaymentsToAcquirePropertyPlantAndEquipment",
                                                 "PaymentsToAcquireProductiveAssets")),
}

_SIMFIN_LOADERS = {"income": "load_income", "balance": "load_balance", "cashflow": "load_cashflow"}


def _norm(ticker: str) -> str:
    """Normalise a ticker to the project convention (BRK.B -> BRK-B)."""
    return str(ticker).upper().replace(".", "-").strip()


# --- SimFin backend --------------------------------------------------------
@lru_cache(maxsize=4)
def _simfin_statement(statement: str) -> pd.DataFrame:
    """Load (download+cache) one quarterly SimFin statement for the US market.

    Index normalised to (ticker, report_date); 'Publish Date' kept as a column.
    """
    import simfin as sf

    key = os.getenv("SIMFIN_API_KEY")
    if not key:
        raise RuntimeError("SIMFIN_API_KEY not set in .env")
    sf.set_api_key(key)
    sf.set_data_dir(os.getenv("SIMFIN_DATA_DIR", str(ROOT / "data" / "simfin_cache")))

    df = getattr(sf, _SIMFIN_LOADERS[statement])(variant="quarterly", market="us")
    df = df.reset_index()
    df["Ticker"] = df["Ticker"].map(_norm)
    df["Report Date"] = pd.to_datetime(df["Report Date"])
    df["Publish Date"] = pd.to_datetime(df["Publish Date"])
    return df


def _pit_panel(
    long: pd.DataFrame,
    value_col: str,
    kind: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long (ticker, report_date, publish_date, value) -> PIT daily wide panel.

    flow items are converted to a trailing-4-quarter sum *by report date* before
    being stamped with their publish date; the result is forward-filled onto the
    daily `calendar`, so each date sees only the latest already-published figure.
    """
    s = long[["Ticker", "Report Date", "Publish Date", value_col]].copy()
    s = s.dropna(subset=[value_col, "Publish Date"])
    # Restatements: keep the latest-published figure for each (ticker, report date).
    s = s.sort_values(["Ticker", "Report Date", "Publish Date"]).drop_duplicates(
        ["Ticker", "Report Date"], keep="last"
    )
    if kind == "flow":
        s[value_col] = (
            s.groupby("Ticker")[value_col]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )
        s = s.dropna(subset=[value_col])

    # If several reports share a publish date, keep the most recent report.
    s = s.sort_values(["Ticker", "Publish Date", "Report Date"]).drop_duplicates(
        ["Ticker", "Publish Date"], keep="last"
    )
    wide = s.pivot(index="Publish Date", columns="Ticker", values=value_col).sort_index()
    full = wide.index.union(calendar)
    return wide.reindex(full).ffill().reindex(calendar)


def load_simfin(
    tickers: list[str],
    items: list[str],
    calendar: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    tickers = [_norm(t) for t in tickers]
    for name in items:
        spec = CANONICAL[name]
        stmt, col = spec.simfin
        long = _simfin_statement(stmt)
        panel = _pit_panel(long, col, spec.kind, calendar)
        out[name] = panel.reindex(columns=tickers)
    return out


# --- yfinance backend (fallback) ------------------------------------------
def _yf_statement_row(stmt_df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    """Pick the first matching row (line item) from a yfinance statement frame."""
    if stmt_df is None or stmt_df.empty:
        return None
    for name in candidates:
        if name in stmt_df.index:
            return stmt_df.loc[name]
    return None


def load_yfinance(
    tickers: list[str],
    items: list[str],
    calendar: pd.DatetimeIndex,
    report_lag_days: int = 60,
) -> dict[str, pd.DataFrame]:
    """Best-effort fundamentals from yfinance. History is shallow (~4-5 quarters)
    and there is no filing date, so report dates are lagged by `report_lag_days`
    to approximate publication and avoid lookahead.
    """
    import yfinance as yf

    tickers = [_norm(t) for t in tickers]
    # statement -> {ticker -> quarterly DataFrame (rows=line items, cols=period ends)}
    raw: dict[str, dict[str, pd.DataFrame]] = {"income": {}, "balance": {}, "cashflow": {}}
    getters = {
        "income": "quarterly_income_stmt",
        "balance": "quarterly_balance_sheet",
        "cashflow": "quarterly_cashflow",
    }
    needed_stmts = {CANONICAL[i].yfinance[0] for i in items}
    for t in tickers:
        tk = yf.Ticker(t.replace("-", "."))  # yfinance wants BRK.B form for some names
        for stmt in needed_stmts:
            try:
                raw[stmt][t] = getattr(tk, getters[stmt])
            except Exception:
                raw[stmt][t] = pd.DataFrame()

    out: dict[str, pd.DataFrame] = {}
    for name in items:
        spec = CANONICAL[name]
        stmt, candidates = spec.yfinance
        cols: dict[str, pd.Series] = {}
        for t in tickers:
            row = _yf_statement_row(raw[stmt].get(t), candidates)
            if row is None:
                continue
            ser = row.copy()
            ser.index = pd.to_datetime(ser.index) + pd.Timedelta(days=report_lag_days)
            ser = ser[~ser.index.duplicated(keep="last")].sort_index()
            cols[t] = ser
        if not cols:
            out[name] = pd.DataFrame(index=calendar, columns=tickers, dtype=float)
            continue
        long = pd.concat(cols, axis=1)  # [publish≈date × ticker], quarterly
        if spec.kind == "flow":
            long = long.rolling(4, min_periods=4).sum()
        out[name] = long.sort_index().reindex(
            long.index.union(calendar)
        ).ffill().reindex(calendar).reindex(columns=tickers)
    return out


# --- SEC EDGAR backend -----------------------------------------------------
# Free XBRL company-facts feed from data.sec.gov. Deepest free source (~2009+) and
# genuinely point-in-time: every fact carries the `filed` date it became public.
# SEC asks for a descriptive User-Agent with contact info and caps requests at
# ~10/sec; companyfacts is one request per ticker (all tags at once) and is cached
# to disk, so a full universe is a one-off ~1 min pull then instant.
_EDGAR_CACHE = ROOT / "data" / "edgar_cache"
_SEC_UA = os.getenv("SEC_USER_AGENT", "algo_trading research nicolas.greiner.1@gmail.com")
_SEC_MIN_INTERVAL = 0.12   # seconds between SEC calls (< 10 req/s)
_sec_last_call = [0.0]


def _sec_get_json(url: str) -> dict | None:
    """GET a data.sec.gov JSON endpoint with the required UA header + rate limit."""
    import time

    import requests

    wait = _SEC_MIN_INTERVAL - (time.time() - _sec_last_call[0])
    if wait > 0:
        time.sleep(wait)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _SEC_UA, "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        _sec_last_call[0] = time.time()
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001 - best-effort network fetch
        print(f"[edgar] fetch failed {url}: {e!r}")
        return None


@lru_cache(maxsize=1)
def _edgar_cik_map() -> dict[str, str]:
    """Map normalised ticker -> zero-padded 10-digit CIK, from SEC's master list."""
    import json

    _EDGAR_CACHE.mkdir(parents=True, exist_ok=True)
    path = _EDGAR_CACHE / "company_tickers.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = _sec_get_json("https://www.sec.gov/files/company_tickers.json")
        if data is None:
            return {}
        path.write_text(json.dumps(data))
    out: dict[str, str] = {}
    for row in data.values():
        out[_norm(row["ticker"])] = str(row["cik_str"]).zfill(10)
    return out


def _edgar_companyfacts(cik: str) -> dict | None:
    """All XBRL facts for one CIK, cached to disk (data/edgar_cache/CIK*.json)."""
    import json

    _EDGAR_CACHE.mkdir(parents=True, exist_ok=True)
    path = _EDGAR_CACHE / f"CIK{cik}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - corrupt cache, refetch
            pass
    data = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if data is not None:
        path.write_text(json.dumps(data))
    return data


def _edgar_pick_unit(units: dict) -> list | None:
    """Choose the fact list to use for a tag, preferring dollars then share counts."""
    for u in ("USD", "shares", "USD/shares"):
        if u in units:
            return units[u]
    return next(iter(units.values())) if units else None


def _edgar_tag_long(recs: list, kind: str) -> pd.DataFrame | None:
    """One XBRL tag's raw facts -> long ``[Report Date, Publish Date, val]``.

    ``stock`` items (balance-sheet instants, plus the diluted-share average) are taken
    as-of their period end. ``flow`` items arrive as fiscal-year-to-date cumulatives,
    so within each fiscal year we take the year-start ladder (3/6/9/12-month figures)
    and difference it into four discrete quarters; the shared ``_pit_panel`` then
    rolls those into a trailing-twelve-month sum, exactly as for the SimFin backend.
    """
    rows: list[tuple] = []
    if kind == "stock":
        for r in recs:
            if r.get("val") is None or "end" not in r or "filed" not in r:
                continue
            # Share counts are duration averages; keep only ~quarterly ones so a
            # ticker's annual average does not smear across the year.
            if "start" in r:
                dur = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
                if dur > 100:
                    continue
            rows.append((r["end"], r["filed"], r["val"]))
        # Later filings re-report a period as a comparative; keep the ORIGINAL
        # disclosure (earliest filing) so the publish date reflects true freshness
        # and the value is as-originally-reported (no hindsight, no lookahead).
    else:  # flow: reconstruct discrete quarters via YTD differencing
        dur = [
            r for r in recs
            if "start" in r and r.get("val") is not None and "filed" in r
            and str(r.get("form", "")).startswith("10-")
        ]
        # `fy`/`fp` describe the *filing*, not the period, so a later 10-K's
        # comparative columns re-report old quarters under a wrong fiscal year.
        # Key instead on (start, end) — the true period — keeping the latest-filed
        # value (restatement + PIT), then group the year-to-date ladder by its
        # shared fiscal-year `start` and difference it into discrete quarters.
        uniq: dict[tuple, dict] = {}
        for r in dur:
            k = (r["start"], r["end"])
            if k not in uniq or r["filed"] < uniq[k]["filed"]:  # earliest = original
                uniq[k] = r
        by_start: dict = {}
        for r in uniq.values():
            by_start.setdefault(r["start"], []).append(r)
        for group in by_start.values():
            group.sort(key=lambda r: pd.Timestamp(r["end"]))
            prev = 0.0
            for r in group:
                rows.append((r["end"], r["filed"], r["val"] - prev))
                prev = r["val"]

    if not rows:
        return None
    out = pd.DataFrame(rows, columns=["Report Date", "Publish Date", "val"])
    out["Report Date"] = pd.to_datetime(out["Report Date"])
    out["Publish Date"] = pd.to_datetime(out["Publish Date"])
    # A value cannot be known before its own period ends: drop malformed filing
    # dates that precede the period end (rare bad `filed` fields) to bar lookahead.
    out = out[out["Publish Date"] >= out["Report Date"]]
    if out.empty:
        return None
    # One value per period, dated to its earliest (original) publication.
    out = out.sort_values("Publish Date").drop_duplicates("Report Date", keep="first")
    return out


def _edgar_item_long(facts: dict, tags: tuple[str, ...], kind: str) -> pd.DataFrame | None:
    """Stitch a canonical item's candidate tags into one long series.

    A concept's XBRL tag can change over time (e.g. ``Revenues`` →
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` at the 2018 ASC 606
    adoption), so no single tag spans the full history. We take the first candidate
    tag as the base and fill *missing* period ends from each lower-priority tag in
    turn — giving the modern definition where available and older tags for the tail.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: pd.DataFrame | None = None
    for tag in tags:
        if tag not in gaap:
            continue
        recs = _edgar_pick_unit(gaap[tag].get("units", {}))
        if not recs:
            continue
        part = _edgar_tag_long(recs, kind)
        if part is None or part.empty:
            continue
        if merged is None:
            merged = part
        else:  # keep base tag's periods; add only ends it does not already cover
            new = part[~part["Report Date"].isin(merged["Report Date"])]
            merged = pd.concat([merged, new], ignore_index=True)
    if merged is None or merged.empty:
        return None
    return merged.sort_values("Report Date").reset_index(drop=True)


def load_edgar(
    tickers: list[str],
    items: list[str],
    calendar: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Canonical PIT fundamentals from SEC EDGAR as ``{item: [date × ticker]}``."""
    tickers = [_norm(t) for t in tickers]
    cik_map = _edgar_cik_map()
    per_item: dict[str, list[pd.DataFrame]] = {name: [] for name in items}

    for t in tickers:
        cik = cik_map.get(t)
        if cik is None:
            continue
        facts = _edgar_companyfacts(cik)
        if facts is None:
            continue
        for name in items:
            spec = CANONICAL[name]
            if not spec.edgar:
                continue
            long = _edgar_item_long(facts, spec.edgar, spec.kind)
            if long is None or long.empty:
                continue
            long.insert(0, "Ticker", t)
            per_item[name].append(long)

    out: dict[str, pd.DataFrame] = {}
    for name in items:
        parts = per_item[name]
        if not parts:
            out[name] = pd.DataFrame(np.nan, index=calendar, columns=tickers)
            continue
        long = pd.concat(parts, ignore_index=True)
        panel = _pit_panel(long, "val", CANONICAL[name].kind, calendar)
        out[name] = panel.reindex(columns=tickers)
    return out


# --- Public interface ------------------------------------------------------
def load_fundamentals(
    tickers: list[str],
    start: str,
    end: str | None = None,
    items: list[str] | None = None,
    sources: tuple[str, ...] = ("edgar", "simfin", "yfinance"),
    report_lag_days: int = 60,
    calendar: pd.DatetimeIndex | None = None,
    return_provenance: bool = False,
):
    """Canonical PIT fundamentals as ``{item: [date × ticker]}`` daily panels.

    `sources` are tried in order; the first with a non-NaN value for a given
    (date, ticker, item) wins. With `return_provenance=True` a second dict of
    ``{item: [date × ticker]}`` string masks ("simfin"/"yfinance"/"") is returned.
    """
    items = items or list(CANONICAL)
    unknown = set(items) - set(CANONICAL)
    if unknown:
        raise KeyError(f"unknown items {unknown}; known: {list(CANONICAL)}")
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if calendar is None:
        calendar = pd.bdate_range(start, end)
    tickers = [_norm(t) for t in tickers]

    backends = {
        "edgar": lambda: load_edgar(tickers, items, calendar),
        "simfin": lambda: load_simfin(tickers, items, calendar),
        "yfinance": lambda: load_yfinance(tickers, items, calendar, report_lag_days),
    }

    merged: dict[str, pd.DataFrame] = {}
    prov: dict[str, pd.DataFrame] = {}
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for src in sources:
        if src not in backends:
            raise KeyError(f"unknown source '{src}'")
        loaded[src] = backends[src]()

    for name in items:
        base = pd.DataFrame(np.nan, index=calendar, columns=tickers)
        mask = pd.DataFrame("", index=calendar, columns=tickers)
        for src in sources:
            panel = loaded[src].get(name)
            if panel is None:
                continue
            panel = panel.reindex(index=calendar, columns=tickers)
            fill = base.isna() & panel.notna()
            base = base.where(~fill, panel)
            mask = mask.where(~fill, src)
        merged[name] = base
        prov[name] = mask

    return (merged, prov) if return_provenance else merged


if __name__ == "__main__":  # smoke test: pull EDGAR for a couple of names
    tks = ["AAPL", "MSFT"]
    items = ["revenue", "net_income", "total_assets", "shares_diluted"]
    cal = pd.bdate_range("2010-01-01", pd.Timestamp.today())
    ed = load_edgar(tks, items, cal)
    for it in items:
        panel = ed[it]
        nonnull = panel.dropna(how="all")
        first = nonnull.index.min().date() if not nonnull.empty else None
        last = nonnull.index.max().date() if not nonnull.empty else None
        latest = panel.iloc[-1].to_dict()
        print(f"{it:16s} history {first}→{last}  latest {latest}")
