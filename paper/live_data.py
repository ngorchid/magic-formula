"""Build live [date × ticker] panels from yfinance for the enhanced Magic Formula.

Live ranking needs only a CURRENT snapshot of fundamentals (no deep PIT history),
plus ~1.5y of daily prices for residual momentum. Rank ratios (FCF/EV, FCF/capital,
YoY growth, momentum) are currency-neutral, so FX is applied only to the market-cap
eligibility filter, not the ranking math.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import yfinance as yf

# ENHANCED_ITEMS  ->  candidate yfinance statement row labels (first match wins).
# Statements: financials (annual income), balance_sheet, cashflow.
ITEM_LABELS: dict[str, list[str]] = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income Continuous Operations", "Net Income From Continuing Operations"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity",
                     "Total Equity Gross Minority Interest"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                            "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE",
              "Net PPE Purchase And Sale"],
    "shares_diluted": ["Diluted Average Shares", "Diluted Average Shares Outstanding",
                       "Basic Average Shares"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation",
                        "Short Term Debt"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "total_current_assets": ["Current Assets", "Total Current Assets"],
    "total_current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "ppe_net": ["Net PPE", "Net Property Plant And Equipment"],
}
# Items where yfinance reports a signed outflow but the signal wants a positive magnitude.
_ABS_ITEMS = {"capex"}
_YEAR = 252


def _first_row(df: pd.DataFrame | None, labels: list[str]) -> pd.Series | None:
    """Return the first matching row (as a date-indexed Series) from a yfinance statement."""
    if df is None or df.empty:
        return None
    idx = {str(i): i for i in df.index}
    for lab in labels:
        if lab in idx:
            s = df.loc[idx[lab]]
            s.index = pd.to_datetime(s.index, errors="coerce")
            return s[s.index.notna()].sort_index()
    return None


# Currencies yfinance quotes in a MINOR unit, mapped to (major_code, minor_per_major).
# London is the one that matters here: LSE equities report currency "GBp" and a price in
# PENCE (SHEL.L quotes ~3344, i.e. £33.44). The trap is that Yahoo's FX endpoint is
# case-insensitive, so "GBpUSD=X" resolves to the POUND rate rather than failing loudly --
# measured 2026-08-28, both "GBpUSD=X" and "GBPUSD=X" returned 1.3528. Without this table a
# UK name's market cap comes out 100x too large, which does not just mis-size it: mcap feeds
# the size floor and the eligibility screen, so every LSE name would sail past the large-cap
# filter and dominate the ranking. A silent 100x is far more dangerous than a NaN.
_MINOR_UNITS = {"GBp": ("GBP", 100.0), "ZAc": ("ZAR", 100.0), "ILA": ("ILS", 100.0)}


def _fx_to_usd(currencies: set[str]) -> dict[str, float]:
    """Spot FX so that value_usd = value_local * fx[local_ccy]. USD -> 1.0.

    Minor-unit quotes (pence, cents) are converted at the major rate divided by the
    subdivision, so the returned factor is always per ONE UNIT AS QUOTED BY YFINANCE.
    """
    fx = {"USD": 1.0}
    for ccy in currencies:
        if ccy in fx or not ccy:
            continue
        major, div = _MINOR_UNITS.get(ccy, (ccy, 1.0))
        try:
            # yfinance quotes e.g. EURUSD=X = USD per 1 EUR
            px = yf.Ticker(f"{major}USD=X").fast_info.get("lastPrice")
            fx[ccy] = float(px) / div if px else np.nan
        except Exception:
            fx[ccy] = np.nan
    return fx


def fetch_live_panels(tickers: list[str], price_days: int = 500, pause: float = 0.0):
    """Return dict with daily-calendar panels ready for enhanced_rank:
        adj    [date × ticker]  local-currency adjusted close (rank momentum is ccy-neutral)
        mcap   [date × ticker]  market cap in USD (for the size filter)
        f      {item: [date × ticker]}  fundamentals, ffilled onto the daily calendar
        sector {ticker: sector}, currency {ticker: ccy}, mcap_usd {ticker: latest USD mcap}
    Tickers that fail (no data) are dropped.
    """
    # ---- daily prices (batch) ----
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=price_days)
    raw = yf.download(tickers, start=start, end=end + pd.Timedelta(days=1),
                      auto_adjust=True, progress=False, threads=True)
    adj = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(tickers[0])
    adj = adj.dropna(how="all", axis=1).dropna(how="all")
    # US+EU batched -> union calendar with holiday holes; ffill so each ticker's last row
    # carries its latest known price (fixes mcap/EV/eligibility on cross-market off-days).
    adj = adj.ffill()
    cal = adj.index  # the daily calendar everything aligns to
    live = [t for t in tickers if t in adj.columns]

    # ---- per-ticker fundamentals + meta ----
    ann: dict[str, dict[str, pd.Series]] = {it: {} for it in ITEM_LABELS}
    sector: dict[str, str] = {}
    currency: dict[str, str] = {}      # QUOTE currency, exactly as yfinance reports it
    fin_ccy: dict[str, str] = {}       # REPORTING currency of the financial statements
    mcap_now: dict[str, float] = {}
    shares_now: dict[str, float] = {}
    debt_now: dict[str, float] = {}
    cash_now: dict[str, float] = {}
    for t in live:
        try:
            tk = yf.Ticker(t)
            inc, bs, cf = tk.financials, tk.balance_sheet, tk.cashflow
            info = tk.get_info()
            sector[t] = info.get("sector") or ""
            # TWO DISTINCT CURRENCIES, and conflating them was a live bug until 2026-08-28.
            # `currency` was taken from `financialCurrency` and then applied to price-derived
            # quantities, but those are two different things for a large minority of names:
            # SHEL/AZN/HSBA/BP/EQNR/ABI all REPORT in USD while QUOTING in GBp/NOK/EUR
            # (measured: 7% of the European universe). The `.upper()` did further damage by
            # folding "GBp" into "GBP", discarding the factor of 100 for all 100 FTSE names.
            #   currency (quote)  -> prices, market cap, position marking
            #   fin_ccy (report)  -> revenue, cash flow, debt, equity, every statement item
            # Everything is converted to USD below so ratios can never mix the two.
            try:
                quote = tk.fast_info.get("currency")
            except Exception:  # noqa: BLE001 - fast_info is a network call like any other
                quote = None
            currency[t] = quote or info.get("currency") or "USD"   # case PRESERVED ("GBp")
            fin_ccy[t] = (info.get("financialCurrency") or currency[t]).upper()
            mcap_now[t] = info.get("marketCap") or np.nan
            shares_now[t] = info.get("sharesOutstanding") or np.nan
            debt_now[t] = info.get("totalDebt") or np.nan
            cash_now[t] = info.get("totalCash") or np.nan
            for it, labels in ITEM_LABELS.items():
                src = cf if it in ("operating_cash_flow", "capex") else \
                      inc if it in ("revenue", "net_income", "shares_diluted") else bs
                s = _first_row(src, labels)
                if s is not None and len(s):
                    ann[it][t] = s.abs() if it in _ABS_ITEMS else s
        except Exception:
            pass
        if pause:
            time.sleep(pause)

    # ---- assemble daily-ffilled fundamental panels aligned to `cal` ----
    def to_daily(series_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
        cols = {}
        for t, s in series_by_ticker.items():
            merged = s.reindex(s.index.union(cal)).ffill().reindex(cal)
            cols[t] = merged
        return pd.DataFrame(cols, index=cal) if cols else pd.DataFrame(index=cal)

    f = {it: to_daily(ann[it]).reindex(columns=live) for it in ITEM_LABELS}

    # shares fallback from info if the diluted-shares statement row is missing
    sh = f["shares_diluted"]
    for t in live:
        if (t not in sh.columns or sh[t].dropna().empty) and not np.isnan(shares_now.get(t, np.nan)):
            sh[t] = shares_now[t]
    f["shares_diluted"] = sh
    # debt/cash fallback from info (single point, broadcast) where statement missing
    for it, now in (("short_term_debt", None), ("long_term_debt", debt_now), ("cash", cash_now)):
        if now is None:
            continue
        panel = f[it]
        for t in live:
            if t not in panel.columns or panel[t].dropna().empty:
                panel[t] = now.get(t, np.nan)
        f[it] = panel

    # ---- convert EVERYTHING to USD -------------------------------------------------
    # Statement items and price-derived items are converted with DIFFERENT rates, because
    # they are denominated in different currencies (see the note in the meta loop). Once both
    # are in USD, EV = mcap + debt - cash and every value ratio is currency-consistent by
    # construction, for every name, instead of relying on quote == report happening to hold.
    fx = _fx_to_usd(set(currency.values()))       # USD per unit AS QUOTED (pence-aware)
    fx_fin = _fx_to_usd(set(fin_ccy.values()))    # USD per unit of the REPORTING currency

    # shares_diluted is a COUNT, not money — converting it would be a silent FX error in
    # every per-share quantity that uses it.
    for it in ITEM_LABELS:
        if it == "shares_diluted":
            continue
        panel = f[it]
        for t in live:
            if t in panel.columns:
                panel[t] = panel[t] * fx_fin.get(fin_ccy.get(t, "USD"), np.nan)
        f[it] = panel

    # ---- market cap panels (USD) ----
    mcap_usd_panel = f["shares_diluted"] * adj.reindex(columns=live)   # still as-quoted here
    for t in live:
        mcap_usd_panel[t] = mcap_usd_panel[t] * fx.get(currency.get(t, "USD"), np.nan)

    # `info["marketCap"]` is denominated in the MAJOR unit of the quote currency, not the
    # quoted unit — verified 2026-08-28 across all four UK names, where marketCap / (price x
    # shares) came out at exactly 0.0100 while every non-pence venue gave 1.0000. So it needs
    # the major-unit rate, which is the quoted rate multiplied back up by the subdivision.
    mcap_usd_latest = {}
    for t in live:
        ccy = currency.get(t, "USD")
        rate = fx.get(ccy, np.nan) * _MINOR_UNITS.get(ccy, (ccy, 1.0))[1]
        m = mcap_now.get(t, np.nan)
        mcap_usd_latest[t] = (m * rate) if not np.isnan(m) \
            else mcap_usd_panel[t].dropna().iloc[-1] if mcap_usd_panel[t].notna().any() else np.nan

    return {
        "adj": adj.reindex(columns=live),
        "mcap": mcap_usd_panel,          # USD — size filter, eligibility AND the value ratios
        "f": f,
        "sector": sector,
        "currency": currency,            # QUOTE ccy, case preserved — "GBp" means pence
        "fin_ccy": fin_ccy,              # REPORTING ccy; kept for diagnostics/reconciliation
        "mcap_usd_latest": mcap_usd_latest,
        "calendar": cal,
    }
