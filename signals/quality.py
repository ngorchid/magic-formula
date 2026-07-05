"""Quality / value signal family.

Each signal consumes the canonical PIT fundamentals dict from
``data.fundamentals.load_fundamentals`` (``{item: [date × ticker]}``) and returns a
wide ``[date × ticker]`` panel, oriented so that **larger = more attractive long** —
the same convention as the price signals, so the downstream
winsorize → z-score → neutralise → combine pipeline is identical.

Flow items in the fundamentals dict are already trailing-twelve-month sums; balance
items are as-of, so the ratios below are well-posed cross-sectionally on each date.
"""
from __future__ import annotations

import pandas as pd

Fundamentals = dict[str, pd.DataFrame]


def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ratio, NaN where the denominator is missing or non-positive."""
    den = den.where(den > 0)
    return num.div(den)


def gross_profitability(f: Fundamentals) -> pd.DataFrame:
    """Novy-Marx (2013): gross profit / total assets. The single most robust quality
    factor — gross profit sits at the top of the income statement, least polluted by
    accounting discretion. Higher = better.
    """
    gp = f.get("gross_profit")
    if gp is None or gp.isna().all().all():
        gp = f["revenue"] - f["cogs"]
    return _safe_div(gp, f["total_assets"])


def return_on_equity(f: Fundamentals) -> pd.DataFrame:
    """ROE = TTM net income / total equity. Classic profitability. Higher = better."""
    return _safe_div(f["net_income"], f["total_equity"])


def accruals(f: Fundamentals) -> pd.DataFrame:
    """Sloan (1996) accruals = (net income − operating cash flow) / total assets.

    High accruals mean earnings are driven by non-cash items and tend to reverse, so
    the factor is *sign-flipped*: low-accrual (cash-backed earnings) names score high.
    """
    acc = _safe_div(f["net_income"] - f["operating_cash_flow"], f["total_assets"])
    return -acc


def earnings_yield(f: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Earnings yield E/P = TTM net income / market cap, with market cap built from
    raw (unadjusted) close × diluted shares. Higher = cheaper = more attractive long.

    `close` must be the *raw* close panel (not adjusted) so the level matches shares
    outstanding; align it to the fundamentals calendar before calling.
    """
    mcap = close.reindex_like(f["shares_diluted"]) * f["shares_diluted"]
    return _safe_div(f["net_income"], mcap)


def ebit_ev_yield(f: Fundamentals, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Greenblatt earnings yield = EBIT / Enterprise Value, capital-structure neutral.

    EV = market cap + total debt − cash. Higher = cheaper relative to operating
    earnings = more attractive. `market_cap` must align to the fundamentals calendar.
    """
    ev = (market_cap.reindex_like(f["operating_income"])
          + f["short_term_debt"].fillna(0.0)
          + f["long_term_debt"].fillna(0.0)
          - f["cash"].fillna(0.0))
    return _safe_div(f["operating_income"], ev)


def return_on_capital(f: Fundamentals) -> pd.DataFrame:
    """Greenblatt return on capital = EBIT / (net working capital + net fixed assets).

    NWC excludes excess cash and short-term debt (operating working capital only);
    net fixed assets = PP&E, net. Measures operating efficiency, leverage-independent.
    Higher = better.
    """
    nwc = ((f["total_current_assets"] - f["cash"].fillna(0.0))
           - (f["total_current_liabilities"] - f["short_term_debt"].fillna(0.0)))
    capital = nwc + f["ppe_net"]
    return _safe_div(f["operating_income"], capital)


def free_cash_flow(f: Fundamentals) -> pd.DataFrame:
    """Free cash flow = operating cash flow − capital expenditure (both TTM).

    Capex is reported as a positive outflow, so it is subtracted. FCF is harder to
    manipulate than EBIT (it nets out accruals and real reinvestment), which is why
    it is the preferred cheapness numerator in the FCF-yield Magic-Formula variant.
    """
    return f["operating_cash_flow"] - f["capex"].fillna(0.0)


def fcf_ev_yield(f: Fundamentals, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Free-cash-flow yield = FCF / Enterprise Value — the FCF analogue of Greenblatt's
    EBIT/EV. EV = market cap + total debt − cash, so it is capital-structure neutral.
    Higher = cheaper on cash the business actually throws off = more attractive.
    """
    fcf = free_cash_flow(f)
    ev = (market_cap.reindex_like(fcf)
          + f["short_term_debt"].fillna(0.0)
          + f["long_term_debt"].fillna(0.0)
          - f["cash"].fillna(0.0))
    return _safe_div(fcf, ev)


def fcf_return_on_capital(f: Fundamentals) -> pd.DataFrame:
    """Return on capital with FCF in place of EBIT = FCF / (NWC + net PP&E).

    Cash-based operating quality: rewards names that convert invested capital into
    real free cash, not just book operating earnings. Higher = better.
    """
    nwc = ((f["total_current_assets"] - f["cash"].fillna(0.0))
           - (f["total_current_liabilities"] - f["short_term_debt"].fillna(0.0)))
    capital = nwc + f["ppe_net"]
    return _safe_div(free_cash_flow(f), capital)


def graham_number_yield(f: Fundamentals, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Benjamin Graham's "Graham Number" as a cross-sectional cheapness signal.

    Graham Number (per share) = √(22.5 · EPS · BVPS) — the price ceiling implied by his
    P/E ≤ 15 and P/B ≤ 1.5 rules (15 · 1.5 = 22.5). The per-share terms cancel against
    price, so as a market-cap ratio it is √(22.5 · NetIncome · Equity) / MarketCap;
    higher = further below Graham's fair value = cheaper. The √ requires positive
    earnings *and* positive book value, so loss-makers and negative-equity firms are
    excluded outright — Graham's built-in margin-of-safety quality gate. Higher = better.
    """
    ni, eq = f["net_income"], f["total_equity"]
    pos = (ni > 0) & (eq > 0)
    graham = (22.5 * ni.where(pos) * eq.where(pos)) ** 0.5
    return _safe_div(graham, market_cap.reindex_like(graham))


# Line items the Piotroski F-score consumes (request these from load_fundamentals).
PIOTROSKI_ITEMS = [
    "net_income", "total_assets", "operating_cash_flow", "revenue", "gross_profit",
    "cogs", "long_term_debt", "total_current_assets", "total_current_liabilities",
    "shares_diluted",
]


def piotroski_f_score(f: Fundamentals, periods: int = 252) -> pd.DataFrame:
    """Piotroski (2000) F-score, 0–9 — a financial-strength screen for value stocks.

    Nine binary tests across profitability (4), leverage/liquidity (3) and operating
    efficiency (2); each passing test scores 1. High (8–9) = fundamentally strong,
    low (0–1) = weak. YoY tests compare to `periods` (~1yr) ago; a test with missing
    inputs simply scores 0. Score is NaN where the core current-year inputs are absent.
    """
    def prev(x: pd.DataFrame) -> pd.DataFrame:
        return x.shift(periods)

    ni, ta = f["net_income"], f["total_assets"]
    cfo, rev = f["operating_cash_flow"], f["revenue"]
    gp = f["gross_profit"].where(f["gross_profit"].notna(), f["revenue"] - f["cogs"])
    ltd = f["long_term_debt"].fillna(0.0)
    cr = _safe_div(f["total_current_assets"], f["total_current_liabilities"])
    sh = f["shares_diluted"]

    roa = _safe_div(ni, ta)
    lev = _safe_div(ltd, ta)
    gm = _safe_div(gp, rev)
    at = _safe_div(rev, ta)

    tests = [
        roa > 0,                        # 1. positive return on assets
        cfo > 0,                        # 2. positive operating cash flow
        roa > prev(roa),                # 3. rising ROA
        cfo > ni,                       # 4. accruals: cash-backed earnings (CFO > NI)
        lev < prev(lev),                # 5. falling leverage (LTD/assets)
        cr > prev(cr),                  # 6. rising current ratio
        sh <= prev(sh),                 # 7. no net share issuance
        gm > prev(gm),                  # 8. rising gross margin
        at > prev(at),                  # 9. rising asset turnover
    ]
    score = sum(t.astype(float) for t in tests)
    valid = ni.notna() & ta.notna() & cfo.notna() & rev.notna()
    return score.where(valid)


# Signals needing the price panel (earnings_yield, ebit_ev_yield) are wired in the stream.
QUALITY_SIGNALS = {
    "gross_profitability": gross_profitability,
    "return_on_equity": return_on_equity,
    "accruals": accruals,
}
