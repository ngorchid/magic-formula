"""The paper-trading candidate universe: broad US + European, deduped.

The ≥$500M USD size floor and sector exclusions are applied downstream at ranking time
(they need live market caps). This just assembles the ticker list. The heavy fundamentals
pull over this universe runs once a month (cached) — see scripts/run_paper.py.
"""
from __future__ import annotations

from data.universe import (european_eur_tickers, european_non_eur_tickers,
                           sp1500_tickers, sp500_tickers)


def paper_universe(broad: bool = False, include_non_eur: bool = True) -> list[str]:
    """Deduped US + European yfinance tickers.

    broad=False (default): CURATED — S&P 500 + European major-index large caps (~940
        liquid names, good data, robust monthly pull). Use this to start.
    broad=True: S&P 1500 + European (~1,960) — reaches smaller caps but the bulk yfinance
        pull is slow / throttle-prone.

    include_non_eur=True (default, set 2026-08-28) adds non-eurozone Europe — UK, Swiss and
    Nordic large caps, ~200 names. These are quoted in GBp/CHF/SEK/DKK/NOK and held UNHEDGED,
    so their FX moves land straight in the P&L; that contribution was measured at 0.02-0.19pp
    of portfolio vol, which is not a reason to exclude a company. The reason to INCLUDE them
    is that they are a genuinely different bet: 0.60-0.87 correlation to the current EUR book
    against 0.96-0.97 among the eurozone indices themselves. Flip to False to revert to
    eurozone-only without touching the universe module.
    """
    us = sp1500_tickers() if broad else sp500_tickers()
    eu = set(european_eur_tickers())
    if include_non_eur:
        eu |= set(european_non_eur_tickers())
    return sorted(set(us) | eu)
