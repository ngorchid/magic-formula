"""The paper-trading candidate universe: broad US + European, deduped.

The ≥$500M USD size floor and sector exclusions are applied downstream at ranking time
(they need live market caps). This just assembles the ticker list. The heavy fundamentals
pull over this universe runs once a month (cached) — see scripts/run_paper.py.
"""
from __future__ import annotations

from data.universe import european_eur_tickers, sp1500_tickers, sp500_tickers


def paper_universe(broad: bool = False) -> list[str]:
    """Deduped US + European yfinance tickers.

    broad=False (default): CURATED — S&P 500 + European major-index large caps (~680
        liquid names, good data, robust monthly pull). Use this to start.
    broad=True: S&P 1500 + European (~1,700) — reaches smaller caps but the bulk yfinance
        pull is slow / throttle-prone.
    """
    us = sp1500_tickers() if broad else sp500_tickers()
    eu = european_eur_tickers()
    return sorted(set(us) | set(eu))
