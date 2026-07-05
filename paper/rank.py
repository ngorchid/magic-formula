"""Today's target ranking for the live book — reuses the validated `enhanced_rank`.

Feeds live yfinance panels into the backtest's ranking logic and returns the current
eligible, ranked candidate list. Staggered entry / no-trade band / sizing all live in
the orchestrator; this module only answers "what does the formula like right now?".
"""
from __future__ import annotations

import pandas as pd

from strategies.magic_formula import EnhancedMagicConfig, enhanced_rank


def build_eligibility(panels: dict, cfg: EnhancedMagicConfig,
                      min_mcap_usd: float = 500e6) -> pd.DataFrame:
    """[date × ticker] bool: tradeable size (≥ min USD mcap) and not an excluded sector."""
    adj = panels["adj"]
    mcap_usd = panels["mcap"].reindex_like(adj)
    sector = pd.Series(panels["sector"]).reindex(adj.columns)
    sector_ok = ~sector.isin(cfg.exclude_sectors)
    elig = pd.DataFrame(sector_ok.values[None, :].repeat(len(adj), axis=0),
                        index=adj.index, columns=adj.columns)
    return elig & (mcap_usd >= min_mcap_usd)


def todays_ranking(panels: dict, cfg: EnhancedMagicConfig,
                   min_mcap_usd: float = 500e6) -> pd.Series:
    """Series (ticker -> rank score, higher = more attractive) for TODAY, eligible only."""
    elig = build_eligibility(panels, cfg, min_mcap_usd)
    # Value ratios must use LOCAL-currency mcap so numerator/denominator currencies match.
    rank = enhanced_rank(panels["f"], panels["mcap_local"], panels["adj"], elig, cfg)
    today = rank.iloc[-1].dropna().sort_values(ascending=False)
    return today


def rank_report(panels: dict, cfg: EnhancedMagicConfig, top_n: int = 30,
                min_mcap_usd: float = 500e6) -> pd.DataFrame:
    """Human-readable top-N table: rank score, USD market cap, sector, currency."""
    r = todays_ranking(panels, cfg, min_mcap_usd)
    mcap = panels["mcap_usd_latest"]
    sector = panels["sector"]
    ccy = panels["currency"]
    rows = [{"ticker": t, "rank": round(float(v), 3),
             "mcap_usd_bn": round(mcap.get(t, float("nan")) / 1e9, 2),
             "sector": sector.get(t, ""), "ccy": ccy.get(t, "")}
            for t, v in r.head(top_n).items()]
    return pd.DataFrame(rows)
