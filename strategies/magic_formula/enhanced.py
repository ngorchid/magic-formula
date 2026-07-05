"""The enhanced Magic Formula — the canonical "best version" from the 2026-07 research.

One place that defines the accumulated winner so it's reproducible and trade-ready, rather
than scattered across experiment scripts. What made the cut (and what didn't):

  KEPT (earned their place)
    * FCF replaces EBIT — value leg = FCF/EV (cheapness) + FCF/capital (return-on-capital);
      quality leg kept (Carlisle test: it's risk control, not dead weight).
    * Residual momentum, 12-month lookback / 1-month skip (beats raw & 6-month).
    * Growth (1yr YoY revenue + FCF) as a soft factor (small positive; persistence hurt).
    * Graham Number √(22.5·NI·Equity)/MktCap as a *second* value family — best clean-data
      Sharpe, but the most multiple-testing-fragile addition, so it's a toggle (default on).
    * Monthly rebalance (momentum needs freshness) + a 30/45 no-trade band (cuts turnover
      ~40% for the same return — the one unambiguous free win).
    * Long-only, equal-weight top-30, ex financials & utilities (Greenblatt exclusions).

  REJECTED (tested, didn't help): health screens (Piotroski / growth-gate), dropping quality,
  small-cap tilt, 6-month momentum, multi-year growth persistence.

Honest framing: clean survivorship-corrected PIT S&P 500 Sharpe ~1.0 (≈1.04 with Graham) vs
SPY 0.92 — a modest edge on a long-only β≈0.9 book, and a best-of-many number (discount for
multiple testing). See memory current-focus for the full trail.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signals import (
    fcf_ev_yield,
    fcf_growth,
    fcf_return_on_capital,
    graham_number_yield,
    residual_momentum,
    revenue_growth,
)
from strategies.magic_formula.construct import combine_ranks, weights_banded

# Fundamentals line items the enhanced signal consumes.
ENHANCED_ITEMS = [
    "revenue", "net_income", "total_equity",
    "operating_cash_flow", "capex", "shares_diluted",
    "short_term_debt", "long_term_debt", "cash",
    "total_current_assets", "total_current_liabilities", "ppe_net",
]


@dataclass
class EnhancedMagicConfig:
    exclude_sectors: tuple[str, ...] = ("Financial Services", "Utilities")
    top_n: int = 30                 # names held
    hold_n: int = 45                # no-trade band: keep a name until it drops out of top-45
    rebalance: str = "ME"           # monthly (momentum needs freshness)
    momentum_lookback: int = 252    # 12-month residual momentum
    momentum_skip: int = 21         # skip most recent month
    use_graham: bool = True         # Graham Number as a 2nd value family (best but fragile)
    weighting: str = "equal"        # "equal" | "inverse_vol" (∝ 1/63d-realised-vol)
    vol_window: int = 63            # lookback for the inverse-vol weighting


def enhanced_rank(f: dict, mcap: pd.DataFrame, adj: pd.DataFrame,
                  eligible: pd.DataFrame, cfg: EnhancedMagicConfig) -> pd.DataFrame:
    """Combined rank (higher = more attractive) from the equal-weighted factor families.

    Families (each an equal-weighted block of cross-sectional percentile ranks):
    FCF-value, [Graham], growth, momentum — so with Graham on there are 4 families at ¼ each.
    """
    fcf_value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
    growth = [revenue_growth(f), fcf_growth(f)]
    momentum = [residual_momentum(adj, lookback=cfg.momentum_lookback, skip=cfg.momentum_skip)]
    families = [fcf_value, growth, momentum]
    if cfg.use_graham:
        families.insert(1, [graham_number_yield(f, mcap)])
    return combine_ranks(families, eligible)


def enhanced_weights(f: dict, mcap: pd.DataFrame, adj: pd.DataFrame,
                     eligible: pd.DataFrame, cfg: EnhancedMagicConfig):
    """Return (target weight panel with the no-trade band applied, combined rank panel)."""
    rank = enhanced_rank(f, mcap, adj, eligible, cfg)
    vol = (adj.pct_change(fill_method=None).rolling(cfg.vol_window).std()
           if cfg.weighting == "inverse_vol" else None)
    weights = weights_banded(rank.where(eligible), adj, cfg.rebalance, cfg.top_n, cfg.hold_n, vol=vol)
    return weights, rank


def current_targets(weights: pd.DataFrame, rank: pd.DataFrame) -> pd.Series:
    """The latest target holdings (equal-weight), ordered by current rank — what to trade."""
    last = weights.iloc[-1]
    held = last[last > 0].index
    return rank.iloc[-1].reindex(held).sort_values(ascending=False)
