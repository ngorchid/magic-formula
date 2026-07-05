"""Stream D — Magic Formula (Greenblatt), long-only.

Rank the broad US universe (ex financials/utilities, above a market-cap floor) on two
metrics and buy the best *combined* names, equal-weighted, held ~1 year:

    earnings yield  = EBIT / Enterprise Value     (cheapness, capital-structure neutral)
    return on capital = EBIT / (NWC + net PP&E)   (operating quality)

Selection is by **rank sum** (not z-scores): each name's rank on each metric is added,
the highest combined rank wins. Ranks are robust to the fat tails in ROC/EV ratios.

Unlike the market-neutral equity stream this is **long-only and fully invested** — it
deliberately keeps market exposure (≈beta 1) to harvest the equity risk premium, so it
is benchmarked directly against SPY. Free SimFin prices cap the window at ~5 years.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import LinearCostModel
from data import broad_universe, load_fundamentals
from signals import ebit_ev_yield, return_on_capital
from strategies.base import StreamResult

_ITEMS = [
    "operating_income", "short_term_debt", "long_term_debt", "cash",
    "total_current_assets", "total_current_liabilities", "ppe_net",
]


@dataclass
class MagicFormulaConfig:
    start: str = "2020-08-01"
    end: str | None = None
    top_n: int = 30
    rebalance: str = "YE"               # annual (calendar year-end); classic Greenblatt
    min_market_cap: float = 3e8
    exclude_sectors: tuple[str, ...] = ("Financial Services", "Utilities")
    notional: float = 1_000_000.0
    half_spread_bps: float = 5.0        # broad/smaller-cap names trade wider than large caps
    impact_coef_bps: float = 20.0


class MagicFormula:
    name = "magic_formula"

    def __init__(self, cfg: MagicFormulaConfig | None = None):
        self.cfg = cfg or MagicFormulaConfig()

    def run(self) -> StreamResult:
        cfg = self.cfg
        tickers, eligible, panels = broad_universe(
            cfg.start, cfg.end, min_market_cap=cfg.min_market_cap,
            exclude_sectors=cfg.exclude_sectors,
        )
        adj = panels["adj_close"]
        close, shares, volume = panels["close"], panels["shares"], panels["volume"]
        cal = adj.index
        mcap = close * shares.reindex_like(close)

        f = load_fundamentals(tickers, cfg.start, cfg.end, items=_ITEMS,
                              sources=("simfin",), calendar=cal)
        ey = ebit_ev_yield(f, mcap).where(eligible)
        roc = return_on_capital(f).where(eligible)

        # Greenblatt rank-sum: higher metric -> higher rank; require both present.
        valid = ey.notna() & roc.notna()
        combined = (ey.rank(axis=1) + roc.rank(axis=1)).where(valid)

        # Annual rebalance: equal-weight the top-N, long-only, fully invested.
        # Use the actual last trading day of each period (period-end calendar dates
        # like Dec-31 are often non-trading and would be silently dropped).
        rebal = pd.DatetimeIndex(cal.to_series().resample(cfg.rebalance).last().dropna().values)
        target = pd.DataFrame(np.nan, index=cal, columns=tickers)
        for dt in rebal:
            row = combined.loc[dt].dropna()
            if len(row) < cfg.top_n:
                continue
            picks = row.nlargest(cfg.top_n).index
            w = pd.Series(0.0, index=tickers)
            w.loc[picks] = 1.0 / cfg.top_n
            target.loc[dt] = w.values
        weights = target.ffill().fillna(0.0).shift(1).fillna(0.0)

        # Clean prices (non-positive -> NaN) and guard against free-data glitch ticks:
        # a >100%/day single-name move is treated as a data error, not P&L.
        adj_clean = adj.where(adj > 0)
        rets = adj_clean.pct_change(fill_method=None)
        rets = rets.where(rets.abs() < 1.0).fillna(0.0)
        gross = (weights * rets).sum(axis=1)

        dw = weights.diff().abs().fillna(weights.abs())
        turnover = dw.sum(axis=1)
        adv = (close * volume).rolling(21).mean().reindex_like(weights)
        adv = adv.ffill().fillna(adv.median().median())
        cost_model = LinearCostModel(half_spread_bps=cfg.half_spread_bps,
                                     impact_coef_bps=cfg.impact_coef_bps)
        costs = cost_model.charge(dw * cfg.notional, adv) / cfg.notional
        net = gross - costs

        held = (weights > 0).sum(axis=1)
        diagnostics = {
            "n_universe": len(tickers),
            "avg_holdings": float(held[held > 0].mean()) if (held > 0).any() else 0.0,
            "n_rebalances": int(len(rebal)),
            "avg_gross_leverage": float(weights.sum(axis=1).mean()),
            "annual_turnover": float(turnover.sum() / max((cal[-1] - cal[0]).days / 365.0, 1e-9)),
        }
        return StreamResult(
            name=self.name,
            weights=weights,
            gross_returns=gross,
            net_returns=net,
            turnover=turnover,
            costs=costs,
            diagnostics=diagnostics,
        )
