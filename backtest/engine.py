"""Vectorized cross-sectional long-short backtester.

The model: at each rebalance date the engine takes a `[date × ticker]` score panel,
goes long the top quantile and short the bottom quantile, equally weighted within
each leg, and dollar-neutral across legs (sum of weights = 0, gross = 1).
Returns from t to t+1 are realised against weights set at end-of-t.

`notional` is the assumed book size in dollars. It only affects the cost model
(participation = traded_dollars / ADV_dollars) — returns are reported as fractions
of the book regardless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import LinearCostModel


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series


class VectorizedBacktester:
    def __init__(
        self,
        top_quantile: float = 0.2,
        rebalance: str = "ME",
        cost_model: LinearCostModel | None = None,
        notional: float = 1_000_000.0,
    ):
        if not 0 < top_quantile < 0.5:
            raise ValueError("top_quantile must be in (0, 0.5)")
        self.top_quantile = top_quantile
        self.rebalance = rebalance
        self.cost_model = cost_model or LinearCostModel()
        self.notional = float(notional)

    @staticmethod
    def _quantile_weights(scores_row: pd.Series, q: float) -> pd.Series:
        s = scores_row.dropna()
        if len(s) < 10:
            return pd.Series(0.0, index=scores_row.index)
        n_side = max(int(len(s) * q), 1)
        ranked = s.sort_values()
        shorts = ranked.index[:n_side]
        longs = ranked.index[-n_side:]
        w = pd.Series(0.0, index=scores_row.index)
        w.loc[longs] = 0.5 / n_side
        w.loc[shorts] = -0.5 / n_side
        return w

    def run(
        self,
        scores: pd.DataFrame,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
    ) -> BacktestResult:
        prices, scores = prices.align(scores, join="inner", axis=0)
        prices = prices.sort_index()
        scores = scores.sort_index()
        rets = prices.pct_change(fill_method=None).fillna(0.0)

        # Build target weights only on rebalance dates (rows left NaN otherwise), then
        # ffill row-wise so the whole previous portfolio carries until the next rebalance.
        # Important: setting non-rebalance rows to 0 and then ffill-on-NaN-only would
        # leak stale weights for names that have dropped out of the current quintile.
        rebalance_dates = prices.resample(self.rebalance).last().index.intersection(prices.index)
        target = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for dt in rebalance_dates:
            if dt in scores.index:
                target.loc[dt] = self._quantile_weights(scores.loc[dt], self.top_quantile).values
        weights = target.ffill().fillna(0.0)
        # Lag by one day: signal at t -> position from t+1. Avoids lookahead.
        weights = weights.shift(1).fillna(0.0)

        gross = (weights * rets).sum(axis=1)

        # Costs: scale Δw by the book size to get dollar-traded; ADV is dollar volume.
        dw = weights.diff().abs().fillna(weights.abs())
        turnover = dw.sum(axis=1)
        if volume is not None:
            adv_dollars = (prices * volume).rolling(21).mean().reindex_like(weights)
            adv_dollars = adv_dollars.ffill().fillna(adv_dollars.median().median())
            traded_dollars = dw * self.notional
            cost_dollars = self.cost_model.charge(traded_dollars, adv_dollars)
            costs = cost_dollars / self.notional
        else:
            costs = turnover * (self.cost_model.half_spread_bps / 1e4)

        net = gross - costs

        return BacktestResult(
            weights=weights,
            gross_returns=gross,
            net_returns=net,
            turnover=turnover,
            costs=costs,
        )
