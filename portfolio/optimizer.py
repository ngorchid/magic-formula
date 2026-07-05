"""cvxpy mean-variance optimizer with risk neutrality and turnover penalty.

Single-period optimisation, solved each rebalance date:

    maximise   alphaᵀw  −  (λ/2)·wᵀΣw  −  γ·‖w − w_prev‖₁
       w
    s.t.       1ᵀw = 0                 dollar-neutral
               βᵀw = 0                 beta-neutral (exact)
               S_kᵀw = 0  ∀ sector k   sector-neutral (exact)
               ‖w‖₁ ≤ L                gross leverage
               |wᵢ| ≤ w_max            position limit

`alpha` is scaled to (holding-period) expected-return units via the Grinold rule
``alpha = IC · σ · z`` so λ and γ are interpretable: λ is risk aversion against the
covariance Σ, and γ is the round-trip trading cost (so the L1 term creates no-trade
bands — names only move when the alpha change beats the cost of trading).

Σ uses Ledoit-Wolf shrinkage (sample covariance is rank-deficient at N≫T); exact
beta/sector neutrality is imposed as constraints rather than baked into Σ. A full
Barra-style factor covariance is the natural v2.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from backtest.costs import LinearCostModel
from backtest.engine import BacktestResult

_SOLVERS = ("CLARABEL", "OSQP", "SCS")


def factor_neutral_optimize(
    alpha: pd.Series,            # expected alpha per name (index = tickers)
    cov: pd.DataFrame,           # covariance [names × names], aligned to alpha.index
    beta: pd.Series,             # market beta per name
    sectors: pd.Series,          # ticker -> sector label
    w_prev: pd.Series,           # previously held weights
    risk_aversion: float = 8.0,
    turnover_cost: float = 0.0006,
    gross_limit: float = 1.0,
    max_position: float = 0.04,
    sector_neutral: bool = True,
) -> pd.Series | None:
    """Solve the single-date problem. Returns weights over `alpha.index`, or None if
    the solve fails (caller should then hold the previous book)."""
    names = list(alpha.index)
    n = len(names)
    a = alpha.values
    S = cov.loc[names, names].values
    S = 0.5 * (S + S.T)  # symmetrise against float error
    b = beta.reindex(names).fillna(0.0).values
    wp = w_prev.reindex(names).fillna(0.0).values

    w = cp.Variable(n)
    objective = a @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(S)) \
        - turnover_cost * cp.norm1(w - wp)
    cons = [cp.sum(w) == 0, b @ w == 0, cp.norm1(w) <= gross_limit, cp.abs(w) <= max_position]
    if sector_neutral:
        sec = sectors.reindex(names)
        for label in sec.dropna().unique():
            mask = (sec == label).values.astype(float)
            cons.append(mask @ w == 0)

    prob = cp.Problem(cp.Maximize(objective), cons)
    for solver in _SOLVERS:
        try:
            prob.solve(solver=solver)
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                return pd.Series(np.asarray(w.value).ravel(), index=names)
        except (cp.error.SolverError, Exception):
            continue
    return None


def run_optimized_backtest(
    alpha: pd.DataFrame,         # combined score panel [date × ticker]
    prices: pd.DataFrame,        # adjusted close [date × ticker]
    betas: pd.DataFrame,         # rolling beta [date × ticker]
    sectors: pd.Series,          # ticker -> sector
    volume: pd.DataFrame | None = None,
    *,
    rebalance: str = "ME",
    cost_model: LinearCostModel | None = None,
    notional: float = 1_000_000.0,
    cov_window: int = 252,
    vol_window: int = 63,
    ic: float = 0.03,
    hold_days: int = 21,
    risk_aversion: float = 8.0,
    turnover_cost: float = 0.0006,
    gross_limit: float = 1.0,
    max_position: float = 0.04,
    sector_neutral: bool = True,
) -> BacktestResult:
    """Backtest the optimizer construction: solve weights each rebalance date, hold
    between, lag one day, charge costs. Mirrors VectorizedBacktester's outputs so the
    two construction methods are directly comparable on the same alpha."""
    prices, alpha = prices.align(alpha, join="inner", axis=0)
    prices, alpha = prices.sort_index(), alpha.sort_index()
    cost_model = cost_model or LinearCostModel()

    rets = prices.pct_change(fill_method=None).fillna(0.0)
    sig = rets.rolling(vol_window, min_periods=max(vol_window // 2, 20)).std()
    rebal_dates = prices.resample(rebalance).last().index.intersection(prices.index)

    target = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    w_prev = pd.Series(0.0, index=prices.columns)
    for dt in rebal_dates:
        a_row = alpha.loc[dt].dropna()
        sec_ok = sectors.reindex(a_row.index).dropna().index
        names = a_row.index.intersection(sec_ok)
        if len(names) < 20:
            continue
        win = rets.loc[:dt, names].tail(cov_window).dropna(axis=1)
        names = win.columns
        if len(names) < 20:
            continue
        # Ledoit-Wolf daily covariance, scaled to the holding period.
        cov = pd.DataFrame(LedoitWolf().fit(win.values).covariance_ * hold_days,
                           index=names, columns=names)
        # Grinold alpha scaling -> holding-period expected return units.
        a_scaled = ic * sig.loc[dt, names].fillna(sig.loc[dt, names].median()) * a_row[names] * hold_days
        w = factor_neutral_optimize(
            a_scaled, cov, betas.loc[dt], sectors, w_prev,
            risk_aversion=risk_aversion, turnover_cost=turnover_cost,
            gross_limit=gross_limit, max_position=max_position, sector_neutral=sector_neutral,
        )
        if w is None:
            continue
        full = w.reindex(prices.columns).fillna(0.0)
        target.loc[dt] = full.values
        w_prev = full

    weights = target.ffill().fillna(0.0).shift(1).fillna(0.0)
    gross = (weights * rets).sum(axis=1)

    dw = weights.diff().abs().fillna(weights.abs())
    turnover = dw.sum(axis=1)
    if volume is not None:
        adv = (prices * volume).rolling(21).mean().reindex_like(weights)
        adv = adv.ffill().fillna(adv.median().median())
        costs = cost_model.charge(dw * notional, adv) / notional
    else:
        costs = turnover * (cost_model.half_spread_bps / 1e4)
    net = gross - costs

    return BacktestResult(weights=weights, gross_returns=gross, net_returns=net,
                          turnover=turnover, costs=costs)
