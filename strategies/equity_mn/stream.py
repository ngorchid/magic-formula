"""Stream A — equity market neutral.

Pipeline:
    raw signals -> winsorize -> cross-sectional z-score -> beta+sector neutralise
    -> IC-weighted combine -> L/S quintile portfolio -> backtest with cost model

Signals come from two families behind one registry:
  - price signals  (momentum, reversal)  consume the adjusted-close panel
  - quality signals (GP/A, ROE, accruals, earnings yield) consume the PIT
    fundamentals dict from `data.load_fundamentals`; earnings yield also needs the
    raw close panel for market cap.

Combination defaults to IC-weighting (`combine="ic"`): each signal is weighted by its
lagged rolling rank-IC, so weak signals are automatically down-weighted rather than
equal-blended. Set `combine="equal"` for the old mean-of-z-scores behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest import LinearCostModel, VectorizedBacktester
from combination import cs_zscore, ic_weighted_combine, winsorize
from data import download_ohlcv, load_fundamentals, sp500_sectors, sp500_tickers
from signals import (
    accruals,
    earnings_yield,
    gross_profitability,
    low_volatility,
    momentum_12_1,
    residual_momentum,
    return_on_equity,
    short_term_reversal,
)
from portfolio import run_optimized_backtest
from strategies.base import StreamResult

from .neutralize import neutralize, rolling_beta

# Each signal is a function of a context dict {prices, close, fundamentals}, so price
# and fundamentals signals share one registry and config list.
SIGNAL_REGISTRY = {
    "momentum_12_1": lambda ctx: momentum_12_1(ctx["prices"]),
    "residual_momentum": lambda ctx: residual_momentum(ctx["prices"]),
    "low_volatility": lambda ctx: low_volatility(ctx["prices"]),
    "short_term_reversal": lambda ctx: short_term_reversal(ctx["prices"]),
    "gross_profitability": lambda ctx: gross_profitability(ctx["fundamentals"]),
    "return_on_equity": lambda ctx: return_on_equity(ctx["fundamentals"]),
    "accruals": lambda ctx: accruals(ctx["fundamentals"]),
    "earnings_yield": lambda ctx: earnings_yield(ctx["fundamentals"], ctx["close"]),
}
FUNDAMENTAL_SIGNALS = {"gross_profitability", "return_on_equity", "accruals", "earnings_yield"}


@dataclass
class EquityMNConfig:
    start: str = "2015-01-01"
    end: str | None = None
    signals: list[str] = field(default_factory=lambda: ["momentum_12_1", "short_term_reversal"])
    top_quantile: float = 0.2
    rebalance: str = "ME"
    notional: float = 1_000_000.0
    benchmark: str = "SPY"
    beta_window: int = 252
    half_spread_bps: float = 2.5
    impact_coef_bps: float = 10.0
    combine: str = "ic"                 # "ic" | "equal"
    ic_weighting: str = "ic"            # "ic" (mean IC) | "icir" (mean/std — rewards stability)
    ic_window: int = 252                # rolling window for IC weights
    ic_horizon: int = 21               # forward-return horizon (also the IC lag)
    fundamental_sources: tuple[str, ...] = ("simfin",)  # yfinance batch is impractical at universe scale
    # Per-signal neutralisation (beta, sector). Default applies to unlisted signals.
    # Low-vol / BAB should skip beta-neutralisation — the beta tilt *is* the bet.
    beta_neutral_default: bool = True
    sector_neutral_default: bool = True
    neutralize_overrides: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    # Portfolio construction: "quintile" (top/bottom baskets) or "optimizer" (cvxpy).
    construction: str = "quintile"
    opt_risk_aversion: float = 8.0
    opt_turnover_cost: float = 0.0006
    opt_gross_limit: float = 1.0
    opt_max_position: float = 0.04


class EquityMarketNeutral:
    name = "equity_mn"

    def __init__(self, cfg: EquityMNConfig | None = None):
        self.cfg = cfg or EquityMNConfig()

    def run(self) -> StreamResult:
        cfg = self.cfg
        end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        unknown = set(cfg.signals) - set(SIGNAL_REGISTRY)
        if unknown:
            raise KeyError(f"unknown signals {unknown}; known: {list(SIGNAL_REGISTRY)}")

        tickers = sp500_tickers()
        # Pull benchmark alongside the universe so we can use a single cached panel.
        full_tickers = sorted(set(tickers + [cfg.benchmark]))
        panel = download_ohlcv(full_tickers, cfg.start, end)
        prices_full = panel["adj_close"].dropna(how="all", axis=1)
        volume = panel["volume"].reindex_like(prices_full)
        close_full = panel["close"].reindex_like(prices_full)
        bench = prices_full[cfg.benchmark].pct_change(fill_method=None)
        prices = prices_full.drop(columns=[cfg.benchmark], errors="ignore")
        close = close_full.drop(columns=[cfg.benchmark], errors="ignore")
        volume = volume.drop(columns=[cfg.benchmark], errors="ignore")

        # Cross-sectional beta panel against the benchmark (per ticker, rolling window).
        rets = prices.pct_change(fill_method=None)
        betas = rolling_beta(rets, bench, window=cfg.beta_window)
        sectors = sp500_sectors().reindex(prices.columns)

        # Load PIT fundamentals only if a fundamentals signal is requested.
        fundamentals: dict[str, pd.DataFrame] = {}
        if set(cfg.signals) & FUNDAMENTAL_SIGNALS:
            fundamentals = load_fundamentals(
                list(prices.columns), cfg.start, end,
                sources=cfg.fundamental_sources, calendar=prices.index,
            )
        ctx = {"prices": prices, "close": close, "fundamentals": fundamentals}

        # Build each signal -> align -> winsorize -> z-score -> (per-signal) neutralise.
        processed: dict[str, pd.DataFrame] = {}
        for s_name in cfg.signals:
            raw = SIGNAL_REGISTRY[s_name](ctx).reindex_like(prices)
            z = cs_zscore(winsorize(raw, 0.01, 0.99))
            bn, sn = cfg.neutralize_overrides.get(
                s_name, (cfg.beta_neutral_default, cfg.sector_neutral_default)
            )
            processed[s_name] = neutralize(
                z, betas=betas if bn else None, sectors=sectors if sn else None
            )

        # Combine. IC-weighting down-weights weak signals using lagged rolling rank-IC.
        ic_diag: dict[str, float] = {}
        if cfg.combine == "ic":
            fwd = prices.pct_change(cfg.ic_horizon, fill_method=None).shift(-cfg.ic_horizon)
            combined, ic_weights = ic_weighted_combine(
                processed, fwd, window=cfg.ic_window, lag=cfg.ic_horizon,
                standardize=(cfg.ic_weighting == "icir"), return_weights=True,
            )
            ic_diag = {f"ic_mean.{n}": float(ic_weights[n].mean()) for n in processed}
        else:
            stacked = pd.concat({n: p for n, p in processed.items()}, axis=1)
            combined = stacked.T.groupby(level=1).mean().T  # mean across signal level
        # Re-z-score the combined signal so downstream quantiles are comparable.
        combined = cs_zscore(combined)

        cost_model = LinearCostModel(
            half_spread_bps=cfg.half_spread_bps, impact_coef_bps=cfg.impact_coef_bps
        )
        if cfg.construction == "optimizer":
            bt_res = run_optimized_backtest(
                combined, prices, betas, sectors, volume=volume,
                rebalance=cfg.rebalance, cost_model=cost_model, notional=cfg.notional,
                risk_aversion=cfg.opt_risk_aversion, turnover_cost=cfg.opt_turnover_cost,
                gross_limit=cfg.opt_gross_limit, max_position=cfg.opt_max_position,
            )
        else:
            bt = VectorizedBacktester(
                top_quantile=cfg.top_quantile, rebalance=cfg.rebalance,
                cost_model=cost_model, notional=cfg.notional,
            )
            bt_res = bt.run(combined, prices, volume=volume)

        diagnostics = {
            "n_tickers": int(prices.shape[1]),
            "signals_used": list(processed),
            "construction": cfg.construction,
            "combine": cfg.combine,
            "avg_daily_gross_leverage": float(bt_res.weights.abs().sum(axis=1).mean()),
            "avg_daily_net_exposure": float(bt_res.weights.sum(axis=1).mean()),
            "avg_net_beta": float((bt_res.weights * betas.reindex_like(bt_res.weights)).sum(axis=1).mean()),
            **ic_diag,
        }
        return StreamResult(
            name=self.name,
            weights=bt_res.weights,
            gross_returns=bt_res.gross_returns,
            net_returns=bt_res.net_returns,
            turnover=bt_res.turnover,
            costs=bt_res.costs,
            diagnostics=diagnostics,
        )
