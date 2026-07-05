"""Stream B — cross-asset trend following.

Per-market signal: sign of (12-month return − 1-month return), i.e. a slow trend
that ignores the very-near past. Each market is independently vol-targeted to
``target_vol_per_market`` annualised, then the portfolio is rescaled to the overall
``portfolio_vol_target``. Long/short is allowed in each market (this is the standard
CTA construction; nothing market-neutral about it at the asset-class level — it
diversifies the equity stream by being uncorrelated, not by being neutral itself).

ETF proxies are used here because they're free via yfinance. The real production
version trades CME / EUREX futures via IBKR — same logic, different instruments
and properly accounting for roll yield.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import LinearCostModel
from data import download_ohlcv
from strategies.base import StreamResult

# A pragmatic 13-market basket using the most-liquid ETF for each. Once IBKR futures
# are wired up, swap each row for the corresponding continuous contract.
DEFAULT_BASKET = {
    "SPY": "us_equity",
    "EFA": "dm_ex_us_equity",
    "EEM": "em_equity",
    "IEF": "us_7_10y",
    "TLT": "us_20y",
    "LQD": "us_ig_credit",
    "HYG": "us_hy_credit",
    "GLD": "gold",
    "SLV": "silver",
    "USO": "wti_oil",
    "UNG": "nat_gas",
    "UUP": "usd_index",
    "FXE": "eur",
}


@dataclass
class TrendConfig:
    start: str = "2015-01-01"
    end: str | None = None
    basket: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BASKET))
    fast_lookback: int = 21
    slow_lookback: int = 252
    vol_window: int = 63
    target_vol_per_market: float = 0.10
    portfolio_vol_target: float = 0.10
    rebalance: str = "ME"
    notional: float = 1_000_000.0
    half_spread_bps: float = 1.5  # ETFs are tight
    max_gross_leverage: float = 4.0


class CrossAssetTrend:
    name = "trend"

    def __init__(self, cfg: TrendConfig | None = None):
        self.cfg = cfg or TrendConfig()

    def run(self) -> StreamResult:
        cfg = self.cfg
        end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        tickers = list(cfg.basket)
        panel = download_ohlcv(tickers, cfg.start, end)
        prices = panel["adj_close"].dropna(how="all", axis=1).sort_index()
        rets = prices.pct_change(fill_method=None).fillna(0.0)

        # Per-market trend score: sign(slow_return − fast_return). Use a continuous
        # version (the difference itself, normalised by vol) so the optimizer can use
        # signal strength, not just sign.
        slow = prices / prices.shift(cfg.slow_lookback) - 1.0
        fast = prices / prices.shift(cfg.fast_lookback) - 1.0
        trend_score = (slow - fast)

        # Realised daily vol (annualised) per market — for vol targeting.
        vol = rets.rolling(cfg.vol_window).std() * np.sqrt(252)
        vol = vol.replace(0.0, np.nan)

        # Per-market position: vol-targeted, sign from trend.
        raw_pos = np.sign(trend_score) * (cfg.target_vol_per_market / vol)
        raw_pos = raw_pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Sample to rebalance frequency, then forward-fill and lag by 1 day.
        rebalance_idx = prices.resample(cfg.rebalance).last().index.intersection(prices.index)
        target = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        target.loc[rebalance_idx] = raw_pos.loc[rebalance_idx]
        weights = target.replace(0.0, np.nan).ffill().fillna(0.0).shift(1).fillna(0.0)

        # Portfolio-level vol target: scale so the rolling portfolio vol matches target.
        port_rets_unscaled = (weights * rets).sum(axis=1)
        port_vol = port_rets_unscaled.rolling(cfg.vol_window).std() * np.sqrt(252)
        scale = (cfg.portfolio_vol_target / port_vol.replace(0.0, np.nan)).clip(upper=10.0).fillna(1.0)
        weights = weights.mul(scale, axis=0)
        # Gross leverage cap.
        gross = weights.abs().sum(axis=1)
        lev_scale = (cfg.max_gross_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(lev_scale, axis=0)

        gross_ret = (weights * rets).sum(axis=1)
        dw = weights.diff().abs().fillna(weights.abs())
        turnover = dw.sum(axis=1)
        cost_model = LinearCostModel(half_spread_bps=cfg.half_spread_bps)
        # ETFs trade tight; use simple half-spread (no participation) — this is fine
        # because ETF ADV is enormous relative to a single-account book.
        costs = turnover * (cfg.half_spread_bps / 1e4)
        net_ret = gross_ret - costs

        diagnostics = {
            "n_markets": int(prices.shape[1]),
            "avg_gross_leverage": float(weights.abs().sum(axis=1).mean()),
            "realised_vol_annualised": float(net_ret.std() * np.sqrt(252)),
        }
        return StreamResult(
            name=self.name,
            weights=weights,
            gross_returns=gross_ret,
            net_returns=net_ret,
            turnover=turnover,
            costs=costs,
            diagnostics=diagnostics,
        )
