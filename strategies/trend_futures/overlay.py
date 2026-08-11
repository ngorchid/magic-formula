"""Cross-asset trend overlay — the managed-futures diversifier, sketched for backtest.

Time-series-momentum across a diversified basket: each market independently long if its
trend is up, short if down, sized inverse-vol (equal risk) and the whole book vol-targeted
to ~10% annualised. The point is not standalone return but the *shape*: low correlation to
equities and positive skew (it tends to make money in sustained selloffs = "crisis alpha"),
so bolted onto a long book it cuts total drawdown.

Free ETF proxies stand in for the real CME/EUREX futures (same logic; production swaps each
row for the continuous contract and accounts for roll yield + futures financing ≈ risk-free).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import download_ohlcv
from strategies.base import StreamResult

# ETF proxy -> sector. ~16 markets, 4 sectors, all with history from ~2011.
TREND_BASKET: dict[str, str] = {
    "SPY": "equity_us", "EFA": "equity_dm_exus", "EEM": "equity_em",
    "IEF": "rates_us_10y", "TLT": "rates_us_30y",
    "LQD": "credit_ig", "HYG": "credit_hy",
    "UUP": "fx_usd", "FXE": "fx_eur", "FXY": "fx_jpy", "FXA": "fx_aud",
    "GLD": "gold", "SLV": "silver", "USO": "oil", "DBA": "agriculture", "CPER": "copper",
}


@dataclass
class TrendOverlayConfig:
    start: str = "2011-01-01"
    end: str | None = None
    basket: dict[str, str] = field(default_factory=lambda: dict(TREND_BASKET))
    lookbacks: tuple[int, ...] = (63, 126, 252)   # 3/6/12-month time-series momentum
    vol_window: int = 60                          # realised-vol lookback for sizing
    target_vol: float = 0.10                      # annualised portfolio vol target
    rebalance: str = "W-FRI"                       # weekly (trend is slow)
    max_leverage: float = 5.0
    half_spread_bps: float = 2.0


def run_trend_overlay(cfg: TrendOverlayConfig | None = None) -> StreamResult:
    cfg = cfg or TrendOverlayConfig()
    end = cfg.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = download_ohlcv(list(cfg.basket), cfg.start, end)["adj_close"]
    prices = prices.dropna(how="all", axis=1).sort_index()
    rets = prices.pct_change(fill_method=None)
    N = prices.shape[1]

    # Signal: blended time-series-momentum sign over the lookbacks, in [-1, +1].
    signal = sum(np.sign(prices / prices.shift(lb) - 1.0) for lb in cfg.lookbacks) / len(cfg.lookbacks)

    # Inverse-vol risk parity: each market targets ~target_vol/sqrt(N) standalone risk,
    # so an uncorrelated aggregate lands near target_vol before the portfolio scalar.
    vol = (rets.rolling(cfg.vol_window).std() * np.sqrt(252)).replace(0.0, np.nan)
    raw = signal * (cfg.target_vol / np.sqrt(N)) / vol
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Weekly rebalance, forward-fill, lag 1 day (no lookahead).
    rebal = prices.resample(cfg.rebalance).last().index.intersection(prices.index)
    target = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    target.loc[rebal] = raw.loc[rebal]
    weights = target.ffill().fillna(0.0).shift(1).fillna(0.0)

    # Portfolio vol target: scale so realised rolling vol ≈ target_vol.
    unscaled = (weights * rets.fillna(0.0)).sum(axis=1)
    pvol = (unscaled.rolling(cfg.vol_window).std() * np.sqrt(252)).replace(0.0, np.nan)
    scale = (cfg.target_vol / pvol).clip(upper=cfg.max_leverage).ffill().fillna(1.0)
    weights = weights.mul(scale, axis=0)
    # Gross leverage cap.
    gross = weights.abs().sum(axis=1)
    weights = weights.mul((cfg.max_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0), axis=0)

    gross_ret = (weights * rets.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().fillna(weights.abs()).sum(axis=1)
    costs = turnover * (cfg.half_spread_bps / 1e4)
    net_ret = gross_ret - costs

    diagnostics = {
        "n_markets": N,
        "avg_gross_leverage": float(weights.abs().sum(axis=1).mean()),
        "realised_vol": float(net_ret.std() * np.sqrt(252)),
        "skew": float(net_ret.skew()),
    }
    return StreamResult(
        name="trend_futures", weights=weights, gross_returns=gross_ret,
        net_returns=net_ret, turnover=turnover, costs=costs, diagnostics=diagnostics,
    )
