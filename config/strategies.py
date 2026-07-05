"""Strategy configuration.

Plain Python so it's easy to edit, IDE-checkable, and doesn't need a YAML/TOML
parsing layer. Toggle streams on/off via `enabled`; toggle signals within a stream
via the per-stream `signals` list.
"""
from __future__ import annotations

CONFIG = {
    "start": "2015-01-01",
    "end": None,  # None = today
    "notional": 1_000_000.0,
    "portfolio_vol_target": 0.08,
    "streams": {
        "equity_mn": {
            "enabled": True,
            # Price-based only. Fundamentals (quality/value) shelved — free SimFin is
            # too shallow (~5y) to evaluate them. residual_momentum is the clean win
            # (Sharpe 0.21->0.29, better drawdown); low_volatility doesn't work in this
            # large-cap/regime (IC stays negative), so it's left out.
            "signals": [
                "momentum_12_1",
                "short_term_reversal",
                "residual_momentum",
            ],
            "top_quantile": 0.2,
            "rebalance": "ME",
            "benchmark": "SPY",
            "beta_window": 252,
            "half_spread_bps": 2.5,
            "impact_coef_bps": 10.0,
            "combine": "ic",          # "ic" = IC-weighted (down-weights weak signals); "equal" = mean
            "ic_window": 252,
            "ic_horizon": 21,
            "fundamental_sources": ("simfin",),
        },
        "trend": {
            "enabled": True,
            "basket": None,  # None = use DEFAULT_BASKET from the module
            "fast_lookback": 21,
            "slow_lookback": 252,
            "vol_window": 63,
            "target_vol_per_market": 0.10,
            "portfolio_vol_target": 0.10,
            "rebalance": "ME",
            "max_gross_leverage": 4.0,
            "half_spread_bps": 1.5,
        },
        "vol_premium": {
            "enabled": False,  # stub — requires options data
        },
    },
}


def build_streams(cfg: dict | None = None):
    """Instantiate Stream objects per config. Skips disabled streams."""
    from strategies.equity_mn import EquityMarketNeutral
    from strategies.equity_mn.stream import EquityMNConfig
    from strategies.trend import CrossAssetTrend, TrendConfig
    from strategies.vol_premium import VolPremiumConfig, VolRiskPremium

    cfg = cfg or CONFIG
    streams = []

    s = cfg["streams"]["equity_mn"]
    if s.get("enabled", False):
        streams.append(
            EquityMarketNeutral(
                EquityMNConfig(
                    start=cfg["start"],
                    end=cfg["end"],
                    signals=s["signals"],
                    top_quantile=s["top_quantile"],
                    rebalance=s["rebalance"],
                    benchmark=s["benchmark"],
                    beta_window=s["beta_window"],
                    half_spread_bps=s["half_spread_bps"],
                    impact_coef_bps=s["impact_coef_bps"],
                    notional=cfg["notional"],
                    combine=s.get("combine", "ic"),
                    ic_window=s.get("ic_window", 252),
                    ic_horizon=s.get("ic_horizon", 21),
                    fundamental_sources=s.get("fundamental_sources", ("simfin",)),
                )
            )
        )

    s = cfg["streams"]["trend"]
    if s.get("enabled", False):
        kwargs = dict(
            start=cfg["start"],
            end=cfg["end"],
            fast_lookback=s["fast_lookback"],
            slow_lookback=s["slow_lookback"],
            vol_window=s["vol_window"],
            target_vol_per_market=s["target_vol_per_market"],
            portfolio_vol_target=s["portfolio_vol_target"],
            rebalance=s["rebalance"],
            max_gross_leverage=s["max_gross_leverage"],
            half_spread_bps=s["half_spread_bps"],
            notional=cfg["notional"],
        )
        if s.get("basket") is not None:
            kwargs["basket"] = s["basket"]
        streams.append(CrossAssetTrend(TrendConfig(**kwargs)))

    s = cfg["streams"]["vol_premium"]
    if s.get("enabled", False):
        streams.append(
            VolRiskPremium(VolPremiumConfig(start=cfg["start"], end=cfg["end"], notional=cfg["notional"]))
        )

    return streams
