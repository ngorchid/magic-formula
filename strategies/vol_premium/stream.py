"""Stream C — variance / volatility risk premium (stub).

Intended implementation (when options data is wired up):

  Primary leg: short ATM SPX straddle, delta-hedged daily. Sizes the short to keep
    vega exposure constant. Earnings the gap between implied (IV) and realised (RV)
    variance.

  Hedge leg: long 5-10% notional in 30-delta OTM SPX put options. Caps the left tail.
    This is the part that turns "naked short vol" into "harvesting VRP with insurance."

  Dispersion overlay (later): short SPX vol against a basket of long single-name
    vol on the top components — pays when correlation drops.

Why this is stubbed:
  - Free options data (yfinance) is unreliable for backtest-grade IV surfaces.
  - The strategy without the hedge is the canonical "picking up nickels in front of
    a steamroller" trade — Feb-2018 and Mar-2020 wiped out unhedged short-vol funds.
    Better to leave a clean interface and implement properly once we have ORATS or
    IVolatility data (~$100-200/mo).

Disabled in the default config; enabling it currently raises NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass

from strategies.base import StreamResult


@dataclass
class VolPremiumConfig:
    start: str = "2015-01-01"
    end: str | None = None
    notional: float = 1_000_000.0


class VolRiskPremium:
    name = "vol_premium"

    def __init__(self, cfg: VolPremiumConfig | None = None):
        self.cfg = cfg or VolPremiumConfig()

    def run(self) -> StreamResult:
        raise NotImplementedError(
            "vol_premium stream is a stub — implement once an IV-surface data source "
            "(ORATS, IVolatility, or CBOE EOD) is wired up. See docstring for design."
        )
