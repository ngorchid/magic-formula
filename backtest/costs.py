"""Transaction cost models.

Total cost charged on each trade = half-spread + market-impact term, both expressed
as fractions of notional. The square-root impact form is the conventional starting
point (Almgren et al. 2005 / Barra) — coefficients are placeholders that should be
calibrated to broker fills before any sizing decision relies on them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LinearCostModel:
    """Half-spread + sqrt-impact cost model.

    cost_bps = half_spread_bps + impact_coef_bps * sqrt(participation)

    where ``participation = traded_notional / adv_notional``. Returns the cost as
    a *fraction* of traded notional (not bps).
    """

    half_spread_bps: float = 2.5  # ≈ 5 bps round-trip on liquid US large caps
    impact_coef_bps: float = 10.0
    min_adv: float = 1.0  # avoid divide-by-zero when ADV unknown

    def charge(self, traded_notional: pd.DataFrame, adv_notional: pd.DataFrame) -> pd.Series:
        """Per-date total cost (fraction of book value)."""
        participation = traded_notional.abs() / adv_notional.clip(lower=self.min_adv)
        cost_bps = self.half_spread_bps + self.impact_coef_bps * np.sqrt(participation)
        per_trade_cost = traded_notional.abs() * (cost_bps / 1e4)
        return per_trade_cost.sum(axis=1)
