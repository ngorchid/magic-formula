"""Shared portfolio-construction helpers for the (long-only) Magic-Formula family.

Kept separate from any one runner so the FCF/growth/momentum study and the size-effect
study build portfolios identically. Everything works on wide ``[date × ticker]`` panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import LinearCostModel


def family_score(panels: list[pd.DataFrame], eligible: pd.DataFrame):
    """Equal-weighted mean of each panel's cross-sectional percentile rank (0..1).

    Percentile ranks make factor families comparable regardless of how many metrics
    each holds, so summing family scores weights each family equally. Returns
    ``(score, valid_mask)`` where valid requires every component present.
    """
    ranks, valid = [], None
    for p in panels:
        pe = p.where(eligible)
        ranks.append(pe.rank(axis=1, pct=True))
        v = pe.notna()
        valid = v if valid is None else (valid & v)
    return sum(ranks) / len(ranks), valid


def combine_ranks(families: list[list[pd.DataFrame]], eligible: pd.DataFrame) -> pd.DataFrame:
    """Sum equal-weighted family scores into one rank; higher = more attractive."""
    scores, valid = [], None
    for fam in families:
        s, v = family_score(fam, eligible)
        scores.append(s)
        valid = v if valid is None else (valid & v)
    return sum(scores).where(valid)


def mcap_cap(mcap: pd.DataFrame, eligible: pd.DataFrame, max_pctile: float) -> pd.DataFrame:
    """Eligibility restricted to names at/below `max_pctile` of that day's cross-sectional
    market cap — an adaptive size (small-cap) tilt."""
    pct = mcap.where(eligible).rank(axis=1, pct=True)
    return eligible & (pct <= max_pctile)


def growth_gate(eligible: pd.DataFrame, *growth_panels: pd.DataFrame,
                min_growth: float = 0.0) -> pd.DataFrame:
    """Eligibility restricted to names whose every growth panel exceeds `min_growth`.

    A *hard* health screen rather than a soft rank: a cheap-but-shrinking value trap is
    excluded outright instead of being rescued by a high value rank. NaN growth (missing
    or <1y history) fails the gate, which is the conservative choice.
    """
    gate = eligible
    for g in growth_panels:
        gate = gate & (g > min_growth)
    return gate


def size_bucket(mcap: pd.DataFrame, eligible: pd.DataFrame,
                lo: float, hi: float) -> pd.DataFrame:
    """Eligibility restricted to names whose cross-sectional market-cap percentile is in
    ``(lo, hi]`` — used to slice a universe into size tiers (e.g. 0-1/3, 1/3-2/3, 2/3-1)."""
    pct = mcap.where(eligible).rank(axis=1, pct=True)
    return eligible & (pct > lo) & (pct <= hi)


def _rebal_dates(cal: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(cal.to_series().resample(rebalance).last().dropna().values)


def weights_top_n(rank: pd.DataFrame, adj: pd.DataFrame, rebalance: str, top_n: int) -> pd.DataFrame:
    """Target weights: equal-weight the top-N by rank each rebalance (fully refreshed)."""
    cal = adj.index
    target = pd.DataFrame(np.nan, index=cal, columns=adj.columns)
    for dt in _rebal_dates(cal, rebalance):
        row = rank.loc[dt].dropna()
        if len(row) < top_n:
            continue
        w = pd.Series(0.0, index=adj.columns)
        w.loc[row.nlargest(top_n).index] = 1.0 / top_n
        target.loc[dt] = w.values
    return target.ffill().fillna(0.0).shift(1).fillna(0.0)


def weights_banded(rank: pd.DataFrame, adj: pd.DataFrame, rebalance: str,
                   top_n: int, hold_n: int, vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Target weights with a no-trade band (hysteresis): buy to fill `top_n` slots, but
    *keep* a held name until its rank falls out of the wider `hold_n` band. Cuts the
    boundary churn where a still-cheap name flips in/out over small rank wiggles.

    Held names are equal-weighted, or — if `vol` (a [date × ticker] volatility panel) is
    given — inverse-volatility weighted (∝ 1/vol), tilting size toward the calmer names."""
    cal = adj.index
    target = pd.DataFrame(np.nan, index=cal, columns=adj.columns)
    held: list[str] = []
    for dt in _rebal_dates(cal, rebalance):
        row = rank.loc[dt].dropna()
        if len(row) < top_n:
            continue
        pos = pd.Series(range(len(row)), index=row.sort_values(ascending=False).index)  # 0=best
        keep = [t for t in held if t in pos.index and pos[t] < hold_n]
        need = top_n - len(keep)
        if need > 0:
            adds = [t for t in pos.sort_values().index if t not in keep][:need]
            held = keep + adds
        else:
            held = sorted(keep, key=lambda t: pos[t])[:top_n]
        w = pd.Series(0.0, index=adj.columns)
        if vol is not None:
            iv = (1.0 / vol.loc[dt, held].where(lambda x: x > 0)).dropna()
            w.loc[iv.index] = (iv / iv.sum()).values if len(iv) else 0.0
        if vol is None or w.sum() == 0:
            w.loc[held] = 1.0 / len(held)
        target.loc[dt] = w.values
    return target.ffill().fillna(0.0).shift(1).fillna(0.0)


def pnl(weights: pd.DataFrame, adj: pd.DataFrame, volume: pd.DataFrame,
        close: pd.DataFrame, notional: float = 1_000_000.0,
        half_spread_bps: float = 2.5, impact_coef_bps: float = 10.0,
        fixed_fee: float = 0.0) -> tuple[pd.Series, float]:
    """Weights -> (net-return series after costs, annualised turnover = Σ|Δw|/yr).

    Cost bps default to a large-cap assumption; raise them (e.g. 20/30) for small caps,
    where the ADV-scaled impact term additionally penalises the least liquid names.

    ⚠ `notional` is the ASSUMED BOOK SIZE and it defaults to $1,000,000, which is 20x the
    live magic-formula sleeve ($50,000). That matters because the spread/impact terms are
    PROPORTIONAL, so they are size-invariant in bps, while real broker costs have a FIXED
    FLOOR per order (IB: $1.00 minimum per US equity order, more on European venues). At
    $1m a position is ~$33k and the floor is ~0.3bp — genuinely negligible, which is why it
    was left out. At $50k a position is $758-$3,033 after the inverse-vol tilt, where the
    same $1 floor is 3-13bp per side. Ignoring it there understates cost by an order of
    magnitude on the smallest positions.

    `fixed_fee` is dollars PER ORDER, charged on every name whose target weight changes.
    It defaults to 0.0 so existing results are unchanged; pass 1.0 (IB US equity minimum)
    together with a realistic `notional` to measure the live configuration.
    """
    adj_clean = adj.where(adj > 0)
    rets = adj_clean.pct_change(fill_method=None)
    rets = rets.where(rets.abs() < 1.0).fillna(0.0)
    gross = (weights * rets).sum(axis=1)

    dw = weights.diff().abs().fillna(weights.abs())
    adv = (close * volume).rolling(21).mean().reindex_like(weights).ffill()
    adv = adv.fillna(adv.median().median())
    cost = LinearCostModel(half_spread_bps=half_spread_bps, impact_coef_bps=impact_coef_bps)
    costs = cost.charge(dw * notional, adv) / notional

    if fixed_fee > 0:
        # One order per name whose weight actually moves. Weights are held constant between
        # rebalances (ffill), so dw is exactly zero on non-trading days and the count is the
        # real order count rather than an artefact of daily indexing.
        orders = (dw * notional > 1e-9).sum(axis=1)
        costs = costs + orders * fixed_fee / notional

    years = max((weights.index[-1] - weights.index[0]).days / 365.0, 1e-9)
    return gross - costs, float(dw.sum(axis=1).sum() / years)


def long_only_backtest(rank: pd.DataFrame, adj: pd.DataFrame, volume: pd.DataFrame,
                       close: pd.DataFrame, rebalance: str, top_n: int,
                       notional: float = 1_000_000.0) -> pd.Series:
    """Rank panel -> equal-weight top-N long-only net-return series (after a cost model)."""
    return pnl(weights_top_n(rank, adj, rebalance, top_n), adj, volume, close, notional)[0]
