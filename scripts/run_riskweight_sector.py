"""Two refinements: sector-neutral momentum, and inverse-volatility (risk) position weighting.

  * Sector-neutral momentum: residual momentum strips the market but not the sector, so a stock
    can rank high just because its sector is hot. Demean momentum within sector each date to
    isolate stock-specific momentum.
  * Inverse-vol weighting: replace equal 1/30 weights with weight ∝ 1/volatility among the
    top-30 — the low-vol-anomaly idea that a calmer positive-momentum name is a better bet.

2x2 (normal vs sector-neutral momentum) x (equal vs inverse-vol weight), clean PIT S&P 500,
top-30 monthly. Value+Graham+growth held fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from data import (
    download_ohlcv,
    load_fundamentals,
    sp500_pit_eligible,
    sp500_pit_universe,
    sp500_sectors,
)
from signals import (
    fcf_ev_yield,
    fcf_growth,
    fcf_return_on_capital,
    graham_number_yield,
    residual_momentum,
    revenue_growth,
)
from strategies.magic_formula import ENHANCED_ITEMS
from strategies.magic_formula.construct import _rebal_dates, combine_ranks, pnl

EXCLUDE = ("Financial Services", "Utilities")


def _sector_neutralize(mom: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Subtract, per date, each sector's cross-sectional mean momentum from its members."""
    secs = sectors.reindex(mom.columns).fillna("Unknown")
    out = mom.copy()
    for sec in secs.unique():
        cols = secs.index[secs == sec]
        out[cols] = mom[cols].sub(mom[cols].mean(axis=1), axis=0)
    return out


def _weights(rank, adj, rebalance, top_n, vol=None):
    """Top-N target weights; equal if vol is None, else inverse-vol (∝ 1/vol, normalised)."""
    target = pd.DataFrame(np.nan, index=adj.index, columns=adj.columns)
    for dt in _rebal_dates(adj.index, rebalance):
        row = rank.loc[dt].dropna()
        if len(row) < top_n:
            continue
        picks = row.nlargest(top_n).index
        if vol is None:
            wp = pd.Series(1.0 / top_n, index=picks)
        else:
            iv = (1.0 / vol.loc[dt, picks].where(lambda x: x > 0)).dropna()
            if len(iv) < top_n // 2:
                continue
            wp = iv / iv.sum()
        full = pd.Series(0.0, index=adj.columns)  # write the FULL row (0 for non-picks)
        full.loc[wp.index] = wp.values
        target.loc[dt] = full.values
    return target.ffill().fillna(0.0).shift(1).fillna(0.0)


def main(start: str = "2012-01-01", end: str | None = None, top_n: int = 30) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sp500_pit_universe(start, end)
    full = sorted(set(tickers + ["SPY"]))
    print(f"[load] PIT S&P 500 {start}→{end} …")
    panel = download_ohlcv(full, start, end)
    adj = panel["adj_close"].dropna(how="all", axis=1)
    close = panel["close"].reindex_like(adj)
    volume = panel["volume"].reindex_like(adj)
    spy = adj["SPY"].pct_change(fill_method=None)
    adj = adj.drop(columns=["SPY"], errors="ignore")
    close = close.drop(columns=["SPY"], errors="ignore")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    sectors = sp500_sectors()
    excluded = sectors.reindex(adj.columns).isin(EXCLUDE)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
    graham = [graham_number_yield(f, mcap)]
    growth = [revenue_growth(f), fcf_growth(f)]
    mom = residual_momentum(adj, lookback=252, skip=21)
    mom_sn = _sector_neutralize(mom, sectors)
    vol = adj.pct_change(fill_method=None).rolling(63).std()  # 3-month realised vol

    print("[run] momentum-type x weighting …\n")
    rows = []
    for mname, mpanel in [("normal mom", mom), ("sector-neutral mom", mom_sn)]:
        rank = combine_ranks([value, graham, growth, [mpanel]], base)
        for wname, vpanel in [("equal-wt", None), ("inverse-vol", vol)]:
            w = _weights(rank.where(base), adj, "ME", top_n, vpanel)
            net, _ = pnl(w, adj, volume, close)
            rows.append((f"{mname}, {wname}", net))

    common = None
    for _, net in rows:
        idx = net.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    print("=" * 72)
    print(f"SECTOR-NEUTRAL MOM & INVERSE-VOL WEIGHTING  ({start}→{end}, PIT S&P 500, net)")
    print("=" * 72)
    print(f"  {'variant':30s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s}")
    for name, net in rows:
        s = summary_stats(net.reindex(common).fillna(0.0))
        print(f"  {name:30s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':30s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
          f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "riskweight_sector"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({n: net for n, net in rows}).to_csv(out / "riskweight_sector_net.csv")
    print(f"\n  wrote {out}/riskweight_sector_net.csv")


if __name__ == "__main__":
    main()
