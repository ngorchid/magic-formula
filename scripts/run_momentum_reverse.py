"""Reverse the momentum leg (contrarian) + rebalance-frequency sweep.

Hypothesis: a stock with POSITIVE 12-month momentum may have run its course, while one with
NEGATIVE momentum but good fundamentals may be unfairly punished and due to rebound. So flip
the momentum family sign (favour losers) and see if it beats the normal (favour-winners)
version. Because a contrarian/mean-reversion bet plays out slowly, also sweep the rebalance
frequency (monthly / quarterly / annual) — monthly suits momentum, slower may suit reversal.

Value+Graham+growth families held fixed; only the momentum leg's sign and the cadence change.
Clean PIT S&P 500 (honest read). Plain top-30 (no band) to isolate the two effects.
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
from strategies.magic_formula.construct import combine_ranks, pnl, weights_top_n

EXCLUDE = ("Financial Services", "Utilities")
REBALS = {"monthly": "ME", "quarterly": "QE", "annual": "YE"}


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
    excluded = sp500_sectors().reindex(adj.columns).isin(EXCLUDE)
    base = pd.DataFrame(True, index=adj.index, columns=adj.columns) & ~pd.Series(excluded, index=adj.columns)
    base = base & sp500_pit_eligible(adj.index, list(adj.columns))
    f = load_fundamentals(list(adj.columns), start, end, items=ENHANCED_ITEMS, sources=("edgar",), calendar=adj.index)
    mcap = close * f["shares_diluted"].reindex_like(close)
    base = base & mcap.notna()

    value = [fcf_ev_yield(f, mcap), fcf_return_on_capital(f)]
    graham = [graham_number_yield(f, mcap)]
    growth = [revenue_growth(f), fcf_growth(f)]
    mom = residual_momentum(adj, lookback=252, skip=21)

    print("[run] momentum-direction x rebalance …\n")
    rows = []
    # reference: no momentum leg at all
    rank_nomom = combine_ranks([value, graham, growth], base)
    net, _ = pnl(weights_top_n(rank_nomom.where(base), adj, "ME", top_n), adj, volume, close)
    rows.append(("no-momentum (monthly)", net))
    for direction, panel_m in [("normal (winners)", mom), ("REVERSED (losers)", -mom)]:
        rank = combine_ranks([value, graham, growth, [panel_m]], base)
        for rname, rb in REBALS.items():
            w = weights_top_n(rank.where(base), adj, rb, top_n)
            net, _ = pnl(w, adj, volume, close)
            rows.append((f"{direction}, {rname}", net))

    common = None
    for _, net in rows:
        idx = net.replace(0.0, np.nan).dropna().index
        common = idx if common is None else common.intersection(idx)

    print("=" * 76)
    print(f"MOMENTUM REVERSED?  ({start}→{end}, PIT S&P 500, top-{top_n}, net)")
    print("=" * 76)
    print(f"  {'variant':26s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s}")
    for name, net in rows:
        s = summary_stats(net.reindex(common).fillna(0.0))
        print(f"  {name:26s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
              f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")
    s = summary_stats(spy.reindex(common).fillna(0.0))
    print(f"  {'SPY':26s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
          f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.2%}")

    out = ROOT / "results" / "momentum_reverse"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({n: net for n, net in rows}).to_csv(out / "momentum_reverse_net.csv")
    print(f"\n  wrote {out}/momentum_reverse_net.csv")


if __name__ == "__main__":
    main()
