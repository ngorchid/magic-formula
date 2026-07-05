"""Walk-forward / out-of-sample sanity checks for the multi-stream portfolio.

Everything in the pipeline is already *causal* (signals use trailing windows, IC-weights
are rolling + lagged, the allocator uses lagged inverse vol, positions are lagged). So
the daily blend series is tradeable. What remains in-sample is human choice — which
signals we kept and the allocator params. This script attacks that:

  1. Selection-bias check — run equity_mn with ALL candidate price signals (no cherry-
     picking) and let rolling IC-weighting decide. If the blend still works, the edge
     wasn't manufactured by dropping the signals that looked bad in-sample.
  2. Sub-period consistency — Sharpe per calendar year and in three disjoint blocks.
     Guards against the result being one lucky regime (e.g. trend's 2022 CTA run).
  3. Held-out second half — split the series in two; the second half is data none of
     the design was *meant* to target. Report blend, SPY, and the 60/40 combo on each.

Not a from-scratch blind walk-forward (impossible once the data has been seen), but it
makes overfitting expensive to hide.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from combination import equal_risk_allocate
from config import CONFIG, build_streams
from data import load_prices

ALL_PRICE_SIGNALS = ["momentum_12_1", "short_term_reversal", "residual_momentum", "low_volatility"]


def _sharpe(r: pd.Series) -> float:
    return summary_stats(r)["sharpe"]


def _build(end: str, equity_signals: list[str]):
    cfg = copy.deepcopy(CONFIG)
    cfg["end"] = end
    cfg["streams"]["equity_mn"]["signals"] = equity_signals
    results = {s.name: s.run() for s in build_streams(cfg)}
    rets = {n: r.net_returns for n, r in results.items()}
    alloc = equal_risk_allocate(rets, target_portfolio_vol=cfg["portfolio_vol_target"])
    return results, rets, alloc


def main(end: str = "2026-05-26") -> None:
    print("Building streams (cherry-picked 3-signal equity_mn)…")
    _, rets3, alloc3 = _build(end, CONFIG["streams"]["equity_mn"]["signals"])
    print("Building streams (ALL 4 price signals, no selection)…")
    res4, rets4, alloc4 = _build(end, ALL_PRICE_SIGNALS)

    blend3, blend4 = alloc3.blended_returns, alloc4.blended_returns

    # SPY aligned to the (all-signals) blend window.
    idx = blend4.index
    spy = load_prices(["SPY"], str(idx.min().date()), str(idx.max().date()), field="adj_close")["SPY"]
    spy_ret = spy.pct_change(fill_method=None).reindex(idx).fillna(0.0)

    # ---- 1. Selection-bias check ----
    print("\n========== 1. Selection-bias check (full window) ==========")
    print(f"  blend, cherry-picked 3 signals : Sharpe {_sharpe(blend3):.3f}")
    print(f"  blend, ALL 4 signals (no pick) : Sharpe {_sharpe(blend4.reindex(blend3.index)):.3f}")
    icw = {k.replace('ic_mean.', ''): round(v, 4)
           for k, v in res4["equity_mn"].diagnostics.items() if k.startswith("ic_mean.")}
    print(f"  equity_mn IC weights (all-4)   : {icw}")
    print("  (low_volatility staying negative = IC-weighting correctly ignores/inverts it on its own)")

    # ---- 2. Sub-period consistency (use the no-cherry-pick blend) ----
    blend, rets = blend4, rets4
    print("\n========== 2. Sub-period consistency (no-cherry-pick blend) ==========")
    print("  per-year Sharpe:")
    yr = pd.DataFrame({
        "equity_mn": rets["equity_mn"].reindex(idx), "trend": rets["trend"].reindex(idx),
        "blend": blend, "SPY": spy_ret,
    })
    for y, g in yr.groupby(yr.index.year):
        if len(g) < 60:
            continue
        s = {c: np.sqrt(252) * g[c].mean() / g[c].std() if g[c].std() else float("nan") for c in g}
        print(f"    {y}:  eqmn {s['equity_mn']:+.2f}   trend {s['trend']:+.2f}   "
              f"blend {s['blend']:+.2f}   SPY {s['SPY']:+.2f}")

    blocks = np.array_split(idx, 3)
    print("  three disjoint blocks (blend / SPY):")
    for b in blocks:
        b = pd.DatetimeIndex(b)
        print(f"    {b.min().date()}..{b.max().date()}:  blend {_sharpe(blend.reindex(b)):+.2f}   "
              f"SPY {_sharpe(spy_ret.reindex(b)):+.2f}")

    # ---- 3. Held-out second half ----
    mid = idx[len(idx) // 2]
    print(f"\n========== 3. Held-out split at {mid.date()} ==========")
    for label, b in (("DEV (1st half)", idx[idx < mid]), ("HOLDOUT (2nd half)", idx[idx >= mid])):
        bl = blend.reindex(b)
        sp = spy_ret.reindex(b)
        combo = 0.6 * sp + 0.4 * bl * (sp.std() / bl.std())
        print(f"  {label:20}  blend {_sharpe(bl):+.2f}   SPY {_sharpe(sp):+.2f}   "
              f"60/40 SPY+blend {_sharpe(combo):+.2f}")


if __name__ == "__main__":
    main()
