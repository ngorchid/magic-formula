"""Backtest the cross-asset trend overlay and show WHY it's a diversifier.

Reports the standalone profile (vol ~10%, Sharpe, drawdown, skew), its correlation to
equities, its behaviour in the worst equity months ("crisis alpha"), and — the point —
how bolting it onto a long equity book (SPY proxy) changes the total portfolio's Sharpe
and drawdown. Free ETF-proxy basket; production would use futures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from data import download_ohlcv
from strategies.trend_futures import TrendOverlayConfig, run_trend_overlay


def _row(name, r, common, bench=None):
    s = summary_stats(r.reindex(common).fillna(0.0))
    c = r.reindex(common).fillna(0.0).corr(bench.reindex(common).fillna(0.0)) if bench is not None else np.nan
    sk = r.reindex(common).fillna(0.0).skew()
    cs = f"{c:>+8.2f}" if bench is not None else f"{'—':>8s}"
    print(f"  {name:26s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} {s['sharpe']:>+7.2f} "
          f"{s['max_drawdown']:>+8.2%} {sk:>+7.2f} {cs}")


def main(start: str = "2011-01-01", end: str | None = None) -> None:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print("[1/2] running trend overlay …")
    res = run_trend_overlay(TrendOverlayConfig(start=start, end=end))
    trend = res.net_returns
    print(f"      markets={res.diagnostics['n_markets']}  avg gross lev="
          f"{res.diagnostics['avg_gross_leverage']:.1f}x")

    print("[2/2] equity benchmark + blends …")
    spy = download_ohlcv(["SPY"], start, end)["adj_close"]["SPY"].pct_change(fill_method=None)
    common = trend.replace(0.0, np.nan).dropna().index.intersection(spy.dropna().index)

    print("\n" + "=" * 82)
    print(f"CROSS-ASSET TREND OVERLAY  ({start}→{end}, ETF proxies, net)")
    print("=" * 82)
    print(f"  {'strategy':26s} {'ann_ret':>8s} {'vol':>7s} {'sharpe':>7s} {'maxDD':>8s} {'skew':>7s} {'corrSPY':>8s}")
    _row("trend overlay", trend, common, spy)
    _row("SPY (long equity)", spy, common, spy)
    # Overlay blends: SPY held fully + trend bolted on via futures margin (additive).
    _row("SPY + 0.5x trend", spy + 0.5 * trend, common, spy)
    _row("SPY + 1.0x trend", spy + 1.0 * trend, common, spy)

    # Crisis alpha: trend's average return in the worst decile of SPY months.
    m_spy = (1 + spy.reindex(common).fillna(0.0)).resample("ME").prod() - 1
    m_trd = (1 + trend.reindex(common).fillna(0.0)).resample("ME").prod() - 1
    worst = m_spy <= m_spy.quantile(0.10)
    print("\n  CRISIS ALPHA (monthly):")
    print(f"    SPY worst-decile months: SPY avg {m_spy[worst].mean():+.2%}  "
          f"trend avg {m_trd[worst].mean():+.2%}  (hit rate trend>0: {(m_trd[worst] > 0).mean():.0%})")
    print(f"    all months:              SPY avg {m_spy.mean():+.2%}  trend avg {m_trd.mean():+.2%}")

    out = ROOT / "results" / "trend_overlay"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trend": trend, "spy": spy}).to_csv(out / "trend_overlay_net.csv")
    print(f"\n  wrote {out}/trend_overlay_net.csv")


if __name__ == "__main__":
    main()
