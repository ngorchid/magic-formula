"""Quality/value sleeve on deep EDGAR fundamentals — Sharpe + correlation test.

The point of the exercise (see memory: current-focus / fundamentals-source): the
momentum book needs a *genuinely uncorrelated* stream to blend with. Value/quality is
the classic diversifier to momentum, but it was un-evaluable on SimFin's ~5y of data.
With the SEC EDGAR backend (~2012→present, point-in-time) we can finally run it over a
multi-regime window and answer two questions:

  1. Does the quality/value sleeve clear a useful Sharpe on its own?
  2. Is it uncorrelated to the existing momentum book (and trend)? — the real prize,
     since diversification, not raw Sharpe, is the lever that lifts the blend.

Runs three streams over the same window and reports per-stream stats, the per-signal
rolling-IC weights the combiner assigned, and the net-return correlation matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from strategies.equity_mn import EquityMarketNeutral
from strategies.equity_mn.stream import EquityMNConfig
from strategies.trend import CrossAssetTrend, TrendConfig

QUALITY_SIGNALS = ["gross_profitability", "return_on_equity", "accruals", "earnings_yield"]
MOMENTUM_SIGNALS = ["momentum_12_1", "short_term_reversal", "residual_momentum"]


def _run_equity_mn(name: str, signals: list[str], start: str, end: str | None,
                   sources: tuple[str, ...]):
    cfg = EquityMNConfig(
        start=start, end=end, signals=signals,
        combine="ic", ic_weighting="icir", fundamental_sources=sources,
    )
    print(f"[{name}] running equity_mn signals={signals} sources={sources} …")
    res = EquityMarketNeutral(cfg).run()
    return res


def _fmt(stats: dict) -> str:
    return (f"ann_ret={stats['ann_return']:+.2%}  vol={stats['ann_vol']:.2%}  "
            f"sharpe={stats['sharpe']:+.2f}  maxDD={stats['max_drawdown']:.2%}")


def main(start: str = "2012-01-01", end: str | None = None) -> None:
    results = {}

    quality = _run_equity_mn("quality", QUALITY_SIGNALS, start, end, ("edgar",))
    results["quality"] = quality

    momentum = _run_equity_mn("momentum", MOMENTUM_SIGNALS, start, end, ("edgar",))
    results["momentum"] = momentum

    print("[trend] running cross-asset trend …")
    trend = CrossAssetTrend(TrendConfig(start=start, end=end)).run()
    results["trend"] = trend

    # --- per-stream performance ------------------------------------------------
    print("\n" + "=" * 78)
    print(f"PER-STREAM PERFORMANCE  ({start} → {end or 'today'}, net of costs)")
    print("=" * 78)
    for name, res in results.items():
        print(f"  {name:9s} {_fmt(summary_stats(res.net_returns))}")

    # --- quality signal IC weights (which value signals actually worked) --------
    ic = {k: v for k, v in quality.diagnostics.items() if k.startswith("ic_mean.")}
    if ic:
        print("\nQuality sleeve — mean rolling rank-IC per signal (combiner weights):")
        for k, v in ic.items():
            print(f"  {k.replace('ic_mean.', ''):22s} {v:+.4f}")

    # --- correlation of net returns (the diversification question) --------------
    rets = pd.DataFrame({n: r.net_returns for n, r in results.items()}).dropna()
    print(f"\nNet-return correlation matrix (common window, {len(rets)} days):")
    print(rets.corr().round(3).to_string())

    qm = rets["quality"].corr(rets["momentum"])
    qt = rets["quality"].corr(rets["trend"])
    print(f"\n>> quality vs momentum corr = {qm:+.3f}   quality vs trend corr = {qt:+.3f}")
    print(">> (near-zero / negative = a genuinely diversifying stream to add)")

    out = ROOT / "results" / "quality_sleeve"
    out.mkdir(parents=True, exist_ok=True)
    rets.to_csv(out / "stream_net_returns.csv")
    pd.DataFrame({n: summary_stats(r.net_returns) for n, r in results.items()}).to_csv(
        out / "stream_stats.csv"
    )
    print(f"\nwrote {out}/stream_net_returns.csv, stream_stats.csv")


if __name__ == "__main__":
    main()
