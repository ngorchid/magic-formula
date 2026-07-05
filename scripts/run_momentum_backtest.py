"""End-to-end baseline run.

Pulls 10 years of S&P 500 daily bars via yfinance, computes the 12-1 momentum
signal, cross-sectionally z-scores it, and runs the long-short backtester. Writes
a cumulative-return PNG and a summary CSV to ``results/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import LinearCostModel, VectorizedBacktester, summary_stats
from combination import cs_zscore
from data import download_ohlcv, sp500_tickers
from signals import momentum_12_1


def main(start: str = "2015-01-01", end: str | None = None) -> dict[str, float]:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"[1/5] Loading S&P 500 universe…")
    tickers = sp500_tickers()
    print(f"      {len(tickers)} tickers")

    print(f"[2/5] Downloading OHLCV {start} → {end} (cached)…")
    panel = download_ohlcv(tickers, start, end)
    prices = panel["adj_close"].dropna(how="all", axis=1)
    volume = panel["volume"].reindex_like(prices)
    print(f"      prices shape {prices.shape}")

    print("[3/5] Computing 12-1 momentum and z-scoring…")
    raw = momentum_12_1(prices)
    scores = cs_zscore(raw)

    print("[4/5] Running backtest (monthly rebalance, top/bottom quintile)…")
    bt = VectorizedBacktester(
        top_quantile=0.2,
        rebalance="ME",
        cost_model=LinearCostModel(half_spread_bps=2.5, impact_coef_bps=10.0),
    )
    res = bt.run(scores, prices, volume=volume)

    stats_gross = summary_stats(res.gross_returns)
    stats_net = summary_stats(res.net_returns)
    print("[5/5] Results")
    print(f"      Gross  ann_ret={stats_gross['ann_return']:.2%}  vol={stats_gross['ann_vol']:.2%}  "
          f"sharpe={stats_gross['sharpe']:.2f}  maxDD={stats_gross['max_drawdown']:.2%}")
    print(f"      Net    ann_ret={stats_net['ann_return']:.2%}  vol={stats_net['ann_vol']:.2%}  "
          f"sharpe={stats_net['sharpe']:.2f}  maxDD={stats_net['max_drawdown']:.2%}")

    summary = pd.DataFrame({"gross": stats_gross, "net": stats_net})
    summary.to_csv(results_dir / "momentum_summary.csv")

    cum_gross = (1 + res.gross_returns).cumprod()
    cum_net = (1 + res.net_returns).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    cum_gross.plot(ax=ax, label="gross", color="tab:blue")
    cum_net.plot(ax=ax, label="net of costs", color="tab:orange")
    ax.set_title("12-1 momentum, S&P 500, dollar-neutral L/S quintiles")
    ax.set_ylabel("cumulative return (×)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "momentum_cumulative.png", dpi=140)
    print(f"      wrote {results_dir/'momentum_cumulative.png'}")
    return stats_net


if __name__ == "__main__":
    main()
