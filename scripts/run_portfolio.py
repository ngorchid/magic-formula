"""End-to-end run of the multi-stream portfolio.

Each enabled stream runs independently, their net P&L is fed into the equal-risk
meta-allocator, and the blended portfolio's stats and plots are written to
``results/``.
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

from backtest import summary_stats
from combination import equal_risk_allocate
from config import CONFIG, build_streams


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Capital: ${CONFIG['notional']:,.0f}   Window: {CONFIG['start']} → {CONFIG['end'] or 'today'}")
    print(f"Portfolio vol target: {CONFIG['portfolio_vol_target']:.0%}")

    streams = build_streams()
    if not streams:
        raise SystemExit("No streams enabled in config")

    stream_results = {}
    for stream in streams:
        print(f"\n[stream:{stream.name}] running…")
        res = stream.run()
        stream_results[stream.name] = res
        stats = summary_stats(res.net_returns)
        print(
            f"  ann_ret={stats['ann_return']:.2%}  vol={stats['ann_vol']:.2%}  "
            f"sharpe={stats['sharpe']:.2f}  maxDD={stats['max_drawdown']:.2%}"
        )
        for k, v in res.diagnostics.items():
            if not isinstance(v, (dict, list)):
                print(f"  diag.{k}={v}")

    # Stream-level correlations.
    rets_df = pd.DataFrame({n: r.net_returns for n, r in stream_results.items()}).dropna(how="all")
    print("\nStream correlations:")
    print(rets_df.corr().round(2).to_string())

    alloc = equal_risk_allocate(
        {n: r.net_returns for n, r in stream_results.items()},
        target_portfolio_vol=CONFIG["portfolio_vol_target"],
    )

    portfolio_stats = summary_stats(alloc.blended_returns)
    print("\n=== Portfolio (after equal-risk allocation + vol targeting) ===")
    for k, v in portfolio_stats.items():
        print(f"  {k}: {v:.4f}")

    # Persist summaries.
    rows = {
        "portfolio": portfolio_stats,
        **{name: summary_stats(r.net_returns) for name, r in stream_results.items()},
    }
    pd.DataFrame(rows).to_csv(results_dir / "portfolio_summary.csv")
    alloc.weights.to_csv(results_dir / "portfolio_weights.csv")

    # Cumulative-return plot.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    cum_port = (1 + alloc.blended_returns.fillna(0.0)).cumprod()
    cum_port.plot(ax=axes[0], color="black", linewidth=2.0, label="portfolio")
    for name, r in stream_results.items():
        cum = (1 + r.net_returns.fillna(0.0)).cumprod()
        cum.plot(ax=axes[0], alpha=0.6, label=name)
    axes[0].set_ylabel("cumulative return (×)")
    axes[0].set_title("Multi-stream portfolio — net of costs")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    drawdown = cum_port / cum_port.cummax() - 1.0
    drawdown.plot(ax=axes[1], color="firebrick")
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.3)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = results_dir / "portfolio_cumulative.png"
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
