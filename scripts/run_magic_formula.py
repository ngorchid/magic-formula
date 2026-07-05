"""Run the Magic Formula stream and benchmark it against SPY (buy & hold).

Long-only and market-exposed, so unlike the market-neutral stream this is a fair
apples-to-apples comparison with the index. Window is capped at ~5y by free SimFin
prices. Writes a summary CSV and cumulative-return plot to results/.
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
from data import load_prices
from strategies.magic_formula import MagicFormula, MagicFormulaConfig


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    cfg = MagicFormulaConfig()
    print(f"Magic Formula — top {cfg.top_n}, {cfg.rebalance} rebalance, "
          f"mcap≥${cfg.min_market_cap/1e6:.0f}M, ex {cfg.exclude_sectors}")
    res = MagicFormula(cfg).run()
    net = res.net_returns
    win_start, win_end = net.index.min().date(), net.index.max().date()
    print(f"Window: {win_start} → {win_end}")
    for k, v in res.diagnostics.items():
        print(f"  diag.{k}: {v}")

    # SPY benchmark over the same window, aligned to the strategy's trading days.
    spy = load_prices(["SPY"], str(win_start), str(win_end), field="adj_close")["SPY"]
    spy_ret = spy.pct_change(fill_method=None).reindex(net.index).fillna(0.0)

    rows = {
        "magic_formula (net)": summary_stats(net),
        "magic_formula (gross)": summary_stats(res.gross_returns),
        "SPY buy & hold": summary_stats(spy_ret),
    }
    table = pd.DataFrame(rows).T[["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]]
    print("\n================ Magic Formula vs SPY ================")
    print(table.to_string(float_format=lambda x: f"{x: .4f}"))
    table.to_csv(results_dir / "magic_formula_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + net).cumprod().plot(ax=ax, label="Magic Formula (net)", color="tab:green", lw=2)
    (1 + res.gross_returns).cumprod().plot(ax=ax, label="Magic Formula (gross)", color="tab:green", alpha=0.4, ls="--")
    (1 + spy_ret).cumprod().plot(ax=ax, label="SPY buy & hold", color="black", lw=1.5)
    ax.set_title(f"Magic Formula vs SPY ({win_start} → {win_end})")
    ax.set_ylabel("cumulative return (×)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "magic_formula_cumulative.png", dpi=140)
    print(f"\nWrote {results_dir/'magic_formula_summary.csv'} and .png")


if __name__ == "__main__":
    main()
