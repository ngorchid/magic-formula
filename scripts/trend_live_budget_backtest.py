"""What Sharpe does the trend book the account can ACTUALLY HOLD deliver?

The published +0.78 Sharpe (variant D, 2011-2026) is a TEN-MARKET backtest with continuous
position sizes. The live book is neither. Integer contract rounding and the per-market cap mean
it holds 3 markets at $50k, 5 at $75k-$200k and 7 at $300k -- and the sleeve's own documentation
says "the breadth is what makes the sleeve worth holding". So the headline number describes a
book that does not exist at any budget being considered.

This is the same shape as the options-vrp deployment gap, where the validated backtest held one
spread at a time on SPX and said nothing about the live basket.

METHOD. Drive the LIVE sizer (`trend_overlay.execution.compute_targets`) day by day over
history, at a range of budgets, and mark the resulting integer contract positions against the
proxy returns. Nothing is reimplemented: the same function the paper book calls each morning
produces the positions here, so the vol target, the vol floor, the per-market cap, the gross
cap and the rounding band are all exactly as deployed. `held` is threaded through so the
hysteresis band applies, which is what keeps turnover realistic.

DELIBERATELY PROXY RETURNS, NOT FUTURES. `trend_proxy_validation.py` already measured the
proxy-vs-futures gap separately (sound where the book sizes; the rates proxies over-signal long
by 8-14pp but barely size). Using proxies here isolates ONE variable -- what integer sizing at a
real budget costs -- rather than mixing two effects and being unable to attribute the result.

The very large budget row is the reference: at that size rounding is immaterial and the caps do
not bind, so it reproduces the continuous-sizing backtest and the gap to it IS the sizing cost.

Usage:  python scripts/trend_live_budget_backtest.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "trend-overlay"))

from trend_overlay.contracts import FUTURES  # noqa: E402
from trend_overlay.execution import TrendPaperConfig, compute_targets  # noqa: E402

warnings.filterwarnings("ignore")
logging_off = True

START, END = "2011-01-01", "2026-09-14"
BUDGETS = [50_000, 75_000, 100_000, 200_000, 300_000, 500_000, 100_000_000]
WARMUP = 300          # need 252d for the slow signal before the first position
COST_BP = 2.0         # half-spread per side, the backtest's documented assumption
OUT = ROOT / "results" / "trend_live_budget"


def load_proxies() -> pd.DataFrame:
    cols = sorted({f.proxy_etf for f in FUTURES})
    px = yf.download(cols, start=START, end=END, auto_adjust=True,
                     progress=False, group_by="column")["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px.dropna(how="all").ffill()


def run_budget(px: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Daily walk-forward at one budget, using the live sizer for every position."""
    cfg = TrendPaperConfig(budget=budget)
    rets = px.pct_change(fill_method=None)
    proxy_of = {f.market: f.proxy_etf for f in FUTURES}
    dates = px.index[WARMUP:]

    held: dict[str, int] = {}
    prev_expo: dict[str, float] = {}
    rows = []
    for i, t in enumerate(dates[:-1]):
        tgt = compute_targets(px.loc[:t], cfg, held=held or None)
        expo = (tgt["contracts"] * tgt["notional"]).to_dict()   # signed $ per market
        held = {m: int(c) for m, c in tgt["contracts"].items()}

        nxt = dates[i + 1]
        pnl = 0.0
        for m, e in expo.items():
            p = proxy_of.get(m)
            if p is None or p not in rets.columns or e == 0:
                continue
            r = rets.at[nxt, p] if nxt in rets.index else np.nan
            if np.isfinite(r):
                pnl += e * r
        turn = sum(abs(expo.get(m, 0.0) - prev_expo.get(m, 0.0)) for m in set(expo) | set(prev_expo))
        pnl -= turn * COST_BP / 1e4
        prev_expo = expo

        rows.append({"date": nxt, "ret": pnl / budget,
                     "n_mkts": int((tgt["contracts"] != 0).sum()),
                     "gross": sum(abs(v) for v in expo.values()) / budget})
    return pd.DataFrame(rows).set_index("date")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else np.nan,
            "maxdd": float((cum / cum.cummax() - 1).min())}


def main() -> None:
    px = load_proxies()
    print(f"  proxy panel {px.shape}, {px.index.min():%Y-%m} -> {px.index.max():%Y-%m}")
    print(f"  walking the LIVE sizer daily at {len(BUDGETS)} budgets "
          f"(~{len(px) - WARMUP} rebalances each)\n")

    out = {}
    for b in BUDGETS:
        d = run_budget(px, b)
        out[b] = d
        s = stats(d["ret"])
        label = "reference (no rounding/cap)" if b >= 10_000_000 else f"${b:,.0f}"
        print(f"  {label:28} Sharpe {s['sharpe']:>+5.2f}  ann {s['ann']:>+6.1%}  "
              f"vol {s['vol']:>5.1%}  maxDD {s['maxdd']:>+6.1%}  "
              f"markets {d['n_mkts'].mean():>4.1f}  gross {d['gross'].mean():>4.2f}x")

    ref = stats(out[BUDGETS[-1]]["ret"])["sharpe"]
    print("\n" + "=" * 92)
    print("COST OF REAL SIZING — gap to the continuous-sizing reference")
    print("=" * 92)
    print(f"  {'budget':14}{'Sharpe':>9}{'vs reference':>15}{'avg markets held':>19}")
    for b in BUDGETS[:-1]:
        s = stats(out[b]["ret"])
        print(f"  ${b:<13,.0f}{s['sharpe']:>+9.2f}{s['sharpe'] - ref:>+15.2f}"
              f"{out[b]['n_mkts'].mean():>19.1f}")
    print(f"  {'reference':14}{ref:>+9.2f}{0.0:>+15.2f}"
          f"{out[BUDGETS[-1]]['n_mkts'].mean():>19.1f}")

    print("\n  The published +0.78 is the reference row's world: continuous sizes, ten markets.")
    print("  The gap below it is what integer contracts and the per-market cap actually cost,")
    print("  and 'avg markets held' is the breadth the sleeve's edge depends on.")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"ret_{b}": out[b]["ret"] for b in BUDGETS}).to_csv(OUT / "by_budget.csv")
    print(f"\nWrote {OUT / 'by_budget.csv'}")


if __name__ == "__main__":
    main()
