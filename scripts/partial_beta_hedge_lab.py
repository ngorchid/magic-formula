"""Lab: how much of the equity sleeve's beta should the book actually hedge?

run_beta_hedge.py showed the enhanced Magic Formula is beta 1.05 with +4.38%/yr of
market-neutral alpha at 8.33% residual vol (Sharpe 0.56, corr to SPY -0.02). That is a
knob, not a new stream: hedging fraction h is exactly a linear blend, because

    hedged   = long - beta*SPY - costs
    partial  = long - h*beta*SPY - h*costs = (1-h)*long + h*hedged

so no re-backtest is needed to price any h.

THE CONTROL THAT MATTERS. Hedging is not the only way to cut equity risk — you can just
hold less of the book and park the rest in T-bills, which costs nothing and cannot break
operationally. At 4-5% short rates that is a real return, not a zero. So every hedged
variant is compared against an UNHEDGED book de-levered to the SAME volatility, with the
released cash earning BIL. If de-levering wins, the hedge is complexity for nothing.

Book construction follows book_assess2.py: magic formula is the CORE (cash), trend and
VRP are bolted on via margin (futures/options post margin, not cash), sized to 7% and 5%
vol contributions.

CAVEAT, loudly: the VRP series here is the idealized variance-swap proxy, NOT the real
defined-risk spread strategy. Its level is optimistic and its correlations are not the
strategy's. Only magic_f and trend are real. Read VRP rows as indicative only.

Run: python scripts/partial_beta_hedge_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import summary_stats
from scripts.book_assess2 import scale_to_vol, vrp_series  # noqa: E402

HEDGE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)


def load_streams() -> tuple[pd.DataFrame, pd.Series]:
    bh = pd.read_csv(ROOT / "results" / "beta_hedge" / "beta_hedge.csv",
                     index_col=0, parse_dates=True)
    trend = pd.read_csv(ROOT / "results" / "trend_overlay" / "trend_overlay_net.csv",
                        index_col=0, parse_dates=True)["trend"]
    vrp = vrp_series()
    bil = yf.download("BIL", start="2011-01-01", auto_adjust=True,
                      progress=False)["Close"].pct_change()
    if isinstance(bil, pd.DataFrame):
        bil = bil.iloc[:, 0]
    S = pd.DataFrame({"long": bh["long_book"], "hedged": bh["beta_hedged"],
                      "trend": trend, "vrp": vrp, "cash": bil}).dropna()
    return S, bh["beta"]


def stats(r: pd.Series) -> dict:
    return summary_stats(r.fillna(0.0))


def line(name: str, r: pd.Series, spy: pd.Series | None = None) -> None:
    s = stats(r)
    extra = ""
    if spy is not None:
        bad = spy <= spy.quantile(0.05)
        # cumulative return ACROSS the worst-5% SPY days, not an annualised daily mean
        extra = (f"  corrSPY {r.corr(spy):>+5.2f}  tail-corr {r[bad].corr(spy[bad]):>+5.2f}"
                 f"  worst5%cum {((1+r[bad]).prod()-1):>+7.1%}")
    print(f"  {name:30s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
          f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.1%}{extra}")


def main() -> None:
    S, beta = load_streams()
    spy = yf.download("SPY", start="2011-01-01", auto_adjust=True,
                      progress=False)["Close"].pct_change().reindex(S.index)
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.fillna(0.0)

    trend_o = scale_to_vol(S["trend"], 0.07)
    vrp_o = scale_to_vol(S["vrp"], 0.05)
    cash = S["cash"]

    print("=" * 118)
    print(f"PARTIAL BETA HEDGE  ({S.index[0].date()} -> {S.index[-1].date()}, "
          f"avg beta {beta.mean():.2f})")
    print("=" * 118)
    print(f"  {'':30s} {'annRet':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s}")
    line("cash (BIL)", cash)

    print("\n  EQUITY SLEEVE ALONE, by hedge fraction h:")
    sleeves = {}
    for h in HEDGE_FRACTIONS:
        eq = (1 - h) * S["long"] + h * S["hedged"]
        sleeves[h] = eq
        line(f"h={h:.2f}", eq, spy)

    # PRIMARY book uses only the two streams that are real backtests (magic + trend).
    print("\n  BOOK (REAL STREAMS ONLY) = sleeve + trend(7% vol):")
    books = {}
    for h, eq in sleeves.items():
        b = eq + trend_o
        books[h] = b
        line(f"h={h:.2f}", b, spy)

    print("\n  [indicative only] + vrp(5% vol) — idealized proxy, Sharpe ~6, NOT real:")
    for h, eq in sleeves.items():
        line(f"h={h:.2f}", eq + trend_o + vrp_o, spy)

    print("\n" + "=" * 118)
    print("THE CONTROL — same volatility reached by DE-LEVERING into T-bills instead")
    print("=" * 118)
    print(f"  {'':30s} {'annRet':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s}")
    v_long = S["long"].std()
    for h in HEDGE_FRACTIONS:
        if h == 0:
            continue
        target = sleeves[h].std()
        w = min(target / v_long, 1.0)
        eq_dl = w * S["long"] + (1 - w) * cash
        b_dl = eq_dl + trend_o
        s_h, s_d = stats(books[h]), stats(b_dl)
        print(f"  --- matching h={h:.2f} (equity weight {w:.0%} + {1-w:.0%} cash) ---")
        line(f"  hedged   book h={h:.2f}", books[h], spy)
        line(f"  delevered book w={w:.2f}", b_dl, spy)
        win = "HEDGE" if s_h["sharpe"] > s_d["sharpe"] else "DE-LEVER"
        print(f"      -> higher book Sharpe: {win} "
              f"({s_h['sharpe']:+.2f} vs {s_d['sharpe']:+.2f}), "
              f"return gap {s_h['ann_return']-s_d['ann_return']:+.2%}/yr")

    print("\n" + "=" * 118)
    print("CRISIS WINDOWS — cumulative book return")
    print("=" * 118)
    wins = [("2015-16 selloff", "2015-08-01", "2016-02-29"),
            ("2018 Q4", "2018-10-01", "2018-12-31"),
            ("2020 COVID", "2020-02-15", "2020-03-31"),
            ("2022 bear", "2022-01-01", "2022-12-31")]
    print("    " + "window".ljust(18) + "".join(f"{'h='+format(h,'.2f'):>10s}" for h in HEDGE_FRACTIONS))
    for lbl, a, b in wins:
        print("    " + lbl.ljust(18)
              + "".join(f"{((1+books[h].loc[a:b]).prod()-1):>+10.1%}" for h in HEDGE_FRACTIONS))

    print("\n  NB vrp = idealized variance-swap proxy, NOT the real defined-risk strategy.")
    print("     Its level is optimistic and its correlations are not the strategy's.")

    out = ROOT / "results" / "partial_hedge"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"book_h{h:.2f}": b for h, b in books.items()}).to_csv(out / "books.csv")
    print(f"\n  wrote {out}/books.csv")


if __name__ == "__main__":
    main()
