"""Overnight vs intraday decomposition — the cheap kill switch.

The claim (Cliff/Cooper/Gulen 2008; Lou, Polk & Skouras JFE 2019; Bogousslavsky 2021): the
whole equity risk premium accrues CLOSE-TO-OPEN, while OPEN-TO-CLOSE is flat to negative. Buy
at the close, sell at the open, and you earn the market's long-run return while sitting out the
hours it is actually tradeable.

TWO VERSIONS, and they are very different animals:

  A  long overnight, flat intraday    1 round trip/day. NOT market neutral — it captures
                                      overnight beta, so it correlates with the equity book we
                                      already have. A timing overlay, not a diversifier.
  B  long overnight, SHORT intraday   always in the market, flipping at each open and close.
                                      Genuinely neutral, but twice the trading.

WHAT ACTUALLY DECIDES IT is not the full-sample result — that is well documented and not in
question — but whether it survives (a) costs at ~252 round trips a year, versus 16 for the
auction trade, and (b) the post-2018 period, after the effect became widely publicised and was
packaged into ETFs. A backtest leaning on 1993-2015 is describing a different world.

Costs are charged in bp of notional per round trip and priced on ES/MES futures rather than
SPY: no PDT constraints, and the futures market is open across the whole window. NB the ES
"overnight" is a continuously traded session, not a gap — but it spans the same clock hours, so
the economics carry over.

Prices are auto-adjusted so dividends do not contaminate the split: SPY goes ex-dividend at the
OPEN, which on raw prices would charge the entire ~1.3%/yr yield against the overnight leg.

Run: python scripts/overnight_intraday_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "overnight"

TICKERS = ["SPY", "QQQ", "IWM"]

# ES round-trip cost in bp of notional: 1 tick crossed (0.25 pts x $50 = $12.50) plus
# commissions, on ~$300k notional. MES is proportionally the same on spread but ~10x worse
# on commission, so ES is the right instrument for a strategy trading every day.
COST_BP_RT = 0.48

ERAS = [
    ("1993-2007 pre-GFC", "1993-01-01", "2007-12-31"),
    ("2008-2014 GFC + QE", "2008-01-01", "2014-12-31"),
    ("2015-2017", "2015-01-01", "2017-12-31"),
    ("2018-2021 post-publicity", "2018-01-01", "2021-12-31"),
    ("2022-", "2022-01-01", "2030-12-31"),
]


def load(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for t in tickers:
        df = yf.download(t, start="1993-01-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "Close"]].dropna()
        df["overnight"] = df["Open"] / df["Close"].shift(1) - 1.0
        df["intraday"] = df["Close"] / df["Open"] - 1.0
        df["total"] = df["Close"] / df["Close"].shift(1) - 1.0
        out[t] = df.dropna()
        print(f"  {t}: {len(out[t])} days {out[t].index.min().date()}..{out[t].index.max().date()}")
    return out


def stats(r: pd.Series, cost_bp: float = 0.0, rt_per_day: float = 0.0) -> dict:
    net = r - cost_bp / 1e4 * rt_per_day
    ann = (1 + net).prod() ** (252 / len(net)) - 1 if len(net) else np.nan
    vol = net.std() * np.sqrt(252)
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else np.nan, "maxdd": dd}


def main() -> None:
    print("Downloading...")
    data = load(TICKERS)
    spy = data["SPY"]

    print("\n" + "=" * 100)
    print("THE DECOMPOSITION — gross, no costs (annualised)")
    print("=" * 100)
    print(f"  {'ticker':7s} {'period':22s} {'overnight':>20s} {'intraday':>20s} {'buy&hold':>20s}")
    print(f"  {'':7s} {'':22s} {'ret':>9s}{'sharpe':>11s} {'ret':>9s}{'sharpe':>11s} "
          f"{'ret':>9s}{'sharpe':>11s}")
    print("  " + "-" * 96)
    for t, df in data.items():
        o, i, b = (stats(df[c]) for c in ("overnight", "intraday", "total"))
        per = f"{df.index.min().date()}..{df.index.max().date()}"
        print(f"  {t:7s} {per:22s} {o['ann']:>+9.2%}{o['sharpe']:>+11.2f} "
              f"{i['ann']:>+9.2%}{i['sharpe']:>+11.2f} {b['ann']:>+9.2%}{b['sharpe']:>+11.2f}")

    # ---- the actual test: does it survive the modern era?
    print("\n" + "=" * 100)
    print("SPY BY ERA — gross. The effect is not in question; its SURVIVAL is.")
    print("=" * 100)
    print(f"  {'era':26s} {'n':>5s} {'overnight':>19s} {'intraday':>19s} {'spread (O-I)':>14s}")
    print(f"  {'':26s} {'':>5s} {'ret':>9s}{'sharpe':>10s} {'ret':>9s}{'sharpe':>10s} {'ret':>14s}")
    print("  " + "-" * 90)
    for label, lo, hi in ERAS:
        g = spy[(spy.index >= lo) & (spy.index <= hi)]
        if len(g) < 200:
            continue
        o, i = stats(g["overnight"]), stats(g["intraday"])
        print(f"  {label:26s} {len(g):>5d} {o['ann']:>+9.2%}{o['sharpe']:>+10.2f} "
              f"{i['ann']:>+9.2%}{i['sharpe']:>+10.2f} {o['ann']-i['ann']:>+14.2%}")

    # ---- tradeable versions, net of costs
    print("\n" + "=" * 100)
    print(f"TRADEABLE, NET OF COSTS (ES futures, {COST_BP_RT}bp per round trip)")
    print("=" * 100)
    print("  A = long overnight, flat intraday   (1 RT/day, NOT market neutral)")
    print("  B = long overnight, short intraday  (2 RT/day, genuinely neutral)\n")
    print(f"  {'era':26s} {'A net ret':>11s}{'A sharpe':>10s}{'A corrSPY':>11s}  "
          f"{'B net ret':>11s}{'B sharpe':>10s}{'B maxDD':>10s}")
    print("  " + "-" * 92)
    for label, lo, hi in [("FULL 1993-2026", "1993-01-01", "2030-12-31")] + ERAS:
        g = spy[(spy.index >= lo) & (spy.index <= hi)]
        if len(g) < 200:
            continue
        a = stats(g["overnight"], COST_BP_RT, 1.0)
        b_r = g["overnight"] - g["intraday"]
        b = stats(b_r, COST_BP_RT, 2.0)
        corr = (g["overnight"] - COST_BP_RT / 1e4).corr(g["total"])
        print(f"  {label:26s} {a['ann']:>+11.2%}{a['sharpe']:>+10.2f}{corr:>+11.2f}  "
              f"{b['ann']:>+11.2%}{b['sharpe']:>+10.2f}{b['maxdd']:>+10.1%}")

    # ---- how much cost can it bear?
    print("\n" + "=" * 100)
    print("COST SENSITIVITY — version B (the market-neutral one), 2018 onward")
    print("=" * 100)
    g = spy[spy.index >= "2018-01-01"]
    b_r = g["overnight"] - g["intraday"]
    print(f"  gross {stats(b_r)['ann']:+.2%}/yr, vol {stats(b_r)['vol']:.2%}, "
          f"Sharpe {stats(b_r)['sharpe']:+.2f}")
    for c in (0.0, 0.24, 0.48, 0.75, 1.00):
        s = stats(b_r, c, 2.0)
        print(f"    cost {c:.2f}bp/RT -> {s['ann']:>+7.2%}/yr  Sharpe {s['sharpe']:>+6.2f}  "
              f"(annual drag {c*2*252/1e4:.2%})")

    OUT.mkdir(parents=True, exist_ok=True)
    spy[["overnight", "intraday", "total"]].to_csv(OUT / "spy_decomposition.csv")
    print(f"\n  wrote {OUT}/spy_decomposition.csv")


if __name__ == "__main__":
    main()
