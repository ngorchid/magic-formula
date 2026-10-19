"""Sector momentum, market neutral: long the trending-up sectors, short the trending-down ones.

THE IDEA (user, 2026-10-19). Sectors that have done well over the last months tend to keep doing
better than sectors that have done badly (industry momentum; Moskowitz & Grinblatt 1999). Long the
leaders and short the laggards via sector ETFs is a pure bet on relative sector performance.

KNOWN RISKS, measured rather than assumed: the published long/short premium is mostly pre-2000;
after bear markets the short leg holds the most beaten sectors, which rebound hardest when the
market turns (the 2009 momentum crash); and equal dollars is NOT market neutral (XLK beta ~1.2,
XLU ~0.6), so the primary version is BETA neutral.

DESIGN (fixed before the first run).
  Universe: the SPDR sector ETFs, each once it has 13 months of history: the nine originals from
    Dec 1998, XLRE (Oct 2015) and XLC (Jun 2018) later.
  Signal: total return over the last 12 months excluding the most recent month (12-1), measured at
    the month-end close. 6-1 and 3-0 reported alongside, not chosen afterwards.
  Trade: at the close of the FIRST trading day of the next month (one day lag, closing auction),
    long the top 3, short the bottom 3, equal weight within a leg; held to the next rebalance.
    Beta neutral (primary): the legs are scaled so that their 252-day betas to SPY offset, gross
    exposure 2 (1 long + 1 short on average). Dollar neutral reported alongside.
  Costs: 2bp per unit of turnover (ETF half-spread + slippage; commissions are negligible at ETF
    prices), borrow 0.3%/yr on the short leg. No interest credited on cash (conservative).

PRIMARY TEST: beta-neutral 12-1, net monthly returns, 2000-2022: mean > 0 with t >= 2 AND
mean > 0 in 2013-2022. 2023 onwards is held out unless --holdout is passed.
Descriptive: per period, the worst months (2009), the long and short legs separately, and the
correlation with the live book (magic-formula core and trend overlay, monthly streams from
tail_conditional_lab.py).

Usage:  python scripts/sector_momentum_lab.py [--holdout]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loaders import download_ohlcv  # noqa: E402

ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
START, END = "1998-12-01", "2026-10-17"
DESIGN_START, DESIGN_END = "2000-01-01", "2022-12-31"
PERIODS = {"2000-2006": ("2000-01-01", "2006-12-31"), "2007-2012": ("2007-01-01", "2012-12-31"),
           "2013-2022": ("2013-01-01", "2022-12-31")}
HOLDOUT = {"2023+ (holdout)": ("2023-01-01", END)}
LOOKBACKS = {"12-1": (12, 1), "6-1": (6, 1), "3-0": (3, 0)}
N_LEG, COST_BP, BORROW, BETA_WIN = 3, 2.0, 0.003, 252
STREAMS = ROOT / "results" / "tail_conditional" / "monthly_streams.csv"
OUT = ROOT / "results" / "sector_momentum"


def backtest(px: pd.DataFrame, spy: pd.Series, look: tuple[int, int], beta_neutral: bool) -> pd.DataFrame:
    """Monthly holding-period returns (net, gross, legs) indexed by the month the position is held."""
    daily = px.pct_change()
    spy_r = spy.pct_change()
    month_end = px.groupby(px.index.to_period("M")).tail(1).index
    first_day = px.groupby(px.index.to_period("M")).head(1).index
    me = px.loc[month_end]
    L, skip = look
    sig = me.shift(skip) / me.shift(L) - 1
    hist = me.notna().cumsum() >= 13                        # 13 month-ends of history
    rows, prev_w = [], pd.Series(0.0, index=px.columns)
    for k in range(len(month_end) - 1):
        s = sig.iloc[k].where(hist.iloc[k]).dropna()
        if len(s) < 2 * N_LEG:
            continue
        entry = first_day[first_day > month_end[k]]
        if len(entry) < 2:
            break
        d0, d1 = entry[0], entry[1]                         # held from close d0 to close d1
        win = daily.loc[:d0].tail(BETA_WIN)
        sw = spy_r.loc[win.index]
        beta = win.apply(lambda c: c.cov(sw) / sw.var())
        longs, shorts = s.nlargest(N_LEG).index, s.nsmallest(N_LEG).index
        bl, bs = beta[longs].mean(), beta[shorts].mean()
        if beta_neutral and bl > 0 and bs > 0:
            wl, ws = 2 * bs / (bl + bs), 2 * bl / (bl + bs)   # wl*bl == ws*bs, wl + ws == 2
        else:
            wl = ws = 1.0
        w = pd.Series(0.0, index=px.columns)
        w[longs] = wl / N_LEG
        w[shorts] = -ws / N_LEG
        r = px.loc[d1] / px.loc[d0] - 1
        days = len(daily.loc[d0:d1]) - 1
        long_r = (w[longs] * r[longs]).sum()
        short_r = (w[shorts] * r[shorts]).sum()
        gross = long_r + short_r
        cost = COST_BP / 1e4 * (w - prev_w).abs().sum() + BORROW * ws * days / 252
        rows.append({"month": d1.to_period("M"), "entry": d0, "gross": gross, "net": gross - cost,
                     "long": long_r, "short": short_r, "beta_long": bl, "beta_short": bs,
                     "longs": ",".join(longs), "shorts": ",".join(shorts)})
        # weights drift with returns until the next rebalance
        prev_w = w * (1 + r.fillna(0)) / (1 + gross)
    return pd.DataFrame(rows).set_index("month")


def stats(x: pd.Series) -> tuple[float, float, float]:
    if len(x) < 12:
        return np.nan, np.nan, np.nan
    t = x.mean() / x.std() * np.sqrt(len(x))
    return x.mean(), t, x.mean() / x.std() * np.sqrt(12)


def main() -> None:
    holdout = "--holdout" in sys.argv
    px = download_ohlcv(ETFS + ["SPY"], START, END)["adj_close"].sort_index()
    spy = px.pop("SPY")
    print("  first price per ETF:", {c: str(px[c].first_valid_index().date()) for c in px.columns})
    periods = {"design 2000-2022": (DESIGN_START, DESIGN_END), **PERIODS, **(HOLDOUT if holdout else {})}
    last = END if holdout else DESIGN_END
    OUT.mkdir(parents=True, exist_ok=True)

    res, runs = {}, {}
    print(f"\n{'variant':24}{'period':>18}{'months':>8}{'net %/mo':>10}{'t':>7}{'Sharpe':>8}"
          f"{'long leg':>10}{'short leg':>10}")
    for name, look in LOOKBACKS.items():
        for bn in (True, False):
            lab = f"{name} {'beta-neutral' if bn else 'dollar-neutral'}"
            bt = backtest(px, spy, look, bn)
            bt = bt[(bt.index >= pd.Period(DESIGN_START, "M")) & (bt.index <= pd.Period(last, "M"))]
            runs[lab] = bt
            bt.to_csv(OUT / f"{name}_{'beta' if bn else 'dollar'}.csv")
            for p, (a, b) in periods.items():
                x = bt[(bt.index >= pd.Period(a, "M")) & (bt.index <= pd.Period(b, "M"))]
                m, t, sh = stats(x["net"])
                res[(lab, p)] = (m, t)
                print(f"{lab:24}{p:>18}{len(x):>8}{m * 100:>+10.2f}{t:>+7.2f}{sh:>+8.2f}"
                      f"{x['long'].mean() * 100:>+10.2f}{x['short'].mean() * 100:>+10.2f}")
            print()

    prim = runs["12-1 beta-neutral"]
    print("worst 6 months, 12-1 beta-neutral:")
    for mth, r in prim.nsmallest(6, "net").iterrows():
        print(f"  {mth}  net {r.net * 100:+6.2f}%   long {r.longs:16} short {r.shorts}")
    eq = (1 + prim["net"]).cumprod()
    print(f"max drawdown {(eq / eq.cummax() - 1).min():.1%}; mean gross beta to SPY by construction ~0")

    if STREAMS.exists():
        st = pd.read_csv(STREAMS, index_col=0, parse_dates=True)
        st.index = st.index.to_period("M")
        j = st.join(prim["net"].rename("sector mom"), how="inner")
        cols = [c for c in ("core", "trend") if c in j]
        print(f"\ncorrelation of monthly net returns with the live book ({j.index.min()}..{j.index.max()}):")
        for c in cols:
            x = j[[c, "sector mom"]].dropna()
            print(f"  {c:6} {x.corr().iloc[0, 1]:+.2f}  (n={len(x)})")

    m, t = res[("12-1 beta-neutral", "design 2000-2022")]
    m13 = res[("12-1 beta-neutral", "2013-2022")][0]
    ok = m > 0 and t >= 2 and m13 > 0
    print(f"\nPRIMARY (12-1 beta-neutral, net, 2000-2022): {m * 100:+.2f}%/month, t {t:+.2f}; "
          f"2013-2022 {m13 * 100:+.2f}%/month  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
