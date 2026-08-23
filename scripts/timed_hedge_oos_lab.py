"""Lab: does the vol-spike beta hedge survive 1993-2011, out of sample?

timed_hedge_speed_lab.py found that hedging on a realised-vol spike beat never hedging
over 2012-2026 (+38.9% terminal wealth, Sharpe 1.00 -> 1.21, maxDD -36% -> -24%), and it
survived a parameter-surface sweep, a holdout split and COVID removal. But it rested on
6 episodes in one regime, and it has a known blind spot: it fires on FAST vol expansions,
and was on 0% of the 2022 slow-grind bear.

1993-2011 is genuine out-of-sample and contains exactly what the first sample lacks:
the 2000-2002 dot-com decline (a slow grind, the blind-spot case) and 2008 (a severe
crash). If the signal only works on fast crashes, 2000-2002 will show it.

WHY SPY IS A VALID PROXY FOR THE LONG BOOK. The hedge is
    hedged = long - on * beta * SPY
and the magic formula decomposes as long = alpha + beta*SPY with beta 1.05 and alpha
uncorrelated to SPY. The hedge decision therefore acts ONLY on the beta term; the alpha
term passes through untouched whatever the signal does. So testing on SPY isolates the
entire effect of the hedge. The same SPY-proxy test is also run on 2012-2026 so the two
periods are compared like for like, not against the magic-formula-based numbers.

Parameters are taken from the CENTRE of the robust region found in-sample (the fast=5
family, 26-35 episodes) rather than the peak cell (10/60, 6 episodes), which is the
discipline that survived in the VRP work.

Run: python scripts/timed_hedge_oos_lab.py
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

TOGGLE_BPS = 3.0
CAP = 300_000
CONFIGS = [(5, 40, 2.0), (5, 60, 2.0), (5, 90, 2.0), (5, 120, 2.0), (10, 60, 2.0)]

CRISES = [
    ("1997 Asia",      "1997-10-01", "1997-11-30"),
    ("1998 LTCM",      "1998-07-15", "1998-10-15"),
    ("2000-02 dotcom", "2000-03-24", "2002-10-09"),
    ("2008 GFC",       "2008-09-01", "2009-03-31"),
    ("2011 downgrade", "2011-07-01", "2011-10-31"),
    ("2015-16",        "2015-08-01", "2016-02-29"),
    ("2018 Q4",        "2018-10-01", "2018-12-31"),
    ("2020 COVID",     "2020-02-15", "2020-03-31"),
    ("2022 bear",      "2022-01-01", "2022-12-31"),
]


def spy() -> pd.Series:
    px = yf.download("SPY", start="1993-01-01", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.dropna()


def hedge_on(px: pd.Series, fast: int, slow: int, k: float) -> pd.Series:
    r = px.pct_change()
    return ((r.rolling(fast).std() > k * r.rolling(slow).std())
            .shift(1).fillna(False).astype(bool))


def run(px: pd.Series, on: pd.Series, a: str, b: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(buy&hold, hedged, on) over [a, b]. Hedged flattens the beta while on."""
    r = px.pct_change().loc[a:b].fillna(0.0)
    o = on.reindex(r.index).fillna(False).astype(bool)
    w = o.astype(float)
    toggles = w.diff().abs().fillna(0.0)
    return r, r * (1 - w) - toggles * TOGGLE_BPS / 1e4, o


def report(px: pd.Series, a: str, b: str, title: str) -> None:
    print("\n" + "=" * 122)
    print(f"{title}   ({a} -> {b})")
    print("=" * 122)
    bh = px.pct_change().loc[a:b].fillna(0.0)
    s = summary_stats(bh)
    print(f"  {'variant':22s} {'CAGR':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s} "
          f"{'final $':>12s} {'on%':>6s} {'eps':>5s}")
    print(f"  {'SPY buy & hold':22s} {s['ann_return']:>+8.2%} {s['ann_vol']:>7.2%} "
          f"{s['sharpe']:>+7.2f} {s['max_drawdown']:>+8.1%} "
          f"{CAP*(1+bh).prod():>12,.0f} {0:>6.1%} {0:>5d}")
    for fast, slow, k in CONFIGS:
        on = hedge_on(px, fast, slow, k)
        _, hd, o = run(px, on, a, b)
        st = summary_stats(hd)
        eps = int(o.astype(float).diff().clip(lower=0).sum())
        flag = "  <-- peak cell (in-sample)" if (fast, slow) == (10, 60) else ""
        print(f"  {f'vol {fast}/{slow} k={k}':22s} {st['ann_return']:>+8.2%} {st['ann_vol']:>7.2%} "
              f"{st['sharpe']:>+7.2f} {st['max_drawdown']:>+8.1%} "
              f"{CAP*(1+hd).prod():>12,.0f} {o.mean():>6.1%} {eps:>5d}{flag}")


def crisis_table(px: pd.Series) -> None:
    print("\n" + "=" * 122)
    print("CRISIS WINDOWS — cumulative return, and how much of each window the signal was ON")
    print("=" * 122)
    cfgs = [(5, 60, 2.0), (5, 90, 2.0), (10, 60, 2.0)]
    hdr = f"  {'window':18s} {'SPY':>9s}"
    for f_, s_, k in cfgs:
        hdr += f"{f'{f_}/{s_}':>11s}{'on%':>7s}"
    print(hdr)
    for lbl, a, b in CRISES:
        r = px.pct_change().loc[a:b].fillna(0.0)
        if len(r) < 5:
            continue
        line = f"  {lbl:18s} {((1+r).prod()-1):>+9.1%}"
        for f_, s_, k in cfgs:
            on = hedge_on(px, f_, s_, k)
            _, hd, o = run(px, on, a, b)
            line += f"{((1+hd).prod()-1):>+11.1%}{o.mean():>7.0%}"
        print(line)


def main() -> None:
    px = spy()
    print(f"[data] SPY {px.index[0].date()} -> {px.index[-1].date()}  ({len(px)} days)")
    report(px, "1993-01-01", "2011-12-31", "OUT OF SAMPLE — 1993-2011 (never used to design the signal)")
    report(px, "2012-01-01", "2026-12-31", "IN SAMPLE — 2012-2026 (SPY proxy, same construction)")
    report(px, "1993-01-01", "2026-12-31", "FULL HISTORY")
    crisis_table(px)

    print("\n" + "=" * 122)
    print("THE BLIND-SPOT QUESTION")
    print("=" * 122)
    on = hedge_on(px, 5, 60, 2.0)
    for lbl, a, b in [("2000-02 dotcom (slow grind)", "2000-03-24", "2002-10-09"),
                      ("2008 GFC (severe crash)", "2008-09-01", "2009-03-31"),
                      ("2022 bear (slow grind)", "2022-01-01", "2022-12-31")]:
        _, hd, o = run(px, on, a, b)
        bh = px.pct_change().loc[a:b].fillna(0.0)
        print(f"  {lbl:30s} on {o.mean():>5.1%} of days | SPY {((1+bh).prod()-1):>+7.1%} "
              f"-> hedged {((1+hd).prod()-1):>+7.1%}  (saved {((1+hd).prod()-(1+bh).prod()):>+6.1%})")

    out = ROOT / "results" / "timed_hedge"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for period, a, b in [("oos_1993_2011", "1993-01-01", "2011-12-31"),
                         ("is_2012_2026", "2012-01-01", "2026-12-31")]:
        for fast, slow, k in CONFIGS:
            _, hd, o = run(px, hedge_on(px, fast, slow, k), a, b)
            st = summary_stats(hd)
            rows.append({"period": period, "fast": fast, "slow": slow, "k": k,
                         "cagr": st["ann_return"], "sharpe": st["sharpe"],
                         "maxdd": st["max_drawdown"], "on_pct": o.mean()})
    pd.DataFrame(rows).to_csv(out / "oos_1993_2011.csv", index=False)
    print(f"\n  wrote {out}/oos_1993_2011.csv")


if __name__ == "__main__":
    main()
