"""Earnings-announcement premium: do stocks beat their sector around a SCHEDULED announcement?

THE IDEA (Frazzini & Lamont 2007; Barber et al. 2013; Savor & Wilson 2016). Individual earnings
moves go both ways, but on average announcing stocks beat the market around the event: payment for
carrying announcement risk, attention-driven buying, and disagreement with limited short selling.
The trade is always LONG the announcer and SHORT its sector ETF in equal dollars, so the market's
direction drops out; the bet is on the average, so it needs many small positions.

NO LOOK-AHEAD IN THE DATES. A company's confirmed date is known weeks ahead live, but no free
history of confirmed dates exists, and the actual 8-K dates include unscheduled profit warnings
nobody could have traded in advance. So every trade is set from a PREDICTED date only: each
earnings reaction day R (EDGAR 8-K Item 2.02, timed by acceptance, as in earnings_gap_lab.py)
projects a predicted reaction day P = R + 364 days (same weekday a year later, the same quarter).
The outcome is whatever actually happened; the share of P with a real announcement within one
trading day is reported.

WINDOWS (fixed before the first run), relative to the predicted reaction session P, all
close-to-close, so every trade is entered and exited in the closing auction:
  before    close P-3 -> close P-1   the run-up, exited before the release (MAIN)
  through   close P-3 -> close P     including the reaction session
  after     close P   -> close P+3   the days after
Return = stock minus its sector ETF over the window (current GICS sector; SPY if unknown).

COSTS per trade: closing-auction fills, so no spread; 1bp per side allowance for auction slippage on
the stock, IB tiered commission on a $5k order ($0.0035/share, $0.35 minimum) both sides, 1bp per
side on the ETF leg.

CONTROL. The same stocks over the same windows around a pseudo date 30 trading days after P, when no
announcement is scheduled. If announcers simply drift up with their sector's beta, the control
shows it too.

UNIVERSE. S&P 500 point in time (a stock trades only while a member at entry), price >= $5.

PRIMARY TEST (fixed before the first run): window "before", net, 2005-2022. PASS = mean > 0 with
date-clustered t >= 2, AND > 0 in 2020-2022 on its own, AND above the control (t >= 2).
"through" and "after" are descriptive. 2023 onwards is held out unless --holdout is passed.

Also reported: a daily portfolio holding every active position in equal weight (out of the market
when none is active), as a first view of what the stream looks like as a sleeve.

Usage:  python scripts/earnings_premium_lab.py [--holdout]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.loaders import download_ohlcv  # noqa: E402
from data.universe import sp1500_constituents, sp500_pit_eligible, sp500_pit_universe  # noqa: E402
from earnings_gap_lab import END, SECTOR_ETF, START, cik_map, events  # noqa: E402

WINDOWS = {"before": (-3, -1), "through": (-3, 0), "after": (0, 3)}
MAIN = "before"
CONTROL_SHIFT = 30
ORDER, COMM_SH, COMM_MIN, SLIP_BP, ETF_BP = 5_000.0, 0.0035, 0.35, 1.0, 1.0
MIN_PX = 5.0
DESIGN_END = "2022-12-31"
PERIODS = {"2005-2012": ("2005-01-01", "2012-12-31"), "2013-2019": ("2013-01-01", "2019-12-31"),
           "2020-2022": ("2020-01-01", "2022-12-31")}
HOLDOUT = {"2023+ (holdout)": ("2023-01-01", END)}
OUT = ROOT / "results" / "earnings_premium"


def load():
    tickers = sp500_pit_universe(START, END)
    panel = download_ohlcv(tickers, START, END)
    close_raw = panel["close"].sort_index()
    adj = panel["adj_close"].reindex_like(close_raw)
    keep = adj.notna().sum() > 300
    adj, close_raw = adj.loc[:, keep], close_raw.loc[:, keep]
    print(f"  {adj.shape[1]} stocks with prices (expect ~670; fewer means a throttled download)")
    cal = adj.index
    member = sp500_pit_eligible(cal, list(adj.columns)).astype(bool)
    sectors = sp1500_constituents().set_index("ticker")["sector"]
    etfs = sorted(set(SECTOR_ETF.values()) | {"SPY"})
    ep = download_ohlcv(etfs, START, END)["adj_close"].reindex(cal)
    etf_of = {t: SECTOR_ETF.get(str(sectors.get(t, "")), "SPY") for t in adj.columns}
    return adj, close_raw, member, ep, etf_of


def trades(adj, close_raw, member, ep, etf_of, react: pd.DataFrame, shift: int = 0) -> pd.DataFrame:
    cal = adj.index
    A, R = adj.to_numpy(), close_raw.to_numpy()
    M = member.to_numpy()
    col = {t: j for j, t in enumerate(adj.columns)}
    E = {e: ep[e].to_numpy() for e in ep.columns}
    actual = {t: set(cal.get_indexer(g["day"])) for t, g in react.groupby("ticker")}
    rows = []
    for t, d in react.itertuples(index=False):
        if t not in col:
            continue
        p = cal.searchsorted(d + pd.Timedelta(days=364)) + shift       # predicted session index
        lo = p + min(w[0] for w in WINDOWS.values())
        hi = p + max(w[1] for w in WINDOWS.values())
        if lo < 1 or hi >= len(cal):
            continue
        j, e = col[t], E[etf_of[t]]
        if not M[p - 3, j] or not np.isfinite(R[p - 3, j]) or R[p - 3, j] < MIN_PX:
            continue
        acts = actual.get(t, set())
        row = {"pred": cal[p], "ticker": t, "hit": any(abs(a - p) <= 1 for a in acts)}
        shares = ORDER / R[p - 3, j]
        comm = 2 * max(COMM_MIN, COMM_SH * shares) / ORDER
        cost = comm + 2 * SLIP_BP / 1e4 + 2 * ETF_BP / 1e4
        ok = True
        for w, (a, b) in WINDOWS.items():
            s0, s1, e0, e1 = A[p + a, j], A[p + b, j], e[p + a], e[p + b]
            if not np.all(np.isfinite([s0, s1, e0, e1])):
                ok = False
                break
            g = (s1 / s0 - 1) - (e1 / e0 - 1)
            row[f"{w}_gross"], row[f"{w}_net"] = g, g - cost
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def t_clustered(x: pd.Series, day: pd.Series) -> float:
    dev = (x - x.mean()).groupby(day.values).sum()
    se = np.sqrt((dev ** 2).sum()) / len(x)
    return x.mean() / se if se > 0 else np.nan


def portfolio(adj, ep, etf_of, tr: pd.DataFrame, window: str) -> pd.Series:
    """Daily return of an equal-weight book of every active position (0 when none is active)."""
    a, b = WINDOWS[window]
    cal = adj.index
    rets = adj.pct_change(fill_method=None)
    er = ep.pct_change(fill_method=None)
    pos = cal.get_indexer(tr["pred"])
    tot, cnt = pd.Series(0.0, index=cal), pd.Series(0, index=cal)
    cost = tr[f"{window}_gross"] - tr[f"{window}_net"]
    for (t, p, c) in zip(tr["ticker"], pos, cost):
        days = cal[p + a + 1: p + b + 1]
        r = rets.loc[days, t] - er.loc[days, etf_of[t]]
        r.iloc[-1] -= c
        tot.loc[days] += r.fillna(0.0).to_numpy()
        cnt.loc[days] += 1
    return (tot / cnt.replace(0, np.nan)).fillna(0.0), cnt


def main() -> None:
    holdout = "--holdout" in sys.argv
    adj, close_raw, member, ep, etf_of = load()
    react, counts = events(list(adj.columns), cik_map(), adj.index)
    print(f"  8-K 2.02 filings {counts['filings']}, reaction days {len(react)}")
    ev = trades(adj, close_raw, member, ep, etf_of, react)
    ctl = trades(adj, close_raw, member, ep, etf_of, react, shift=CONTROL_SHIFT)
    if not holdout:
        ev, ctl = ev[ev.pred <= DESIGN_END], ctl[ctl.pred <= DESIGN_END]
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "events.csv", index=False)
    print(f"  predicted events {len(ev)} ({len(ev) / ev.pred.dt.year.nunique():.0f} a year); real announcement "
          f"within one session of the prediction: {ev.hit.mean():.0%}")

    periods = {"design 2005-2022": ("2005-01-01", DESIGN_END), **PERIODS, **(HOLDOUT if holdout else {})}
    print(f"\n{'window':9}{'kind':9}{'period':>18}{'n':>7}{'gross bp':>10}{'net bp':>9}{'t(net)':>8}{'win':>6}")
    res = {}
    for w in WINDOWS:
        for kind, df in (("event", ev), ("control", ctl)):
            for p, (a, b) in periods.items():
                x = df[(df.pred >= a) & (df.pred <= b)]
                if len(x) < 20:
                    continue
                t = t_clustered(x[f"{w}_net"], x["pred"])
                res[(w, kind, p)] = (x[f"{w}_net"].mean(), t)
                print(f"{w:9}{kind:9}{p:>18}{len(x):>7}{x[f'{w}_gross'].mean() * 1e4:>+10.1f}"
                      f"{x[f'{w}_net'].mean() * 1e4:>+9.1f}{t:>+8.2f}{(x[f'{w}_net'] > 0).mean():>6.0%}")
        print()

    print("Daily book of all active positions (equal weight, out of the market when none):")
    for w in WINDOWS:
        r, cnt = portfolio(adj, ep, etf_of, ev, w)
        r = r[r.index <= (END if holdout else DESIGN_END)]
        r = r[r.index >= "2005-01-01"]
        act = cnt.loc[r.index]
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        print(f"  {w:8} Sharpe {sh:+.2f}   ann {r.mean() * 252:+.1%}   vol {r.std() * np.sqrt(252):.1%}   "
              f"days in market {(act > 0).mean():.0%}   mean positions when in {act[act > 0].mean():.0f}")

    p = "design 2005-2022"
    m, t = res[(MAIN, "event", p)]
    m20 = res[(MAIN, "event", "2020-2022")][0]
    e = ev[ev.pred <= DESIGN_END][f"{MAIN}_net"]
    c = ctl[ctl.pred <= DESIGN_END][f"{MAIN}_net"]
    td = (e.mean() - c.mean()) / np.sqrt(e.var() / len(e) + c.var() / len(c))
    ok = m > 0 and t >= 2 and m20 > 0 and td >= 2
    print(f"\nPRIMARY ({MAIN}, net, 2005-2022): mean {m * 1e4:+.1f}bp, t {t:+.2f}; 2020-2022 {m20 * 1e4:+.1f}bp; "
          f"event minus control {(e.mean() - c.mean()) * 1e4:+.1f}bp (t {td:+.2f})  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
