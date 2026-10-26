"""Insider purchases: do S&P 500 stocks beat their sector after officers or directors buy?

THE IDEA (Jeng, Metrick & Zeckhauser 2003; Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012).
Officers and directors know their company better than anyone else. They sell for many reasons
(diversification, taxes, liquidity, scheduled plans), so sales carry little information; they buy
in the open market for essentially one reason: they think the stock is cheap. Purchases are
reported on Form 4 within two business days and are public, free and point in time.

For the book this would be a market-neutral stream (long the stock, short its sector ETF) at a
monthly horizon, so few trades and low turnover, with no overlap with anything tested so far
(nothing in the documentation uses insider data). If it works only as a tilt, it can also go into
the magic-formula ranking at no extra turnover.

DATA. scripts/insider_data.py -> data/cache/insider_trades.parquet (SEC Form 3/4/5 data sets,
2006 onwards). Matched to the point-in-time S&P 500 by the issuer's trading symbol on the filing,
then by CIK via SEC's current ticker map.

RULE (fixed before the first run, 2026-10-26):
  Qualifying purchase: Form 4, code P, by an officer or a director (a 10%-owner who is neither is
    excluded: funds, not insiders), at least $10,000 in value.
  Event: the first filing date with a qualifying purchase in a stock with no qualifying purchase
    in the previous 63 trading days (later buys in a cluster do not start new events).
  Entry: close of the FIRST session AFTER the filing date (the data sets carry no acceptance time,
    so a filing may have come after the close). Exit: close 63 sessions after entry (MAIN, about
    3 months); 21 and 126 sessions reported alongside.
  Return: stock minus its sector ETF (current GICS sector, SPY if unknown), adjusted closes.
  Universe: S&P 500 member at entry, price >= $5.
  Costs: closing-auction entry and exit, IB commission on a $5k order both sides, 1bp slippage a
    side on the stock and on the ETF, 0.3% a year borrow on the ETF short for the days held.

PRIMARY TEST: the calendar-time portfolio (every active position in equal weight, daily, net of
costs, flat when none is active), monthly returns, 2006-2022. PASS = mean monthly net > 0 with
t >= 2, AND mean > 0 in 2013-2022 on its own. 2023 onwards is held out unless --holdout is passed
(looked at once, after the design period has been judged).

ALSO REPORTED (descriptive, not part of the pass): per-event means with t clustered by entry month;
the same rule for SALES (code S) as a contrast, where little is expected; variants: two or more
insiders buying within 30 days (cluster), CEO or CFO buying, purchases of $100k or more; and the
stream's correlation with the book's monthly streams (results/tail_conditional/monthly_streams.csv),
including its mean in the core's worst decile of months.
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
from earnings_gap_lab import SECTOR_ETF, cik_map  # noqa: E402

START, END = "2005-06-01", "2026-09-01"
DESIGN_START, DESIGN_END = "2006-01-01", "2022-12-31"
MAIN, HORIZONS = 63, (21, 63, 126)
GAP, MIN_VALUE, BIG_VALUE, CLUSTER_DAYS = 63, 10_000.0, 100_000.0, 30
ORDER, COMM_SH, COMM_MIN, SLIP_BP, ETF_BP, BORROW = 5_000.0, 0.0035, 0.35, 1.0, 1.0, 0.003
MIN_PX = 5.0
TRADES = ROOT / "data" / "cache" / "insider_trades.parquet"
STREAMS = ROOT / "results" / "tail_conditional" / "monthly_streams.csv"
OUT = ROOT / "results" / "insider_purchase"


def load_prices():
    tickers = sp500_pit_universe(START, END)
    panel = download_ohlcv(tickers, START, END)
    close_raw = panel["close"].sort_index()
    adj = panel["adj_close"].reindex_like(close_raw)
    keep = adj.notna().sum() > 300
    adj, close_raw = adj.loc[:, keep], close_raw.loc[:, keep]
    print(f"  {adj.shape[1]} stocks with prices")
    cal = adj.index
    member = sp500_pit_eligible(cal, list(adj.columns)).astype(bool)
    sectors = sp1500_constituents().set_index("ticker")["sector"]
    etfs = sorted(set(SECTOR_ETF.values()) | {"SPY"})
    ep = download_ohlcv(etfs, START, END)["adj_close"].reindex(cal)
    etf_of = {t: SECTOR_ETF.get(str(sectors.get(t, "")), "SPY") for t in adj.columns}
    return adj, close_raw, member, ep, etf_of


def attach_ticker(tr: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """Ticker in our price universe: the filing's own symbol first, else the CIK's current ticker."""
    sym = tr["symbol"].str.replace(".", "-", regex=False).str.replace(" ", "", regex=False)
    by_cik = {}
    for t, c in cik_map().items():
        if t in universe:
            by_cik.setdefault(c, t)
    t = sym.where(sym.isin(universe))
    t = t.fillna(tr["issuer_cik"].map(lambda c: by_cik.get(int(c)) if pd.notna(c) else None))
    return tr.assign(ticker=t).dropna(subset=["ticker"])


def qualifying(tr: pd.DataFrame, code: str) -> pd.DataFrame:
    q = tr[(tr["code"] == code) & (tr["officer"] | tr["director"])]
    # one row per stock, filing date and insider: total value, and the insider's role flags
    g = (q.groupby(["ticker", "filing_date", "owner_cik"])
           .agg(value=("value", "sum"), ceo_cfo=("ceo_cfo", "max")).reset_index())
    return g[g["value"] >= MIN_VALUE]


def events(q: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """First qualifying filing per stock after GAP sessions without one; with variant flags."""
    rows = []
    for t, g in q.groupby("ticker"):
        g = g.sort_values("filing_date")
        days = g["filing_date"].drop_duplicates().sort_values()
        idx = cal.searchsorted(days.to_numpy())             # session on/after the filing date
        last = -10 ** 9
        for d, i in zip(days, idx):
            if i - last <= GAP:
                continue
            last = i
            win = g[(g["filing_date"] >= d) & (g["filing_date"] <= d + pd.Timedelta(days=CLUSTER_DAYS))]
            same = g[g["filing_date"] == d]
            rows.append({"ticker": t, "filing": d, "value": same["value"].sum(),
                         "cluster": win["owner_cik"].nunique() >= 2,
                         "ceo_cfo": bool(same["ceo_cfo"].any()),
                         "big": same["value"].sum() >= BIG_VALUE})
    return pd.DataFrame(rows)


def trades(ev: pd.DataFrame, adj, close_raw, member, ep, etf_of, side: int = 1) -> pd.DataFrame:
    cal = adj.index
    A, R, M = adj.to_numpy(), close_raw.to_numpy(), member.to_numpy()
    col = {t: j for j, t in enumerate(adj.columns)}
    E = {e: ep[e].to_numpy() for e in ep.columns}
    out = []
    for r in ev.itertuples(index=False):
        j = col.get(r.ticker)
        if j is None:
            continue
        # entry: the first session strictly after the filing date
        i = cal.searchsorted(r.filing, side="right")
        if i >= len(cal) or not M[i, j] or not np.isfinite(R[i, j]) or R[i, j] < MIN_PX:
            continue
        e = E[etf_of[r.ticker]]
        row = {"entry": cal[i], "ticker": r.ticker, "i": i, "cluster": r.cluster,
               "ceo_cfo": r.ceo_cfo, "big": r.big}
        comm = 2 * max(COMM_MIN, COMM_SH * ORDER / R[i, j]) / ORDER
        for h in HORIZONS:
            if i + h >= len(cal) or not np.all(np.isfinite([A[i, j], A[i + h, j], e[i], e[i + h]])):
                row[f"g{h}"] = row[f"n{h}"] = np.nan
                continue
            g = side * ((A[i + h, j] / A[i, j] - 1) - (e[i + h] / e[i] - 1))
            cost = comm + 2 * (SLIP_BP + ETF_BP) / 1e4 + BORROW * h / 252
            row[f"g{h}"], row[f"n{h}"] = g, g - cost
        out.append(row)
    return pd.DataFrame(out)


def calendar_portfolio(tr: pd.DataFrame, adj, ep, etf_of, h: int, side: int = 1) -> pd.Series:
    """Daily net return of an equal-weight book of all active positions; 0 when none."""
    cal = adj.index
    sr = adj.pct_change(fill_method=None).to_numpy()
    er = ep.pct_change(fill_method=None)
    col = {t: j for j, t in enumerate(adj.columns)}
    tot, cnt = np.zeros(len(cal)), np.zeros(len(cal))
    for r in tr.itertuples(index=False):
        i, j = r.i, col[r.ticker]
        if i + h >= len(cal):
            continue
        x = side * (sr[i + 1:i + h + 1, j] - er[etf_of[r.ticker]].to_numpy()[i + 1:i + h + 1])
        x = np.nan_to_num(x)
        x[-1] -= getattr(r, f"g{h}") - getattr(r, f"n{h}") if np.isfinite(getattr(r, f"g{h}")) else 0.0
        tot[i + 1:i + h + 1] += x
        cnt[i + 1:i + h + 1] += 1
    daily = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0), index=cal)
    return daily, pd.Series(cnt, index=cal)


def monthly(daily: pd.Series) -> pd.Series:
    return (1 + daily).resample("ME").prod() - 1


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan


def t_clustered(x: pd.Series, grp: pd.Series) -> float:
    dev = (x - x.mean()).groupby(grp.values).sum()
    se = np.sqrt((dev ** 2).sum()) / len(x)
    return x.mean() / se if se > 0 else np.nan


def main() -> None:
    holdout = "--holdout" in sys.argv
    end = END if holdout else DESIGN_END
    adj, close_raw, member, ep, etf_of = load_prices()
    tr = attach_ticker(pd.read_parquet(TRADES), set(adj.columns))
    cal = adj.index
    OUT.mkdir(parents=True, exist_ok=True)
    periods = {"design 2006-2022": (DESIGN_START, DESIGN_END), "2006-2012": ("2006-01-01", "2012-12-31"),
               "2013-2022": ("2013-01-01", "2022-12-31"), "2020-2022": ("2020-01-01", "2022-12-31")}
    if holdout:
        periods["2023+ (holdout)"] = ("2023-01-01", END)

    results = {}
    for name, code, side in (("PURCHASES", "P", 1), ("SALES (contrast, short side)", "S", -1)):
        ev = events(qualifying(tr, code), cal)
        t = trades(ev, adj, close_raw, member, ep, etf_of, side)
        t = t[(t.entry >= DESIGN_START) & (t.entry <= end)]
        t.to_csv(OUT / f"events_{code}.csv", index=False)
        print(f"\n=== {name}: {len(t)} events in the S&P 500 ({len(t) / max(t.entry.dt.year.nunique(), 1):.0f} a year)")
        print(f"{'horizon':>8}{'period':>18}{'n':>7}{'gross bp':>10}{'net bp':>9}{'t(month)':>9}{'win':>6}")
        for h in HORIZONS:
            for p, (a, b) in periods.items():
                x = t[(t.entry >= a) & (t.entry <= b)].dropna(subset=[f"n{h}"])
                if len(x) < 20:
                    continue
                tc = t_clustered(x[f"n{h}"], x.entry.dt.to_period("M"))
                print(f"{h:>8}{p:>18}{len(x):>7}{x[f'g{h}'].mean() * 1e4:>+10.1f}{x[f'n{h}'].mean() * 1e4:>+9.1f}"
                      f"{tc:>+9.2f}{(x[f'n{h}'] > 0).mean():>6.0%}")
        if code == "P":
            print("\n  variants (63 sessions, design period, per event):")
            d = t[t.entry <= DESIGN_END].dropna(subset=["n63"])
            for v in ("cluster", "ceo_cfo", "big"):
                for flag in (True, False):
                    x = d[d[v] == flag]
                    if len(x) >= 20:
                        print(f"    {v:8}={str(flag):5}  n {len(x):5}  net {x['n63'].mean() * 1e4:+7.1f}bp  "
                              f"t {t_clustered(x['n63'], x.entry.dt.to_period('M')):+.2f}")
        daily, cnt = calendar_portfolio(t, adj, ep, etf_of, MAIN, side)
        m = monthly(daily[(daily.index >= DESIGN_START) & (daily.index <= end)])
        results[code] = (m, cnt)
        print(f"\n  calendar-time book ({MAIN} sessions, net), monthly:")
        for p, (a, b) in periods.items():
            x = m[(m.index >= a) & (m.index <= b)]
            if len(x) < 12:
                continue
            print(f"    {p:>18}: mean {x.mean():+.2%}/month  t {tstat(x):+.2f}  Sharpe {x.mean() / x.std() * np.sqrt(12):+.2f}"
                  f"  ann vol {x.std() * np.sqrt(12):.1%}")
        c = cnt[(cnt.index >= DESIGN_START) & (cnt.index <= end)]
        print(f"    positions: mean {c.mean():.1f}, median {c.median():.0f}, days with none {(c == 0).mean():.0%}")

    m = results["P"][0]
    m.to_frame("insider_purchases").to_csv(OUT / "monthly_purchases.csv")
    if STREAMS.exists():
        s = pd.read_csv(STREAMS, index_col=0, parse_dates=True)
        s.index = s.index + pd.offsets.MonthEnd(0)
        j = s.join(m.rename("insider"), how="inner")
        print("\n  correlation with the book's monthly streams (overlap only):")
        for c in [c for c in s.columns if "PROXY" not in c]:
            x = j[[c, "insider"]].dropna()
            if len(x) >= 24:
                print(f"    {c:22} {x.corr().iloc[0, 1]:+.2f}  ({len(x)} months)")
        x = j[["core", "insider"]].dropna()
        if len(x) >= 60:
            worst = x["core"] <= x["core"].quantile(0.10)
            print(f"    in the core's worst decile ({worst.sum()} months): insider book {x.loc[worst, 'insider'].mean():+.2%}/month,"
                  f" core {x.loc[worst, 'core'].mean():+.2%}")

    d = m[(m.index >= DESIGN_START) & (m.index <= DESIGN_END)]
    d2 = m[(m.index >= "2013-01-01") & (m.index <= DESIGN_END)]
    ok = d.mean() > 0 and tstat(d) >= 2 and d2.mean() > 0
    print(f"\nPRIMARY (calendar-time, {MAIN} sessions, net, 2006-2022): mean {d.mean():+.2%}/month, t {tstat(d):+.2f}; "
          f"2013-2022 {d2.mean():+.2%}/month (t {tstat(d2):+.2f})  ->  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
