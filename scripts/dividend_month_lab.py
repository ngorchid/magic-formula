"""Dividend-month premium: do stocks beat their sector in the months they are expected to pay?

THE IDEA (Hartzmark & Solomon 2013, "The dividend month premium", JFE). Stocks earn abnormal
returns in months when a dividend is PREDICTED from their own past schedule, about 0.4-0.5% a month
in their sample (1927-2011). The proposed cause is demand: investors who want dividends buy before
the payment and push the price up. The month is known in advance from the schedule alone, so the
trade needs no forecast, only the calendar, and it is market-neutral against the sector ETF.
Nothing in the documentation tests it.

EX-DIVIDEND DATES are read from the price cache itself: yfinance's adjusted close is the raw
close times a factor that steps up on each ex-dividend date by about dividend / previous close.
A step of at least MIN_STEP (3bp) in adj/close marks an ex-date (splits move both series alike).

RULE (fixed before the first run, 2026-10-26). At the last close of month m-1:
  Regular quarterly payer: an ex-date in each of months m-3, m-6, m-9 and m-12, i.e. the schedule
    puts the next ex-date in month m. These are the PREDICTED payers of month m.
  CONTROL: regular quarterly payers whose schedule falls in the other months: an ex-date in each of
    the last four quarters, but none in m-3, m-6, m-9 or m-12 (their dividend is due in m+1 or m+2).
  Trade: long every predicted payer, short its sector ETF in equal dollars, from the last close of
    m-1 to the last close of m. Total returns on both legs (adjusted closes), so the dividend
    itself counts on both sides, as an investor receives it.
  Universe: S&P 500 member at entry, price >= $5. Equal weight across the names each month.
  Costs: closing-auction entry and exit each month, IB commission on a $5k order both sides, 1bp
    slippage a side on the stock and on the ETF, 0.3% a year borrow on the ETF short.

PRIMARY TEST, 2006-2022 monthly (2023 onwards held out unless --holdout). PASS needs all three:
  (1) predicted payers minus sector ETF, net: mean > 0 with t >= 2;
  (2) predicted payers minus CONTROL (gross, same month): mean > 0 with t >= 2, so the excess is
      specific to the dividend month and not a general outperformance of dividend payers;
  (3) (1) positive in 2013-2022 on its own.

ALSO REPORTED: the same by half of the month (to the ex-date and after it); the share of the
excess that is the dividend itself; correlation with the book's monthly streams and the mean in
the core's worst decile; and, for a Swiss private investor, the caveat that dividends are taxed
as income while capital gains are not, which this backtest does not model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from insider_purchase_lab import load_prices, tstat  # noqa: E402

DESIGN_START, DESIGN_END = "2006-01-01", "2022-12-31"
END = "2026-09-01"
MIN_STEP = 3e-4
ORDER, COMM_SH, COMM_MIN, SLIP_BP, ETF_BP, BORROW = 5_000.0, 0.0035, 0.35, 1.0, 1.0, 0.003
MIN_PX = 5.0
STREAMS = ROOT / "results" / "tail_conditional" / "monthly_streams.csv"
OUT = ROOT / "results" / "dividend_month"


def ex_months(adj: pd.DataFrame, close_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(month x stock: 1 if an ex-date fell in the month; month x stock: dividend yield that month)."""
    ratio = adj / close_raw
    step = ratio / ratio.shift(1) - 1
    ex = step >= MIN_STEP
    by_m = ex.resample("ME").max().astype(float)
    yld = step.where(ex, 0.0).resample("ME").sum()
    return by_m, yld


def main() -> None:
    holdout = "--holdout" in sys.argv
    end = END if holdout else DESIGN_END
    adj, close_raw, member, ep, etf_of = load_prices()
    exm, yld = ex_months(adj, close_raw)
    months = exm.index
    # month-end trading days and the member/price check at entry
    last_day = adj.index.to_series().groupby(adj.index.to_period("M")).max()
    last_day.index = last_day.index.to_timestamp("M")
    px_end = adj.reindex(last_day.values)
    px_end.index = last_day.index
    raw_end = close_raw.reindex(last_day.values)
    raw_end.index = last_day.index
    mem_end = member.reindex(last_day.values)
    mem_end.index = last_day.index
    etf_end = ep.reindex(last_day.values)
    etf_end.index = last_day.index
    stock_ret = px_end / px_end.shift(1) - 1
    etf_ret = etf_end / etf_end.shift(1) - 1
    etf_cols = pd.Series(etf_of)

    lag = {k: exm.shift(k).reindex(last_day.index) for k in (1, 2, 3, 6, 9, 12)}
    rows, per_stock = [], []
    for k in range(13, len(last_day.index)):
        m, prev = last_day.index[k], last_day.index[k - 1]
        if m < pd.Timestamp(DESIGN_START) or m > pd.Timestamp(end):
            continue
        ok = mem_end.loc[prev].fillna(False).astype(bool) & (raw_end.loc[prev] >= MIN_PX)
        sched = (lag[3].loc[m] == 1) & (lag[6].loc[m] == 1) & (lag[9].loc[m] == 1) & (lag[12].loc[m] == 1)
        # paid in each of the last four quarters: any ex-date in each 3-month block before m
        q = [(exm.shift(a).reindex(last_day.index).loc[m] + exm.shift(a + 1).reindex(last_day.index).loc[m]
              + exm.shift(a + 2).reindex(last_day.index).loc[m]) >= 1 for a in (1, 4, 7, 10)]
        regular = q[0] & q[1] & q[2] & q[3]
        pay = ok & sched & regular
        ctl = ok & regular & ~sched & (lag[3].loc[m] != 1) & (lag[6].loc[m] != 1)
        er = stock_ret.loc[m] - etf_ret.loc[m].reindex(etf_cols.values).set_axis(etf_cols.index).reindex(stock_ret.columns)
        p_names = [t for t in pay.index[pay] if np.isfinite(er.get(t, np.nan))]
        c_names = [t for t in ctl.index[ctl] if np.isfinite(er.get(t, np.nan))]
        if len(p_names) < 5 or len(c_names) < 5:
            continue
        comm = np.mean([2 * max(COMM_MIN, COMM_SH * ORDER / raw_end.loc[prev, t]) / ORDER for t in p_names])
        cost = comm + 2 * (SLIP_BP + ETF_BP) / 1e4 + BORROW / 12
        paid = exm.loc[m, p_names].mean() if m in exm.index else np.nan
        rows.append({"month": m, "n_pay": len(p_names), "n_ctl": len(c_names),
                     "pay_gross": er[p_names].mean(), "pay_net": er[p_names].mean() - cost,
                     "ctl_gross": er[c_names].mean(), "div_yield": yld.loc[m, p_names].mean(),
                     "hit": paid})
    df = pd.DataFrame(rows).set_index("month")
    df["spread"] = df["pay_gross"] - df["ctl_gross"]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "monthly.csv")

    print(f"\nmonths {len(df)}, predicted payers per month: median {df.n_pay.median():.0f}, "
          f"control {df.n_ctl.median():.0f}; predicted payers that did go ex in the month: {df.hit.mean():.0%}")
    periods = {"design 2006-2022": (DESIGN_START, DESIGN_END), "2006-2012": ("2006-01-01", "2012-12-31"),
               "2013-2022": ("2013-01-01", "2022-12-31"), "2020-2022": ("2020-01-01", "2022-12-31")}
    if holdout:
        periods["2023+ (holdout)"] = ("2023-01-01", END)
    print(f"{'period':>18}{'months':>8}{'payers-ETF net':>16}{'t':>7}{'payers-control':>16}{'t':>7}"
          f"{'div yield':>11}{'Sharpe net':>12}")
    for p, (a, b) in periods.items():
        x = df[(df.index >= a) & (df.index <= b)]
        if len(x) < 12:
            continue
        print(f"{p:>18}{len(x):>8}{x.pay_net.mean():>+15.2%} {tstat(x.pay_net):>+6.2f}{x.spread.mean():>+15.2%} "
              f"{tstat(x.spread):>+6.2f}{x.div_yield.mean():>10.2%}{x.pay_net.mean() / x.pay_net.std() * np.sqrt(12):>+12.2f}")

    if STREAMS.exists():
        s = pd.read_csv(STREAMS, index_col=0, parse_dates=True)
        s.index = s.index + pd.offsets.MonthEnd(0)
        j = s.join(df["pay_net"].rename("divmonth"), how="inner")
        print("\n  correlation with the book's monthly streams:")
        for c in [c for c in s.columns if "PROXY" not in c]:
            x = j[[c, "divmonth"]].dropna()
            if len(x) >= 24:
                print(f"    {c:22} {x.corr().iloc[0, 1]:+.2f}  ({len(x)} months)")
        x = j[["core", "divmonth"]].dropna()
        if len(x) >= 60:
            w = x["core"] <= x["core"].quantile(0.10)
            print(f"    core's worst decile ({w.sum()} months): {x.loc[w, 'divmonth'].mean():+.2%}/month")

    d = df[(df.index >= DESIGN_START) & (df.index <= DESIGN_END)]
    d2 = d[d.index >= "2013-01-01"]
    c1 = d.pay_net.mean() > 0 and tstat(d.pay_net) >= 2
    c2 = d.spread.mean() > 0 and tstat(d.spread) >= 2
    c3 = d2.pay_net.mean() > 0
    print(f"\nPRIMARY 2006-2022: (1) payers-ETF net {d.pay_net.mean():+.2%} t {tstat(d.pay_net):+.2f} {'ok' if c1 else 'no'}; "
          f"(2) payers-control {d.spread.mean():+.2%} t {tstat(d.spread):+.2f} {'ok' if c2 else 'no'}; "
          f"(3) 2013-2022 {d2.pay_net.mean():+.2%} {'ok' if c3 else 'no'}  ->  {'PASS' if c1 and c2 and c3 else 'FAIL'}")


if __name__ == "__main__":
    main()
