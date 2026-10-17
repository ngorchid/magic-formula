"""Buy-on-gap: a one-day mean-reversion trade on S&P 500 stocks that open sharply down.

Ernie Chan, "Algorithmic Trading" (2013), Example 4.1, reported on the S&P 500 for 2006-2011, so
2013 onwards is out of sample with respect to the publication.

THE RULE, each trading day t:
  1. gap_t = open_t / low_{t-1} - 1. Keep the stocks with gap_t < -sd_t, where sd_t is the
     standard deviation of daily close-to-close returns over the 90 days ending t-1.
  2. Keep those with open_t > the 20-day moving average of closes ending t-1.
  3. Buy the 10 with the lowest gap_t (all of them if fewer than 10) at the open.
  4. Sell everything at the close. No position is held overnight.

Two capital conventions, both reported:
  slots   each pick gets 1/10 of the budget; unused slots sit in cash (what a live sleeve does)
  full    the budget is split equally over however many picks there are (Chan's convention)

DATA. yfinance daily OHLC for every stock ever in the S&P 500 since 2004 (fja05680 point-in-time
membership; a stock is eligible only on days it was a member). All four prices are scaled by
adj_close / close for that day, so splits and dividends do not create false gaps. Former members
with no yfinance history (~310, disproportionately acquired or failed names) cannot be picked:
the survivorship bias here works in the strategy's favour, since a stock that gaps down and then
fails is exactly what the rule buys.

COSTS. Two trades per pick per day, the open (market-on-open auction) and the close
(market-on-close). IB tiered commission $0.0035/share, $0.35 minimum per order, on a stated
budget, plus a slippage scenario in bp per side for the difference between the yfinance open and
an auction fill. The yfinance open is not guaranteed to be the official auction price, so the
0bp line is optimistic.

Usage:  python scripts/buy_on_gap_lab.py [budget]     (default $50,000)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loaders import download_ohlcv  # noqa: E402
from data.universe import sp500_pit_eligible, sp500_pit_universe  # noqa: E402

START, END = "2004-01-01", "2026-09-01"
SD_WIN, MA_WIN, TOP_N = 90, 20, 10
PUB_END, OOS_START = "2011-12-31", "2013-01-01"
COMM_PER_SHARE, COMM_MIN = 0.0035, 0.35
SLIPPAGE_BP = (0.0, 2.0, 5.0)
OUT = Path("results/buy_on_gap")


def load() -> dict[str, pd.DataFrame]:
    tickers = sp500_pit_universe(START, END)
    panel = download_ohlcv(tickers, START, END)
    close = panel["close"].sort_index()
    f = (panel["adj_close"].reindex_like(close) / close).where(lambda x: x > 0)
    out = {k: panel[k].reindex_like(close) * f for k in ("open", "high", "low")}
    out["close"] = close * f
    out["raw_open"] = panel["open"].reindex_like(close)
    keep = out["close"].notna().sum() > SD_WIN
    out = {k: v.loc[:, keep] for k, v in out.items()}
    out["member"] = sp500_pit_eligible(out["close"].index, list(out["close"].columns)).astype(bool)
    return out


def picks(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Boolean frame: True where the stock is bought at today's open."""
    o, lo, c = d["open"], d["low"], d["close"]
    sd = c.pct_change(fill_method=None).rolling(SD_WIN, min_periods=SD_WIN).std().shift(1)
    ma = c.rolling(MA_WIN, min_periods=MA_WIN).mean().shift(1)
    gap = o / lo.shift(1) - 1.0
    ok = d["member"] & (gap < -sd) & (o > ma) & c.notna() & o.notna() & (o > 0)
    g = gap.where(ok)
    rank = g.rank(axis=1, method="first")
    return rank <= TOP_N


def daily_returns(d: dict[str, pd.DataFrame], sel: pd.DataFrame, budget: float) -> pd.DataFrame:
    intraday = (d["close"] / d["open"] - 1.0).where(sel)
    n = sel.sum(axis=1)
    # Commission: shares = notional / raw price; two orders per pick per day.
    raw = d["raw_open"]
    out = {}
    for conv in ("slots", "full"):
        notional = (budget / TOP_N) if conv == "slots" else (budget / n.replace(0, np.nan))
        w = (1.0 / TOP_N) if conv == "slots" else (1.0 / n.replace(0, np.nan))
        gross = (intraday.sum(axis=1) * w).fillna(0.0)
        shares = sel.mul(notional, axis=0) / raw
        comm = (np.maximum(shares * COMM_PER_SHARE, COMM_MIN).where(sel)).sum(axis=1) * 2 / budget
        out[f"{conv}_gross"] = gross
        out[f"{conv}_comm"] = comm.fillna(0.0)
        out[f"{conv}_exposure"] = (n * w).fillna(0.0).clip(upper=1.0)
    out["n_picks"] = n
    out["universe_intraday"] = (d["close"] / d["open"] - 1.0).where(d["member"]).mean(axis=1)
    return pd.DataFrame(out)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() else np.nan
    return {"Sharpe": ann / vol if vol else np.nan, "ann": ann, "vol": vol,
            "maxDD": float((cum / cum.cummax() - 1).min()), "t": t}


def net(df: pd.DataFrame, conv: str, slip_bp: float) -> pd.Series:
    return df[f"{conv}_gross"] - df[f"{conv}_comm"] - 2 * slip_bp / 1e4 * df[f"{conv}_exposure"]


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 50_000.0
    d = load()
    print(f"[load] {d['close'].shape[1]} stocks with prices, {d['close'].index[0].date()} "
          f"to {d['close'].index[-1].date()}")
    sel = picks(d)
    df = daily_returns(d, sel, budget).iloc[SD_WIN + 1:]
    days = len(df)
    print(f"  days with >= 1 pick: {(df.n_picks > 0).mean():.0%}; mean picks {df.n_picks.mean():.1f}; "
          f"days with the full 10: {(df.n_picks >= TOP_N).mean():.0%}")

    periods = {"full": (None, None), "Chan's sample 2006-2011": ("2006-01-01", PUB_END),
               "post-publication 2013+": (OOS_START, None), "2020+": ("2020-01-01", None)}
    rows = []
    for conv in ("slots", "full"):
        for slip in SLIPPAGE_BP:
            label = "gross" if slip == 0 and False else f"net, {slip:.0f}bp/side"
            series = {"gross": df[f"{conv}_gross"], label: net(df, conv, slip)} if slip == 0 else \
                {label: net(df, conv, slip)}
            for name, s in series.items():
                for p, (a, b) in periods.items():
                    st = stats(s.loc[a:b])
                    rows.append({"capital": conv, "returns": name, "period": p, **st})
    res = pd.DataFrame(rows).drop_duplicates(["capital", "returns", "period"])
    pd.set_option("display.width", 200)
    print(f"\nBudget ${budget:,.0f}; commission IB tiered ${COMM_PER_SHARE}/share, ${COMM_MIN} min per order")
    print(res.to_string(index=False, formatters={
        "Sharpe": "{:+.2f}".format, "ann": "{:+.2%}".format, "vol": "{:.2%}".format,
        "maxDD": "{:+.1%}".format, "t": "{:+.2f}".format}))

    # Is it selection or just intraday beta? Picks vs the average member on the same day.
    pick_days = df.n_picks > 0
    per_pick = (df["full_gross"] - df["universe_intraday"])[pick_days]
    print("\nSelection vs the average S&P member's open-to-close on the same day (full capital, gross):")
    for p, (a, b) in periods.items():
        x = per_pick.loc[a:b]
        print(f"  {p:26} mean {x.mean() * 1e4:+6.1f} bp/day  t {x.mean() / x.std() * np.sqrt(len(x)):+.2f}"
              f"  corr(picks, universe) {df.loc[a:b][pick_days.loc[a:b]][['full_gross', 'universe_intraday']].corr().iloc[0, 1]:+.2f}")
    comm_bp = (df["full_comm"][pick_days] / df["full_exposure"][pick_days]).mean() * 1e4
    print(f"\n  commission alone: {comm_bp:.1f} bp of traded notional per round trip at ${budget:,.0f}")
    yr = net(df, "slots", 2.0).groupby(df.index.year).apply(lambda r: r.sum())
    print("\n  calendar-year return, slots, net 2bp/side:")
    print("  " + "  ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "daily.csv")
    res.to_csv(OUT / "summary.csv", index=False)
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
