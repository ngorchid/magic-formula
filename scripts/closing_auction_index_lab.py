"""Does forced flow at the close reverse overnight? The premise of the closing-auction trade (H.13).

H.13 would absorb the published closing-auction imbalance and unwind at the next open, earning the
part of the price push that reverses. Imbalance history is not free, so this tests the PREMISE on
the cleanest forced flow there is: an S&P 500 ADDITION. Index funds must own the stock from the
effective date, so they buy it in the closing auction of the trading day before, in size, on a
date announced days in advance. If even that push does not reverse overnight, recording the
imbalance feed for a year is not worth starting.

EVENTS. Wikipedia's "Historical components of the S&P 500" changes table (effective date, ticker
added, ticker removed). Event close d = the last trading day before the effective date.
  additions   index funds BUY at the close of d     -> a provider SELLS into it, buys back later
  deletions   index funds SELL at the close of d    -> a provider BUYS, sells back later
              (only deletions whose stock still trades the next day; a stock moved to the S&P 400
              is bought by mid-cap funds in the same auction, so the net flow is smaller)

MEASURED, per event, SPY-hedged (the stock's return minus SPY's over the same window), with
prices adjusted for splits and dividends:
  push        open of d -> close of d          (the event-day move; whichever way the net
                                                pressure ran, not only the auction itself)
  provider    close of d -> open of d+1        (the trade H.13 describes; sign: positive = the
                                                provider made money, i.e. the push reversed)
  provider+1  close of d -> close of d+1       (holding to the next close instead)
yfinance's close is the official close (the auction price) and its open is close to the opening
auction price; neither is guaranteed to the cent.

COST. Both legs are auctions, so no spread is crossed: IB commission on about $5k per order
(~1.4bp a round trip) plus the SPY hedge, taken as 3bp a round trip in total.

SPIN-OFFS AND NEW LISTINGS ARE EXCLUDED (added after the first run, which included them): a
spun-off company joins the index because index funds RECEIVE its shares in the distribution, so
there is no forced buy at the close to absorb, and its first prices mix when-issued and regular
trading (the first run's worst "provider" losses were Howmet -29%, Carrier, PayPal, GE HealthCare,
SanDisk). An event also needs 20 trading days of prices before it. Both runs are reported.

PRIMARY TEST (fixed before the first run): additions, provider close -> next open, hedged, net of
3bp. The premise holds if the mean is positive with t >= 2 over all events AND positive in 2020+.
Deletions and the next-close variant are descriptive.

SECOND READING (added after the first runs, so descriptive, not a test). The primary test assumes
additions are a BUY imbalance; on the event day the added stock in fact FALLS against SPY
(arbitrageurs sell to the index funds). So the script also asks whether the event-day move,
whichever way it ran, reverses overnight -- against a CONTROL of the same stocks on ordinary days
in the year before (excluding the last week) -- and what "trade against the day's move at the
close, exit at the open" earns on event days. That rule was chosen after seeing the data.

Usage:  python scripts/closing_auction_index_lab.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.loaders import download_ohlcv  # noqa: E402

WIKI = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
START, END = "2009-01-01", "2026-10-17"
COST_BP = 3.0
OUT = ROOT / "results" / "closing_auction_index"
PERIODS = {"2011-2015": ("2011", "2015"), "2016-2019": ("2016", "2019"), "2020+": ("2020", "2026")}


def changes() -> pd.DataFrame:
    r = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 algo_trading/0.1"}, timeout=60)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    t.columns = ["date", "add_t", "add_s", "rem_t", "rem_s", "reason", "refs"]
    t["date"] = pd.to_datetime(t["date"], errors="coerce", format="mixed")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "sp500_changes.csv", index=False)
    return t.dropna(subset=["date"])


def norm(t) -> str | None:
    return str(t).strip().upper().replace(".", "-") if isinstance(t, str) and t.strip() else None


def measure(ev: pd.DataFrame, px: dict[str, pd.DataFrame], side: float) -> pd.DataFrame:
    """side = -1 for additions (provider sells), +1 for deletions (provider buys)."""
    close_raw = px["close"]
    f = px["adj_close"] / close_raw
    o, c = px["open"] * f, px["adj_close"]
    cal = c.index
    rows = []
    for d_eff, tk in ev[["date", "ticker"]].itertuples(index=False):
        if tk not in c.columns:
            continue
        i = cal.searchsorted(d_eff) - 1                  # last trading day before the effective date
        if i < 20 or i + 1 >= len(cal) or not np.isfinite(c.iat[i - 20, c.columns.get_loc(tk)]):
            continue                                     # needs 20 trading days of history
        d, n = cal[i], cal[i + 1]
        vals = [o.at[d, tk], c.at[d, tk], o.at[n, tk], c.at[n, tk],
                o.at[d, "SPY"], c.at[d, "SPY"], o.at[n, "SPY"], c.at[n, "SPY"]]
        if not np.all(np.isfinite(vals)) or close_raw.at[d, tk] < 5:
            continue
        so, sc, no, nc, mo, mc, mno, mnc = vals
        rows.append({
            "event_close": d, "ticker": tk,
            "push": (sc / so - 1) - (mc / mo - 1),
            "provider": side * ((no / sc - 1) - (mno / mc - 1)),
            "provider_1": side * ((nc / sc - 1) - (mnc / mc - 1)),
        })
    return pd.DataFrame(rows)


def line(x: pd.Series) -> str:
    if len(x) < 3:
        return f"n {len(x):3d}"
    t = x.mean() / x.std() * np.sqrt(len(x))
    return f"n {len(x):3d}  mean {x.mean() * 1e4:+7.1f}bp  median {x.median() * 1e4:+7.1f}bp  t {t:+5.2f}  win {(x > 0).mean():.0%}"


def main() -> None:
    ch = changes()
    ch = ch[ch["date"] >= "2010-01-01"]
    spin = ch["reason"].fillna("").str.contains(r"spun|spin|split off|separat", case=False, regex=True)
    print(f"spin-off additions excluded: {int(spin.sum())}")
    adds = pd.DataFrame({"date": ch["date"], "ticker": ch["add_t"].map(norm)})[~spin].dropna()
    dels = pd.DataFrame({"date": ch["date"], "ticker": ch["rem_t"].map(norm)}).dropna()
    print(f"changes since 2010: {len(ch)} rows, {len(adds)} additions, {len(dels)} deletions; "
          f"by year: {ch.groupby(ch['date'].dt.year).size().to_dict()}")
    tickers = sorted(set(adds["ticker"]) | set(dels["ticker"]) | {"SPY"})
    panel = download_ohlcv(tickers, START, END)
    px = {k: panel[k].sort_index() for k in ("open", "close", "adj_close")}
    have = px["close"].notna().sum()
    print(f"prices for {(have > 0).sum()} of {len(tickers)} tickers (renamed or delisted names have none)")

    res = {}
    for name, ev, side in (("ADDITIONS (provider sells at the close)", adds, -1.0),
                           ("DELETIONS still trading (provider buys)", dels, +1.0)):
        m = measure(ev, px, side)
        res[name] = m
        m.to_csv(OUT / f"{name.split()[0].lower()}.csv", index=False)
        print(f"\n=== {name}: {len(m)} events with prices ===")
        for col, lab in (("push", "event-day move (open d -> close d), hedged"),
                         ("provider", "PROVIDER close d -> open d+1, hedged, gross"),
                         ("provider_1", "provider close d -> close d+1, hedged, gross")):
            print(f"  {lab}")
            print(f"    {'all':10} {line(m[col])}")
            for p, (a, b) in PERIODS.items():
                x = m[(m.event_close >= a) & (m.event_close <= f'{b}-12-31')][col]
                print(f"    {p:10} {line(x)}")
        net = m["provider"] - COST_BP / 1e4
        print(f"  provider close -> open NET of {COST_BP:g}bp: {line(net)}")

    a = res["ADDITIONS (provider sells at the close)"]
    net = a["provider"] - COST_BP / 1e4
    t = net.mean() / net.std() * np.sqrt(len(net))
    recent = net[a.event_close >= "2020-01-01"]
    ok = net.mean() > 0 and t >= 2 and recent.mean() > 0
    print(f"\nPRIMARY (additions, provider close -> next open, hedged, net {COST_BP:g}bp): "
          f"mean {net.mean() * 1e4:+.1f}bp, t {t:+.2f}, n {len(net)}; 2020+ {recent.mean() * 1e4:+.1f}bp "
          f"(n {len(recent)})  ->  premise {'HOLDS' if ok else 'NOT SHOWN'}")
    second_reading(res, px)
    print(f"\nWrote {OUT}/")


def second_reading(res: dict, px: dict) -> None:
    f = px["adj_close"] / px["close"]
    o, c = px["open"] * f, px["adj_close"]
    day = (c / o - 1).sub(c["SPY"] / o["SPY"] - 1, axis=0)
    night = (o.shift(-1) / c - 1).sub(o["SPY"].shift(-1) / c["SPY"] - 1, axis=0)
    a = res["ADDITIONS (provider sells at the close)"]
    ev = pd.DataFrame({"day": a["push"], "night": -a["provider"]})     # unsigned overnight move
    ctl = []
    for tk, d in a[["ticker", "event_close"]].itertuples(index=False):
        i, j = c.index.get_loc(d), c.columns.get_loc(tk)
        for k in range(max(0, i - 250), i - 5):
            x, y = day.iat[k, j], night.iat[k, j]
            if np.isfinite(x) and np.isfinite(y):
                ctl.append((x, y))
    ctl = pd.DataFrame(ctl, columns=["day", "night"])
    rng = np.random.default_rng(0)
    boot = [np.polyfit(*ev.sample(len(ev), replace=True, random_state=int(rng.integers(1e9)))
                       [["day", "night"]].to_numpy().T, 1)[0] for _ in range(2000)]
    print("\nSECOND READING (descriptive): does the event-day move reverse overnight?")
    print(f"  additions: corr {ev.day.corr(ev.night):+.2f}, slope {np.polyfit(ev.day, ev.night, 1)[0]:+.3f} "
          f"(95% interval {np.quantile(boot, .025):+.3f} to {np.quantile(boot, .975):+.3f}), n {len(ev)}")
    print(f"  control, same stocks on ordinary days: corr {ctl.day.corr(ctl.night):+.2f}, "
          f"slope {np.polyfit(ctl.day, ctl.night, 1)[0]:+.3f}, n {len(ctl)}")
    allev = pd.concat([res["ADDITIONS (provider sells at the close)"].assign(side=-1.0),
                       res["DELETIONS still trading (provider buys)"].assign(side=1.0)])
    pnl = -np.sign(allev["push"]) * allev["provider"] * allev["side"] - COST_BP / 1e4
    print(f"  against the day's move, close -> open, net {COST_BP:g}bp (rule chosen after the data): "
          f"{line(pnl)}")
    big = allev["push"].abs() >= allev["push"].abs().median()
    print(f"    day move above median size: {line(pnl[big])}")


if __name__ == "__main__":
    main()
