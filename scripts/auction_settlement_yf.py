"""Settlement validation on deep history: does CMT + static DV01s price the real fly?

The IB path (`auction_settlement_validation.py`) prices each event on the exact contract the
sleeve held, which is the right way -- but IB retains only ~1 YEAR of expired futures
(measured 2026-09-10: ZT/ZF to 20250930, ZN/TN/ZB to 20251219), and a fly needs all three
legs. yfinance carries the same contracts with deep history (TN from 2010, rest from 2005).

WHAT IS BEING COMPARED, and this is the part an earlier version got wrong. The comparison must
hold the POSITION fixed and vary only the pricing source:

    model   = -SUM_i  lots_i x DV01_i x d(CMT yield at leg i)     <- what the backtest assumes
    actual  =  SUM_i  lots_i x d(futures price_i) x point_value_i  <- what the market did

Any gap is then attributable to the thing under test: cheapest-to-deliver drift, DV01s that
are static estimates, basis and carry.

⚠ DO NOT use the auction lab's `fly` series as the model. It is a conventional 2:1:1 YIELD
butterfly on whatever WINGS says (5y -> 3s5s7s), whereas the sleeve trades a DV01-weighted
SLOPE-NEUTRAL fly on 2s5s7s -- ZT not the 3y, because the 3y future is illiquid. Scoring the
live position against the lab fly compares a belly DV01 of 360 with wings 180/180 against the
real 180 with wings 76/65, on partly different curve points. It produces a correlation near
zero and looks like the model failing when it is the comparison that is wrong.

ROLLS. yfinance's Treasury =F series turn out to be BACK-ADJUSTED, not spliced front-month:
the largest daily moves are all real events (Lehman, QE1 2009-03-18, taper 2013-06-20, COVID,
the 2016 election, the 2022-11-10 CPI print) with no quarterly pattern, and a calendar roll
mask made the masked window LESS volatile than the rest (ratio 0.81-0.92). So a calendar
exclusion is unnecessary here and merely discards events. What IS kept is a data-driven
anomaly guard: any window containing a move beyond `OUTLIER_SD` times trailing volatility is
dropped, which catches genuine data errors (e.g. ZB 2015-03-23, +15.44 points in a day).

Usage:  python scripts/auction_settlement_yf.py --since 2015-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from auction_settlement_validation import event_windows, fly_lots  # noqa: E402
from treasury_auction_lab import fetch_yields  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "treasury-auction"))
from treasury_auction.contracts import CONTRACTS  # noqa: E402

YF = {"ZT": "ZT=F", "ZF": "ZF=F", "ZN": "ZN=F", "TN": "TN=F", "ZB": "ZB=F"}
# Each leg's effective curve point -> the nearest FRED constant-maturity series. ZN is a ~7y
# instrument, not a 10y (its CTD sits at the short end of the basket below a 6% coupon), and
# ZB's basket centres nearer 20y than 30y. Both are the sleeve's own documented view.
LEG_CMT = {"ZT": 2, "ZF": 5, "ZN": 7, "TN": 10, "ZB": 20}
OUTLIER_SD = 6.0
OUT = ROOT / "results" / "auction_settlement"


def load_prices(start: str) -> pd.DataFrame:
    raw = yf.download(list(YF.values()), start=start, end="2026-09-11",
                      auto_adjust=False, progress=False, group_by="column")["Close"]
    raw = raw.rename(columns={v: k for k, v in YF.items()}).sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw


def anomaly_days(px: pd.DataFrame) -> set:
    """Days with a move beyond OUTLIER_SD x trailing vol in any leg — data errors or splices."""
    bad: set = set()
    for sym in px.columns:
        d = px[sym].diff()
        z = (d.abs() / d.rolling(60).std()).dropna()
        hits = z[z > OUTLIER_SD]
        if len(hits):
            print(f"    {sym}: {len(hits)} anomalous day(s): "
                  + ", ".join(f"{i.date()} ({d[i]:+.2f}, {z[i]:.1f}sd)" for i in hits.index[:5]))
        bad |= set(hits.index)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2015-01-01")
    args = ap.parse_args()

    ev = event_windows(args.since)
    if ev.empty:
        print("no events"); return
    px = load_prices("2004-01-01")
    ybp = fetch_yields() * 100.0          # CMT in basis points

    print("=" * 92)
    print("STEP 1 — data-driven anomaly guard (yfinance Treasury =F are back-adjusted,")
    print("         so there are no quarterly splices to exclude; only real errors)")
    print("=" * 92)
    bad = anomaly_days(px)
    print(f"    total anomalous sessions flagged: {len(bad)}")

    rows, drop_bad, drop_missing = [], 0, 0
    for _, r in ev.iterrows():
        lots = fly_lots(int(r["tenor"]))
        e0, e1 = r["entry"], r["exit"]
        win = px.loc[(px.index >= e0) & (px.index <= e1)]
        if any(d in bad for d in win.index):
            drop_bad += 1
            continue
        model = actual = 0.0
        ok = True
        for sym, n in lots.items():
            cmt = LEG_CMT[sym]
            if sym not in px.columns or cmt not in ybp.columns:
                ok = False; break
            try:
                p0, p1 = px.at[e0, sym], px.at[e1, sym]
                y0, y1 = ybp.at[e0, cmt], ybp.at[e1, cmt]
            except KeyError:
                ok = False; break
            if not all(np.isfinite(v) for v in (p0, p1, y0, y1)):
                ok = False; break
            # price P&L, and the SAME position priced off CMT with the static DV01s
            actual += n * (p1 - p0) * CONTRACTS[sym].point_value
            model += -n * CONTRACTS[sym].dv01 * (y1 - y0)
        if ok:
            rows.append({"tenor": int(r["tenor"]), "auctionDate": r["auctionDate"],
                         "entry": e0, "exit": e1,
                         "model_usd": model, "actual_usd": actual})
        else:
            drop_missing += 1

    res = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("STEP 2 — SAME POSITION, TWO PRICING SOURCES: CMT+static DV01 vs real futures")
    print("=" * 92)
    print(f"  events in window          {len(ev)}")
    print(f"  dropped, anomalous price  {drop_bad}")
    print(f"  dropped, missing data     {drop_missing}")
    print(f"  PRICED                    {len(res)}")
    if res.empty:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "settlement_validation_yf.csv", index=False)

    for tenor, g in list(res.groupby("tenor")) + [("ALL", res)]:
        if len(g) < 5:
            continue
        slope, intercept = np.polyfit(g["model_usd"], g["actual_usd"], 1)
        corr = float(g["model_usd"].corr(g["actual_usd"]))
        resid = g["actual_usd"] - (slope * g["model_usd"] + intercept)
        se = float(np.sqrt(resid.var(ddof=2)
                           / ((g["model_usd"] - g["model_usd"].mean()) ** 2).sum()))
        lots = fly_lots(tenor) if tenor != "ALL" else None
        print(f"\n  --- {tenor} ---  n={len(g)}" + (f"   lots {lots}" if lots else ""))
        print(f"    model  mean ${g['model_usd'].mean():>+9,.0f}  sd ${g['model_usd'].std():>8,.0f}")
        print(f"    actual mean ${g['actual_usd'].mean():>+9,.0f}  sd ${g['actual_usd'].std():>8,.0f}")
        print(f"    correlation {corr:>+.3f}")
        print(f"    slope {slope:>+.3f} (+/-{1.96 * se:.3f})   1.00 = CMT+static DV01 is right")
        print(f"    mean |tracking error| ${(g['actual_usd'] - g['model_usd']).abs().mean():>7,.0f}")
        print(f"    sign agreement {float((np.sign(g['actual_usd']) == np.sign(g['model_usd'])).mean()):.0%}")

    print("\n  READ: high correlation with slope ~1 means CMT plus the static DV01 table")
    print("  prices the real fly, so the backtest's edge carries over to the futures. A slope")
    print("  below 1 means the futures deliver LESS than the model per unit of curve move —")
    print("  the edge would be overstated by that factor and sizing is too big by 1/slope.")
    print(f"\nWrote {OUT / 'settlement_validation_yf.csv'}")


if __name__ == "__main__":
    main()
