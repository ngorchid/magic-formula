"""Can the options-VRP sleeve's premium income pay for convex tail protection?

THE QUESTION (user, 2026-08-10). The book's real weak point is a FAST equity crash: magic-formula
is long beta, options-vrp is short puts, and the trend overlay is far too slow to help — measured
in `trend-as-beta-hedge`, its signal arrives after 61-91% of the fall and is still short through
the rebound. The instrument that closes that gap is CONVEXITY (long OTM puts), which pays on
impact. And because we are already SELLING volatility, the natural structure is a BARBELL: fund
the tail hedge out of the VRP premium rather than treating them as two independent bets.

WHAT THIS MEASURES. Both legs are priced on the SAME real OPRA SPX chain (2013-2026,
`results/spx_vrp/chain.parquet`, built by convert_opra.py + spx_chain.py), so the comparison is
like-for-like and needs no option model:

  HEDGE   buy an OTM put at a target delta, DTE in [dte_lo, dte_hi], roll every `roll_days`
          trading days. Exit at the roll using the SAME contract's real close; if it has expired
          by then, settle at intrinsic. Cost and payoff are expressed as a fraction of the NOTIONAL
          the put covers (spot x 100), which makes the result scale-free and directly comparable
          to a book's equity exposure.
  INCOME  the live-spec VRP sleeve (16d/10d, VRP>2, no stop, regime gate on) from spx_vrp_lab,
          converted to the same units.

WHY DELTA AND NOT "% OTM": a fixed 15%-OTM strike buys wildly different amounts of protection
depending on the vol regime — in calm markets it is many sigma away and nearly worthless, in a
panic it is close and expensive. Targeting delta holds the PROBABILITY roughly fixed, which is
what "tail insurance" actually means.

The honest question is not "does the hedge make money" — bought insurance loses money on average
and should. It is whether the VRP premium COVERS the drag, and whether the payoff arrives in the
episodes where the rest of the book is bleeding.

Run: python scripts/tail_hedge_lab.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUT = ROOT / "results" / "spx_vrp"

# Fast-crash windows: where a lagging trend signal cannot help and the hedge has to earn its keep.
CRISES = [("2015 Aug", "2015-08-01", "2015-09-30"),
          ("2018 Feb (Volmageddon)", "2018-01-26", "2018-02-28"),
          ("2018 Q4", "2018-10-01", "2018-12-31"),
          ("2020 COVID", "2020-02-19", "2020-04-30"),
          ("2022 bear", "2022-01-01", "2022-10-31"),
          ("2025-26 (recent)", "2025-01-01", "2026-08-06")]


@dataclass
class Hedge:
    delta: float = 0.05          # |delta| of the put bought
    dte_lo: int = 30
    dte_hi: int = 60
    roll_days: int = 21          # trading days between rolls
    cost_pts: float = 0.50       # half-spread paid per side, index points (same as the VRP lab)


def _pick(day: pd.DataFrame, h: Hedge):
    """The OTM put whose |delta| is nearest the target, within the DTE window."""
    c = day[(day.cp == "P") & (day.dte.between(h.dte_lo, h.dte_hi)) &
            (day.delta.notna()) & (day.close > 0)]
    if c.empty:
        return None
    c = c.assign(err=(c.delta.abs() - h.delta).abs())
    return c.nsmallest(1, "err").iloc[0]


def run_hedge(ch: pd.DataFrame, h: Hedge) -> pd.DataFrame:
    by_date = {d: g for d, g in ch.groupby("date")}
    lut = {d: dict(zip(g.contract, g.close)) for d, g in ch.groupby("date")}
    dates = np.array(sorted(by_date))
    rows = []
    i = 0
    while i < len(dates) - 1:
        d = dates[i]
        leg = _pick(by_date[d], h)
        if leg is None:
            i += 1
            continue
        j = min(i + h.roll_days, len(dates) - 1)
        d2 = dates[j]
        entry = float(leg.close) + h.cost_pts          # pay the offer
        spot0 = float(leg.spot)
        px = lut.get(d2, {}).get(leg.contract)
        if px is None:                                  # expired inside the holding window
            s2 = float(by_date[d2].spot.iloc[0])
            exitp = max(float(leg.strike) - s2, 0.0)    # settle intrinsic
            how = "expired"
        else:
            exitp = max(float(px) - h.cost_pts, 0.0)    # sell at the bid
            how = "rolled"
        rows.append({"entry_date": d, "exit_date": d2, "strike": float(leg.strike),
                     "expiry": leg.expiry, "dte": int(leg.dte), "delta": float(leg.delta),
                     "spot": spot0, "entry": entry, "exit": exitp, "how": how,
                     "notional": spot0 * 100.0,
                     "pnl_pts": exitp - entry, "pnl_usd": (exitp - entry) * 100.0,
                     "cost_usd": entry * 100.0})
        i = j
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame, label: str) -> dict:
    yrs = (t.exit_date.max() - t.entry_date.min()).days / 365.25
    # Everything as a fraction of the notional each put covers, so it scales to any book size.
    spend = (t.cost_usd / t.notional).sum() / yrs
    net = (t.pnl_usd / t.notional).sum() / yrs
    return {"label": label, "n": len(t), "per_yr": len(t) / yrs,
            "avg_dte": t.dte.mean(), "avg_otm": (t.strike / t.spot - 1).mean(),
            "spend_yr": spend, "net_yr": net, "hit%": (t.pnl_usd > 0).mean(),
            "best": (t.pnl_usd / t.notional).max(), "worst": (t.pnl_usd / t.notional).min()}


def crisis_payoff(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nm, a, b in CRISES:
        w = t[(t.entry_date >= a) & (t.entry_date <= b)]
        if w.empty:
            rows.append({"episode": nm, "n": 0, "spend": np.nan, "payoff": np.nan})
            continue
        rows.append({"episode": nm, "n": len(w),
                     "spend": (w.cost_usd / w.notional).sum(),
                     "payoff": (w.pnl_usd / w.notional).sum()})
    return pd.DataFrame(rows)


def main() -> None:
    ch = pd.read_parquet(OUT / "chain.parquet")
    ch = ch[ch.iv.notna()]
    print(f"chain {len(ch):,} rows  {ch.date.min().date()} -> {ch.date.max().date()}")

    print("\n" + "=" * 104)
    print("TAIL HEDGE — rolling long OTM SPX puts, priced on real OPRA closes")
    print("=" * 104)
    print(f"  {'variant':22s} {'n':>4s} {'/yr':>5s} {'avg DTE':>8s} {'avg OTM':>8s} | "
          f"{'SPEND/yr':>9s} {'NET/yr':>9s} {'hit%':>6s} {'best':>8s} {'worst':>7s}")
    print("  " + "-" * 100)
    res = {}
    for d in (0.10, 0.05, 0.02):
        for roll in (21,):
            h = Hedge(delta=d, roll_days=roll)
            t = run_hedge(ch, h)
            s = summarise(t, f"{d:.0%}-delta put, {roll}d roll")
            res[d] = t
            print(f"  {s['label']:22s} {s['n']:>4d} {s['per_yr']:>5.1f} {s['avg_dte']:>8.0f} "
                  f"{s['avg_otm']:>8.1%} | {s['spend_yr']:>8.2%} {s['net_yr']:>+8.2%} "
                  f"{s['hit%']:>6.0%} {s['best']:>+7.2%} {s['worst']:>+6.2%}")
    print("\n  SPEND/yr and NET/yr are fractions of the NOTIONAL each put covers (spot x 100).")
    print("  NET is after payoffs — bought insurance is SUPPOSED to be negative on average.")

    print("\n" + "=" * 104)
    print("WHERE THE PAYOFF LANDS — spend vs payoff per episode (fraction of notional)")
    print("=" * 104)
    print(f"  {'episode':24s} " + "".join(f"{f'{d:.0%}d spend':>12s}{f'{d:.0%}d payoff':>13s}"
                                          for d in res))
    print("  " + "-" * 100)
    tabs = {d: crisis_payoff(t).set_index("episode") for d, t in res.items()}
    for nm, _, _ in CRISES:
        line = f"  {nm:24s} "
        for d in res:
            r = tabs[d].loc[nm]
            line += (f"{r['spend']:>11.2%} {r['payoff']:>+12.2%}" if r["n"]
                     else f"{'—':>11s} {'—':>12s}")
        print(line)

    OUT.mkdir(parents=True, exist_ok=True)
    for d, t in res.items():
        t.to_csv(OUT / f"tail_hedge_{int(d*100):02d}delta.csv", index=False)
    print(f"\n  wrote {OUT}/tail_hedge_*.csv")


if __name__ == "__main__":
    main()
