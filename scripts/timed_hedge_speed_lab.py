"""Lab: can a FAST-reacting timed beta hedge beat not hedging at all?

The static hedge is now priced and rejected: ~7.5%/yr of return, a Sharpe gain that is
not convertible at IB margin rates, and no differential circuit-breaker penalty. A TIMED
hedge is the one version that could still work, because it only pays the carrying cost
while it is on.

The prior attempt (run_tactical_hedge.py, SPY 20/50/200-day MAs) and the trend-overlay
signal (63/126/252-day lookbacks) were both judged "61-91% too late — shorts the
rebound". The open question, never tested: was that a property of TIMED HEDGING, or just
of those LOOKBACKS? This runs a much faster ladder plus two trigger families that are not
moving averages at all.

    A  price vs N-day MA        N = 3, 5, 10, 20, 50, 100, 200
    B  drawdown from N-day high  breach X% below a 60-day high
    C  volatility spike          10d realised vol > k x 60d realised vol
    D  combined MA + vol         both conditions, to cut whipsaw

All signals are EX-ANTE: computed on data through t-1 and applied to the return at t.
Toggle costs are charged on every switch, which is what punishes the fast variants.

Each trigger is also run at partial intensity (hedge to h=0.5 when on, not h=1.0), since
the static lab showed intensity and timing are separable knobs.

Benchmarks: never hedge (the incumbent, and the one to beat), static h=0.50, static h=1.

Run: python scripts/timed_hedge_speed_lab.py
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

CAP = 300_000
TOGGLE_BPS = 3.0     # cost of switching the hedge on or off
WINDOWS = [("2015-16 selloff", "2015-08-01", "2016-02-29"),
           ("2018 Q4", "2018-10-01", "2018-12-31"),
           ("2020 COVID", "2020-02-15", "2020-03-31"),
           ("2022 bear", "2022-01-01", "2022-12-31")]


def load() -> tuple[pd.Series, pd.Series, pd.Series]:
    bh = pd.read_csv(ROOT / "results" / "beta_hedge" / "beta_hedge.csv",
                     index_col=0, parse_dates=True).dropna()
    px = yf.download("SPY", start="2010-01-01", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return bh["long_book"], bh["beta_hedged"], px


def signals(px: pd.Series, idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Ex-ante hedge-on booleans. Everything is shifted so date t uses data through t-1."""
    out: dict[str, pd.Series] = {}
    for n in (3, 5, 10, 20, 50, 100, 200):
        out[f"A: below {n}d MA"] = px < px.rolling(n).mean()
    for x in (0.03, 0.05, 0.08):
        dd = px / px.rolling(60).max() - 1.0
        out[f"B: >{x:.0%} below 60d high"] = dd <= -x
    r = px.pct_change()
    v10, v60 = r.rolling(10).std(), r.rolling(60).std()
    for k in (1.3, 1.6, 2.0):
        out[f"C: vol10 > {k:.1f}x vol60"] = v10 > k * v60
    out["D: below 50d MA AND vol10>1.3x"] = (px < px.rolling(50).mean()) & (v10 > 1.3 * v60)
    out["D: below 20d MA AND vol10>1.6x"] = (px < px.rolling(20).mean()) & (v10 > 1.6 * v60)
    return {k: v.shift(1).reindex(idx).fillna(False).astype(bool) for k, v in out.items()}


def apply_hedge(long: pd.Series, hedged: pd.Series, on: pd.Series,
                intensity: float) -> pd.Series:
    """Blend, charging a toggle cost whenever the hedge weight changes."""
    w = on.astype(float) * intensity
    toggles = w.diff().abs().fillna(0.0)
    return (1 - w) * long + w * hedged - toggles * TOGGLE_BPS / 1e4


def row(name: str, r: pd.Series, on: pd.Series | None = None) -> dict:
    s = summary_stats(r.fillna(0.0))
    fv = CAP * (1 + r.fillna(0.0)).prod()
    d = {"name": name, "cagr": s["ann_return"], "vol": s["ann_vol"],
         "sharpe": s["sharpe"], "maxdd": s["max_drawdown"], "final": fv,
         "on%": float(on.mean()) if on is not None else 0.0,
         "toggles": int(on.astype(float).diff().abs().sum()) if on is not None else 0}
    for lbl, a, b in WINDOWS:
        d[lbl] = (1 + r.loc[a:b].fillna(0.0)).prod() - 1
    return d


def show(rows: list[dict], title: str) -> None:
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)
    hdr = (f"  {'variant':32s} {'CAGR':>8s} {'vol':>7s} {'Sh':>6s} {'maxDD':>7s} "
           f"{'final $':>11s} {'on%':>6s} {'tog':>5s} " + "".join(f"{l[:9]:>10s}" for l, _, _ in WINDOWS))
    print(hdr)
    for d in rows:
        print(f"  {d['name']:32s} {d['cagr']:>+8.2%} {d['vol']:>7.2%} {d['sharpe']:>+6.2f} "
              f"{d['maxdd']:>+7.1%} {d['final']:>11,.0f} {d['on%']:>6.0%} {d['toggles']:>5d} "
              + "".join(f"{d[l]:>+10.1%}" for l, _, _ in WINDOWS))


def main() -> None:
    long, hedged, px = load()
    idx = long.index
    sigs = signals(px, idx)

    base = [row("NEVER hedge (incumbent)", long),
            row("static h=0.50", 0.5 * long + 0.5 * hedged),
            row("static h=1.00", hedged)]
    show(base, "BENCHMARKS")

    for intensity in (1.0, 0.5):
        rows = [row(n, apply_hedge(long, hedged, on, intensity), on)
                for n, on in sigs.items()]
        rows.sort(key=lambda d: -d["final"])
        show(rows, f"TIMED HEDGE — intensity {intensity:.2f} (hedge to h={intensity:.2f} when on), "
                   f"sorted by terminal wealth")

    best_never = base[0]["final"]
    print("\n" + "=" * 132)
    print("VERDICT")
    print("=" * 132)
    winners = []
    for intensity in (1.0, 0.5):
        for n, on in sigs.items():
            d = row(n, apply_hedge(long, hedged, on, intensity), on)
            if d["final"] > best_never:
                winners.append((d["final"], f"{n} @ h={intensity:.2f}", d))
    if winners:
        print(f"  {len(winners)} variant(s) beat NEVER hedging on terminal wealth:")
        for fv, lbl, d in sorted(winners, reverse=True):
            print(f"    {lbl:44s} {fv:>11,.0f} vs {best_never:,.0f}  "
                  f"(+{fv/best_never-1:.1%})  maxDD {d['maxdd']:+.1%} vs {base[0]['maxdd']:+.1%}")
        print("\n  CAUTION: this is a sweep over 13 signals x 2 intensities = 26 tests.")
        print("  Treat any winner as a HYPOTHESIS, not a result, until it survives a")
        print("  holdout split — the VRP work already showed parameter search overfits here.")
    else:
        print(f"  NONE of the {len(sigs)*2} timed variants beat simply never hedging "
              f"({best_never:,.0f}).")
        print("  Timing is not the missing ingredient; the carrying cost of being short")
        print("  beta in a rising market dominates whatever the trigger avoids.")

    robustness(long, hedged, px)

    out = ROOT / "results" / "timed_hedge"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row(n, apply_hedge(long, hedged, on, 1.0), on)
                  for n, on in sigs.items()]).to_csv(out / "speed_ladder.csv", index=False)
    print(f"\n  wrote {out}/speed_ladder.csv")


def robustness(long: pd.Series, hedged: pd.Series, px: pd.Series) -> None:
    """Try to break the vol-spike winner: parameter surface, holdout, and COVID removal.

    A single winning cell out of a 26-test sweep is a hypothesis, not a result. What
    would make it credible is a BROAD parameter surface (neighbours also win), survival
    in both halves of the sample, and survival with the largest episode removed.
    """
    idx = long.index
    r = px.pct_change()
    base = CAP * (1 + long.fillna(0)).prod()

    def sig(fast: int, slow: int, k: float) -> pd.Series:
        return ((r.rolling(fast).std() > k * r.rolling(slow).std())
                .shift(1).reindex(idx).fillna(False).astype(bool))

    print("\n" + "=" * 132)
    print("ROBUSTNESS OF THE VOL-SPIKE WINNER")
    print("=" * 132)
    print(f"  parameter surface at k=2.0 (never-hedge = ${base:,.0f}):")
    fires = beats = 0
    for fast in (5, 10, 15, 20):
        for slow in (40, 60, 90, 120):
            if fast >= slow:
                continue
            on = sig(fast, slow, 2.0)
            eps = int(on.astype(float).diff().clip(lower=0).sum())
            if eps == 0:
                continue
            fires += 1
            f = CAP * (1 + apply_hedge(long, hedged, on, 1.0).fillna(0)).prod()
            beats += f > base
    print(f"    {beats} of {fires} FIRING window pairs beat never-hedging "
          f"(5 further pairs never trigger at all)")

    on = sig(10, 60, 2.0)
    h = apply_hedge(long, hedged, on, 1.0)
    for lbl, a, b in [("first half 2012-2019", "2012-01-01", "2019-12-31"),
                      ("second half 2020-2026", "2020-01-01", "2026-12-31")]:
        sl, sh = summary_stats(long.loc[a:b].fillna(0)), summary_stats(h.loc[a:b].fillna(0))
        eps = int(on.loc[a:b].astype(float).diff().clip(lower=0).sum())
        print(f"  {lbl:22s} {eps:2d} episodes  never {sl['ann_return']:>+7.2%}/Sh {sl['sharpe']:>+5.2f}"
              f"  hedged {sh['ann_return']:>+7.2%}/Sh {sh['sharpe']:>+5.2f}"
              f"  delta {sh['ann_return']-sl['ann_return']:>+6.2%}")

    m = ~((idx >= "2020-02-01") & (idx <= "2020-05-31"))
    s0, s1 = summary_stats(long[m].fillna(0)), summary_stats(h[m].fillna(0))
    print(f"  ex-COVID               never {s0['ann_return']:>+7.2%}/Sh {s0['sharpe']:>+5.2f}"
          f"  hedged {s1['ann_return']:>+7.2%}/Sh {s1['sharpe']:>+5.2f}"
          f"  delta {s1['ann_return']-s0['ann_return']:>+6.2%}")

    print("\n  KNOWN BLIND SPOT: the signal fires on FAST vol expansions only. The 2022 bear")
    print("  was a slow grind and the k=2.0 signal was on 0% of it. This would not have")
    print("  helped in a 2000-2002 style decline. Test pre-2012 SPY before believing it.")


if __name__ == "__main__":
    main()
