"""Should component CHANGES (dEY, dROC) become factors? And is the bulk of the book just beta?

FOUR QUESTIONS, in the order that decides whether the idea is worth building.

1. SINCE-ENTRY classification. The exit tests classified drops by the trailing-252d change in
   each component's rank, but names are often held only a few months, so that window includes
   time BEFORE purchase. This redoes it measuring the change SINCE ENTRY, which is what the
   hypothesis actually says.

2. IS dEY JUST MOMENTUM? EY = FCF/EV and EV contains market cap, so "EY rank fell over the past
   year" is largely "the price rose over the past year" -- which is residual momentum, ALREADY
   one of the three families in the live rank. If they correlate strongly the factor is
   double-counting, and it would explain why the conditional-hold rule added nothing.
   Same question for dROC against the growth family (fcf_growth is itself a change measure).

3. UNIVERSE-WIDE IC. The exit finding was conditional on having dropped out of the band --
   a selected subsample. A factor has to work across the WHOLE eligible universe to earn a place
   in the ranking. Measured monotonically (Spearman) AND by quintile, because the exit pattern
   was NON-MONOTONE ("exactly one fell" good, both/neither bad) and a Spearman IC reads ~zero on
   an XOR shape even when it is real.

4. IS THE BULK JUST BETA? The hypothesis is that the 80% of names that leave by relative
   displacement contribute market exposure rather than alpha. Tested directly by regressing the
   book on SPY and reading the alpha.

⚠ A LINEAR IC-WEIGHTED BLEND CANNOT EXPRESS AN XOR. combine_ranks averages percentile ranks and
assumes monotonicity. If Q3 (no change) is worst and both tails are good, no weight on a rank of
dEY produces that. Q4 below is therefore the test of whether the proposed MECHANISM even fits the
proposed VEHICLE.

Run: python3 scripts/magic_delta_factor_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from backtest import summary_stats                               # noqa: E402
from signals.quality import fcf_ev_yield, fcf_return_on_capital   # noqa: E402
from signals.momentum import residual_momentum                    # noqa: E402
from strategies.magic_formula import (EnhancedMagicConfig,        # noqa: E402
                                      enhanced_weights)
from run_best_magic import _load                                   # noqa: E402

FWD, LOOK, THR = 252, 252, -0.10


def ic(sig, fwd, elig):
    out = {}
    for d in sig.index[::21]:                       # monthly, keeps overlap manageable
        a, b = sig.loc[d].where(elig.loc[d]), fwd.loc[d].where(elig.loc[d])
        ok = a.notna() & b.notna()
        if ok.sum() >= 30:
            out[d] = a[ok].corr(b[ok], method="spearman")
    s = pd.Series(out).dropna()
    yr = s.groupby(s.index.year).mean()
    return s.mean(), (yr.mean() / yr.std() * np.sqrt(len(yr)) if len(yr) > 2 else np.nan)


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    w, rank = enhanced_weights(f, mcap, adj, base, cfg)
    ey = fcf_ev_yield(f, mcap).reindex_like(adj).rank(axis=1, pct=True)
    roc = fcf_return_on_capital(f).reindex_like(adj).rank(axis=1, pct=True)
    d_ey, d_roc = ey - ey.shift(LOOK), roc - roc.shift(LOOK)
    fwd = adj.shift(-FWD) / adj - 1.0

    # ---------- 1. SINCE-ENTRY classification of drops ----------
    heldm = w > 0
    fwd_book = pd.Series({d: fwd.loc[d][heldm.loc[d]].mean() for d in w.index})
    # SINGLE PASS. An object-dtype DataFrame of entry dates did not survive .loc assignment and
    # silently produced zero records; tracking the entry date in a plain dict is both simpler and
    # unambiguous.
    recs, entry_of = [], {}
    prev = set()
    for d in w.index:
        now = set(w.columns[heldm.loc[d].values])
        for t in now - prev:
            entry_of[t] = d                       # newly opened
        for t in prev - now:                      # dropped today
            e = entry_of.pop(t, None)
            if e is None:
                continue
            de, dr = ey.at[d, t] - ey.at[e, t], roc.at[d, t] - roc.at[e, t]
            v, b = fwd.at[d, t], fwd_book.get(d, np.nan)
            if np.isfinite(de) and np.isfinite(dr) and np.isfinite(v) and np.isfinite(b):
                recs.append({"date": d, "d_ey": de, "d_roc": dr, "ex": v - b,
                             "hold_m": (d - e).days / 30.4})
        prev = now
    ev = pd.DataFrame(recs)
    ev["bucket"] = np.where((ev.d_roc <= THR) & (ev.d_ey > THR), "ROC only",
                   np.where((ev.d_ey <= THR) & (ev.d_roc > THR), "EY only",
                   np.where((ev.d_ey <= THR) & (ev.d_roc <= THR), "both", "neither")))
    print("\n" + "=" * 88)
    print(f"1. SINCE-ENTRY classification (median hold {ev.hold_m.median():.1f} months)")
    print("=" * 88)
    print(f"  {'bucket':12}{'n':>6}{'ex252 vs book':>16}{'vs other drops':>17}{'t annual':>11}")
    for b in ["EY only", "ROC only", "both", "neither"]:
        g, o = ev[ev.bucket == b], ev[ev.bucket != b]
        if len(g) < 3:
            continue
        ga = g.groupby(g.date.dt.year).ex.mean().dropna()
        oa = o.groupby(o.date.dt.year).ex.mean().dropna()
        j = ga.index.intersection(oa.index); dd = ga[j] - oa[j]
        t = dd.mean() / dd.std() * np.sqrt(len(dd)) if len(dd) > 2 else np.nan
        print(f"  {b:12}{len(g):>6}{g.ex.mean():>+16.2%}{g.ex.mean()-o.ex.mean():>+17.2%}{t:>+11.1f}")

    # ---------- 2. overlap with factors already in the model ----------
    mom = residual_momentum(adj, lookback=cfg.momentum_lookback,
                            skip=cfg.momentum_skip).reindex_like(adj).rank(axis=1, pct=True)
    print("\n" + "=" * 88)
    print("2. IS IT ALREADY IN THE MODEL? cross-sectional corr with the live factors")
    print("=" * 88)
    for nm, a in (("d_EY", d_ey), ("d_ROC", d_roc)):
        c = a.corrwith(mom, axis=1).mean()
        print(f"  corr({nm}, residual momentum) = {c:+.3f}"
              + ("   <- largely the SAME SIGNAL, adding it double-counts" if abs(c) > 0.4 else ""))

    # ---------- 3. universe-wide IC, monotone and by quintile ----------
    print("\n" + "=" * 88)
    print(f"3. UNIVERSE-WIDE IC vs {FWD}d forward return (whole eligible universe, not just drops)")
    print("=" * 88)
    for nm, a in (("d_EY", d_ey), ("d_ROC", d_roc), ("|d_EY| (XOR proxy)", d_ey.abs())):
        m, t = ic(a, fwd, base)
        print(f"  {nm:22} IC {m:>+8.4f}   t(annual) {t:>+5.1f}")
    print("\n  by quintile of d_EY (mean forward return, eligible universe):")
    q = d_ey.rank(axis=1, pct=True)
    for i in range(5):
        m = (q > i / 5) & (q <= (i + 1) / 5) & base
        print(f"    Q{i+1} {fwd.where(m).mean().mean():>+8.2%}", end="")
    print("\n  ⚠ a LINEAR IC-weighted blend can only exploit a MONOTONE pattern across these.")

    # ---------- 4. beta vs alpha ----------
    net = (w * adj.pct_change(fill_method=None).fillna(0.0)).sum(axis=1)
    idx = net.replace(0.0, np.nan).dropna().index
    n, s_ = net.reindex(idx).fillna(0.0), spy.reindex(idx).fillna(0.0)
    beta = n.cov(s_) / s_.var()
    alpha_d = n.mean() - beta * s_.mean()
    resid = n - beta * s_
    print("\n" + "=" * 88)
    print("4. IS THE BOOK JUST BETA?  regression of the strategy on SPY")
    print("=" * 88)
    print(f"  beta {beta:.2f}   alpha {alpha_d*252:+.2%}/yr   "
          f"t(alpha) {alpha_d/resid.std()*np.sqrt(len(n)):+.1f}")
    print(f"  R^2 {n.corr(s_)**2:.2f}   -> {n.corr(s_)**2:.0%} of daily variance is market")
    print(f"  strategy {summary_stats(n)['ann_return']:+.2%}/yr vs "
          f"SPY {summary_stats(s_)['ann_return']:+.2%}/yr")


if __name__ == "__main__":
    main()
