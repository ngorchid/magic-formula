"""Conditional hold: keep a dropped name longer ONLY when the reason for the drop says to.

WHERE THIS COMES FROM. The band sweep showed that widening the band UNCONDITIONALLY does not pay
-- but widening it holds everything longer, including the 80% of drops that are mere relative
displacement (a name stood still while better names arrived) and underperform by -4.1%/yr.

The exit decomposition suggested a conditional version. Splitting drops by which value component
moved -- fcf_ev_yield CONTAINS price, fcf_return_on_capital does NOT -- gives a 2x2 where the two
"exactly one component fell" cells outperform the other drops:

    EY fell only (price rose, business intact)   +10.4pp vs other drops, t=+2.1 annual
    ROC fell only (already punished / investing)  +12.4pp, t=+1.3
    both fell (deteriorating AND still dear)       -8.0pp
    neither fell (just displaced by better names)  -7.1pp

⚠ THAT IS EVENT-LEVEL EXCESS RETURN, NOT PORTFOLIO RETURN. It says a name did well after being
sold; it does not account for position weight, timing, or the fact that holding it means NOT
holding something else. Only a portfolio backtest settles that, which is this file.

THE REPRIEVED NAME OCCUPIES A SLOT. It is not held "extra" -- it competes directly with the
replacement it displaces, which is the whole cost of the rule and the reason the event table
cannot stand in for this test.

⚠ SELECTION. "EY only" was pre-specified by the user before seeing any of this data and deserves
the credit that carries. "Exactly one" is a post-hoc reading of the 2x2 by me and deserves less.
Both are reported IS/OOS; a rule that wins only in one half is noise. `reprieve=ALL` is the
control -- it should reproduce a wider band and therefore NOT help.

Run: python3 scripts/magic_conditional_hold_lab.py
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

from backtest import summary_stats                              # noqa: E402
from signals.quality import fcf_ev_yield, fcf_return_on_capital  # noqa: E402
from strategies.magic_formula import (EnhancedMagicConfig,       # noqa: E402
                                      enhanced_rank)
from strategies.magic_formula.construct import _rebal_dates, pnl  # noqa: E402
from run_best_magic import _load                                  # noqa: E402

THR, SPLIT, LOOK = -0.10, "2019-07-01", 252


def weights_conditional(rank, adj, top_n, hold_n, d_ey, d_roc, mode, reprieve_m):
    """`weights_banded` plus a reprieve. mode: none | ey | one | all.

    A name falling out of `hold_n` is normally sold. With a reprieve it is kept for up to
    `reprieve_m` further rebalances IF the drop reason qualifies -- but it still occupies one of
    the `top_n` slots, so it is competing with the name that would have replaced it.
    """
    cal = adj.index
    target = pd.DataFrame(np.nan, index=cal, columns=adj.columns)
    held: list[str] = []
    left: dict[str, int] = {}                 # ticker -> reprieves remaining
    for dt in _rebal_dates(cal, rebalance="ME"):
        row = rank.loc[dt].dropna()
        if len(row) < top_n:
            continue
        pos = pd.Series(range(len(row)), index=row.sort_values(ascending=False).index)
        keep = []
        for t in held:
            if t not in pos.index:
                left.pop(t, None)
                continue
            if pos[t] < hold_n:               # still in the band: normal hold, reprieve resets
                keep.append(t)
                left.pop(t, None)
                continue
            if mode == "none":
                left.pop(t, None)
                continue
            if t in left:                     # already on a reprieve -- run the clock down
                if left[t] > 0:
                    left[t] -= 1
                    keep.append(t)
                else:
                    left.pop(t, None)
                continue
            de = d_ey.at[dt, t] if t in d_ey.columns else np.nan
            dr = d_roc.at[dt, t] if t in d_roc.columns else np.nan
            if not (np.isfinite(de) and np.isfinite(dr)):
                continue
            ey_fell, roc_fell = de <= THR, dr <= THR
            grant = (mode == "all"
                     or (mode == "ey" and ey_fell and not roc_fell)
                     or (mode == "one" and (ey_fell != roc_fell)))
            if grant and reprieve_m > 0:
                left[t] = reprieve_m - 1
                keep.append(t)
        need = top_n - len(keep)
        if need > 0:
            adds = [t for t in pos.sort_values().index if t not in keep][:need]
            held = keep + adds
        else:
            held = sorted(keep, key=lambda t: pos[t])[:top_n]
            for t in list(left):
                if t not in held:
                    left.pop(t, None)
        w = pd.Series(0.0, index=adj.columns)
        w.loc[held] = 1.0 / len(held)
        target.loc[dt] = w.values
    return target.ffill().fillna(0.0).shift(1).fillna(0.0)


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)               # LIVE config
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    rank = enhanced_rank(f, mcap, adj, base, cfg).where(base)
    ey = fcf_ev_yield(f, mcap).reindex_like(adj).rank(axis=1, pct=True)
    roc = fcf_return_on_capital(f).reindex_like(adj).rank(axis=1, pct=True)
    d_ey, d_roc = ey - ey.shift(LOOK), roc - roc.shift(LOOK)

    variants = [("BASELINE band 45 (live)", "none", 0)]
    for m in (3, 6, 12):
        variants.append((f"reprieve EY-only, {m}m", "ey", m))
    for m in (3, 6, 12):
        variants.append((f"reprieve exactly-one, {m}m", "one", m))
    variants.append(("reprieve ALL 6m (control)", "all", 6))

    rows = []
    for lab, mode, m in variants:
        w = weights_conditional(rank, adj, cfg.top_n, cfg.hold_n, d_ey, d_roc, mode, m)
        net, turn = pnl(w, adj, volume, close)
        idx = net.replace(0.0, np.nan).dropna().index
        net = net.reindex(idx).fillna(0.0)
        r = {"variant": lab, "turnover": turn}
        for k, sl in (("FULL", slice(None)), ("IS", slice(None, SPLIT)), ("OOS", slice(SPLIT, None))):
            s = summary_stats(net.loc[sl])
            r[f"sh_{k}"], r[f"ret_{k}"] = s["sharpe"], s["ann_return"]
        rows.append(r)
        print(f"  {lab} done")

    df = pd.DataFrame(rows)
    base_row = df.iloc[0]
    print("\n" + "=" * 96)
    print(f"CONDITIONAL HOLD — {label}, top 30 / band 45, monthly, use_graham=False")
    print("=" * 96)
    print(f"  {'variant':30}{'turn':>7}{'ann ret':>10}{'Sharpe':>9}{'dSh':>7}"
          f"   {'Sh IS':>7}{'Sh OOS':>8}")
    print("  " + "-" * 92)
    for _, x in df.iterrows():
        d = x["sh_FULL"] - base_row["sh_FULL"]
        print(f"  {x['variant']:30}{x['turnover']:>6.1f}x{x['ret_FULL']:>+10.2%}"
              f"{x['sh_FULL']:>+9.2f}{d:>+7.2f}   {x['sh_IS']:>+7.2f}{x['sh_OOS']:>+8.2f}")
    n_yr = 13.5
    print(f"\n  ⚠ Sharpe SE on {n_yr:.1f} years is ~{1 / np.sqrt(n_yr):.2f}. A dSh inside that is")
    print("     noise, and any rule must beat the baseline in BOTH halves to be worth deploying.")
    best = df.iloc[1:].loc[df.iloc[1:]["sh_FULL"].idxmax()]
    print(f"\n  best rule: {best['variant']} -> {best['sh_FULL']:+.2f} "
          f"({best['sh_FULL'] - base_row['sh_FULL']:+.2f} vs baseline), "
          f"IS {best['sh_IS']:+.2f} vs {base_row['sh_IS']:+.2f}, "
          f"OOS {best['sh_OOS']:+.2f} vs {base_row['sh_OOS']:+.2f}")
    df.to_csv(ROOT / "results" / "magic_conditional_hold.csv", index=False)


if __name__ == "__main__":
    main()
