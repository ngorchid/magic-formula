"""Is d_ROC a NEW factor, or is fcf_growth already carrying it?

d_ROC (change in the return-on-capital percentile rank) was the one factor signal to survive its
own robustness check: universe-wide IC +0.0180, t=+1.7, and monotone, so unlike the XOR pattern it
could actually enter a linear rank blend.

But the model already contains `fcf_growth`, and ROC = FCF / capital -- so d_ROC is roughly
"FCF growth minus capital growth". The two could be close to the same signal, in which case
adding it is double-counting rather than new information.

THE DECISIVE TEST IS NOT THE PAIRWISE CORRELATION. A factor earns a place by adding IC to what is
ALREADY THERE, so d_ROC is residualised cross-sectionally against:
  (a) fcf_growth alone -- the specific overlap suspected;
  (b) the FULL existing combined rank -- the real incumbent, since the live model is all three
      families together and a new factor competes with the whole thing, not with one component.

If the residual IC survives (b), the factor is genuinely additive and the backtest below shows
what it is worth. If it collapses, the signal was already priced into the ranking.

⚠ THIS IS THE LAST CANDIDATE FROM A LONG SEARCH. Every other rule and parameter tested in this
session came back flat or negative, so a marginal positive here should be read against how many
things were looked at, not on its own. The IS/OOS split is the check that matters.

Run: python3 scripts/magic_droc_lab.py
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
from signals import fcf_growth, revenue_growth                    # noqa: E402
from signals.quality import (fcf_ev_yield,                        # noqa: E402
                             fcf_return_on_capital)
from signals.momentum import residual_momentum                    # noqa: E402
from strategies.magic_formula import EnhancedMagicConfig          # noqa: E402
from strategies.magic_formula.construct import (combine_ranks,    # noqa: E402
                                                pnl, weights_banded)
from run_best_magic import _load                                   # noqa: E402

FWD, LOOK, SPLIT = 252, 252, "2019-07-01"


def ic_series(sig, fwd, elig):
    """Monthly cross-sectional Spearman IC."""
    out = {}
    for d in sig.index[::21]:
        a, b = sig.loc[d].where(elig.loc[d]), fwd.loc[d].where(elig.loc[d])
        ok = a.notna() & b.notna()
        if ok.sum() >= 30:
            out[d] = a[ok].corr(b[ok], method="spearman")
    return pd.Series(out).dropna()


def report(name, s):
    yr = s.groupby(s.index.year).mean()
    t = yr.mean() / yr.std() * np.sqrt(len(yr)) if len(yr) > 2 else np.nan
    print(f"  {name:44} IC {s.mean():>+8.4f}   t(annual) {t:>+5.1f}")
    return s.mean(), t


def residualise(a, b):
    """Cross-sectional residual of `a` after regressing on `b`, per day (both rank-transformed)."""
    out = pd.DataFrame(np.nan, index=a.index, columns=a.columns)
    for d in a.index[::21]:
        x, y = b.loc[d], a.loc[d]
        ok = x.notna() & y.notna()
        if ok.sum() < 30:
            continue
        xv, yv = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        beta = (xv * yv).sum() / (xv ** 2).sum() if (xv ** 2).sum() else 0.0
        out.loc[d, ok[ok].index] = (yv - beta * xv).values
    return out.ffill()


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    fwd = adj.shift(-FWD) / adj - 1.0
    rk = lambda x: x.reindex_like(adj).rank(axis=1, pct=True)   # noqa: E731

    roc = rk(fcf_return_on_capital(f))
    d_roc = roc - roc.shift(LOOK)
    fcfg = rk(fcf_growth(f))
    live_rank = combine_ranks([[fcf_ev_yield(f, mcap), fcf_return_on_capital(f)],
                               [revenue_growth(f), fcf_growth(f)],
                               [residual_momentum(adj, lookback=cfg.momentum_lookback,
                                                  skip=cfg.momentum_skip)]], base)

    print("\n" + "=" * 92)
    print("1. PAIRWISE OVERLAP (necessary but not sufficient)")
    print("=" * 92)
    c1 = d_roc.corrwith(fcfg, axis=1).mean()
    c2 = d_roc.corrwith(rk(live_rank), axis=1).mean()
    print(f"  corr(d_ROC, fcf_growth)        = {c1:+.3f}")
    print(f"  corr(d_ROC, LIVE combined rank) = {c2:+.3f}")

    print("\n" + "=" * 92)
    print(f"2. IC vs {FWD}d forward return — standalone, then residualised")
    print("=" * 92)
    report("d_ROC standalone", ic_series(d_roc, fwd, base))
    report("fcf_growth standalone", ic_series(fcfg, fwd, base))
    report("LIVE combined rank standalone", ic_series(rk(live_rank), fwd, base))
    print()
    m_a, t_a = report("d_ROC residualised on fcf_growth", ic_series(residualise(d_roc, fcfg), fwd, base))
    m_b, t_b = report("d_ROC residualised on the LIVE RANK  <- decisive",
                      ic_series(residualise(d_roc, rk(live_rank)), fwd, base))

    print("\n" + "=" * 92)
    print("3. PORTFOLIO TEST — add d_ROC as a 4th family (equal weight with the other three)")
    print("=" * 92)
    rows = []
    for lab, fams in (
            ("BASELINE (live, 3 families)", None),
            ("+ d_ROC as a 4th family", "family"),
            ("+ d_ROC inside the growth family", "growth")):
        if fams is None:
            r = live_rank
        elif fams == "family":
            r = combine_ranks([[fcf_ev_yield(f, mcap), fcf_return_on_capital(f)],
                               [revenue_growth(f), fcf_growth(f)],
                               [residual_momentum(adj, lookback=cfg.momentum_lookback,
                                                  skip=cfg.momentum_skip)],
                               [d_roc]], base)
        else:
            r = combine_ranks([[fcf_ev_yield(f, mcap), fcf_return_on_capital(f)],
                               [revenue_growth(f), fcf_growth(f), d_roc],
                               [residual_momentum(adj, lookback=cfg.momentum_lookback,
                                                  skip=cfg.momentum_skip)]], base)
        w = weights_banded(r.where(base), adj, cfg.rebalance, cfg.top_n, cfg.hold_n)
        net, turn = pnl(w, adj, volume, close)
        idx = net.replace(0.0, np.nan).dropna().index
        net = net.reindex(idx).fillna(0.0)
        d = {"variant": lab, "turnover": turn}
        for k, sl in (("FULL", slice(None)), ("IS", slice(None, SPLIT)), ("OOS", slice(SPLIT, None))):
            st = summary_stats(net.loc[sl])
            d[f"sh_{k}"], d[f"ret_{k}"] = st["sharpe"], st["ann_return"]
        rows.append(d)
    df = pd.DataFrame(rows)
    b0 = df.iloc[0]
    print(f"  {'variant':34}{'turn':>7}{'ann ret':>10}{'Sharpe':>9}{'dSh':>7}"
          f"   {'Sh IS':>7}{'Sh OOS':>8}")
    print("  " + "-" * 88)
    for _, x in df.iterrows():
        print(f"  {x['variant']:34}{x['turnover']:>6.1f}x{x['ret_FULL']:>+10.2%}"
              f"{x['sh_FULL']:>+9.2f}{x['sh_FULL']-b0['sh_FULL']:>+7.2f}"
              f"   {x['sh_IS']:>+7.2f}{x['sh_OOS']:>+8.2f}")
    print("\n  ⚠ Sharpe SE ~0.27 on 13.5 years. A dSh inside that is noise, and after a search")
    print("     this long a marginal positive should be assumed fitted until it wins BOTH halves.")
    df.to_csv(ROOT / "results" / "magic_droc.csv", index=False)


if __name__ == "__main__":
    main()
