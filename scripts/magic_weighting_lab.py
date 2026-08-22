"""Equal weight vs rank/Kelly-tilted weight for the magic-formula book.

THE QUESTION. The book holds 30 names at 1/30 each. The combined rank is monotonically
predictive (IC +0.0629, t=+1.9), so the #1 name should in principle be worth more than the #30.
Should capital follow the rank?

THE THEORY CUTS BOTH WAYS.
  FOR   Kelly / mean-variance says w ~ inv(Sigma) mu. With similar vols that is w ~ mu, and via
        Grinold's E[r] = IC x sigma x z, mu is proportional to the rank z-score. So a rank tilt
        IS the Kelly direction.
  AGAINST  it is w ~ inv(Sigma) mu that makes this dangerous: it amplifies error in the input you
        know LEAST well. DeMiguel, Garlappi & Uppal (2009) found 1/N beats mean-variance out of
        sample on most datasets for exactly this reason. And the selection has already done the
        work -- picking 30 from ~500 exploits the rank across its full range, while the spread
        WITHIN the surviving 30 is narrow.

So this is an empirical question, not a theoretical one. Schemes tested, all on the same
selection (top 30, band 45) so ONLY the weighting differs:

  equal           1/30                                    (live)
  inverse-vol     1/sigma, the existing alternative       (already supported)
  rank-linear     linear in rank position, best gets most
  rank-z          w ~ z-score of rank among the held      (Grinold, sigma assumed equal)
  kelly-diag      w ~ z / sigma^2, diagonal covariance     (the actual Kelly form)
  half-kelly      50/50 blend of kelly-diag and equal      (the standard prudence discount)

⚠ CONCENTRATION IS THE REAL RISK, not return. A tilt that helps on average can put 8% of the
book in one name, so max weight is reported alongside Sharpe. A +0.03 Sharpe bought with double
the single-name concentration is not obviously a good trade.

Run: python3 scripts/magic_weighting_lab.py
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

from backtest import summary_stats                          # noqa: E402
from strategies.magic_formula import (EnhancedMagicConfig,   # noqa: E402
                                      enhanced_rank)
from strategies.magic_formula.construct import (_rebal_dates,  # noqa: E402
                                                pnl)
from run_best_magic import _load                              # noqa: E402

SPLIT = "2019-07-01"
CAP = 0.10          # no single name above this, whatever the scheme says


def build(rank, adj, vol, top_n, hold_n, scheme):
    """weights_banded, but the weighting of the HELD set is pluggable. Selection is identical
    across schemes, so any difference is attributable to sizing alone."""
    cal = adj.index
    target = pd.DataFrame(np.nan, index=cal, columns=adj.columns)
    held: list[str] = []
    for dt in _rebal_dates(cal, "ME"):
        row = rank.loc[dt].dropna()
        if len(row) < top_n:
            continue
        pos = pd.Series(range(len(row)), index=row.sort_values(ascending=False).index)
        keep = [t for t in held if t in pos.index and pos[t] < hold_n]
        need = top_n - len(keep)
        if need > 0:
            held = keep + [t for t in pos.sort_values().index if t not in keep][:need]
        else:
            held = sorted(keep, key=lambda t: pos[t])[:top_n]

        p = pos[held].astype(float)                       # 0 = best
        sd = vol.loc[dt, held] if vol is not None else pd.Series(1.0, index=held)
        sd = sd.where(sd > 0).fillna(sd.median() if sd.notna().any() else 1.0)
        if scheme == "equal":
            raw = pd.Series(1.0, index=held)
        elif scheme == "inverse_vol":
            raw = 1.0 / sd
        elif scheme == "rank_linear":
            raw = pd.Series(float(len(held)), index=held) - p.rank(method="first") + 1.0
        else:
            # z-score of rank POSITION, sign-flipped so the best name is the most positive.
            z = -(p - p.mean()) / (p.std() if p.std() else 1.0)
            if scheme == "rank_z":
                raw = z - z.min() + 0.25          # shift positive; keeps ordering, avoids shorts
            elif scheme == "kelly_diag":
                raw = (z - z.min() + 0.25) / (sd ** 2)
            elif scheme == "half_kelly":
                k = (z - z.min() + 0.25) / (sd ** 2)
                raw = 0.5 * k / k.sum() + 0.5 * (1.0 / len(held))
            else:
                raise ValueError(scheme)
        w = raw / raw.sum()
        w = w.clip(upper=CAP)
        w = w / w.sum()
        out = pd.Series(0.0, index=adj.columns)
        out.loc[held] = w.values
        target.loc[dt] = out.values
    return target.ffill().fillna(0.0).shift(1).fillna(0.0)


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load(
        "sp500_pit", cfg, "2012-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    rank = enhanced_rank(f, mcap, adj, base, cfg).where(base)
    vol = adj.pct_change(fill_method=None).rolling(cfg.vol_window).std()

    rows = []
    for scheme in ("equal", "inverse_vol", "rank_linear", "rank_z", "kelly_diag", "half_kelly"):
        w = build(rank, adj, vol, cfg.top_n, cfg.hold_n, scheme)
        net, turn = pnl(w, adj, volume, close)
        idx = net.replace(0.0, np.nan).dropna().index
        net = net.reindex(idx).fillna(0.0)
        d = {"scheme": scheme, "turnover": turn,
             "max_wt": float(w.max(axis=1).mean()),
             "eff_n": float((1.0 / (w ** 2).sum(axis=1)).replace(np.inf, np.nan).mean())}
        for k, sl in (("FULL", slice(None)), ("IS", slice(None, SPLIT)), ("OOS", slice(SPLIT, None))):
            s = summary_stats(net.loc[sl])
            d[f"sh_{k}"], d[f"ret_{k}"], d[f"dd_{k}"] = s["sharpe"], s["ann_return"], s["max_drawdown"]
        rows.append(d)
        print(f"  {scheme} done")

    df = pd.DataFrame(rows)
    b0 = df.iloc[0]
    print("\n" + "=" * 100)
    print(f"WEIGHTING SCHEMES — {label}, same selection (top 30 / band 45), cap {CAP:.0%}/name")
    print("=" * 100)
    print(f"  {'scheme':14}{'max wt':>8}{'eff N':>7}{'turn':>7}{'ann ret':>10}{'Sharpe':>9}"
          f"{'dSh':>7}{'maxDD':>9}   {'Sh IS':>7}{'Sh OOS':>8}")
    print("  " + "-" * 96)
    for _, x in df.iterrows():
        print(f"  {x['scheme']:14}{x['max_wt']:>8.1%}{x['eff_n']:>7.1f}{x['turnover']:>6.1f}x"
              f"{x['ret_FULL']:>+10.2%}{x['sh_FULL']:>+9.2f}{x['sh_FULL']-b0['sh_FULL']:>+7.2f}"
              f"{x['dd_FULL']:>+9.2%}   {x['sh_IS']:>+7.2f}{x['sh_OOS']:>+8.2f}")
    print("\n  'eff N' = 1/sum(w^2), the effective number of positions. Equal weight gives 30;")
    print("   lower means the book is concentrated in fewer names than it appears to hold.")
    print("\n  ⚠ Sharpe SE ~0.27 on 13.5 years. A tilt must beat equal weight in BOTH halves AND")
    print("     justify whatever concentration it adds.")
    df.to_csv(ROOT / "results" / "magic_weighting.csv", index=False)


if __name__ == "__main__":
    main()
