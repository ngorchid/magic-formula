"""Lab: does credit quality predict returns AMONG magic-formula selections?

Follow-up to magic_downgrade_lab.py, which established (Fisher p=0.0006, OR 2.14) that
the strategy over-selects names that are eventually downgraded, but could not test the
consequence: the trailing-12m downgrade screen touched only 2.63% of weight across 11
events, far short of the ~64 needed.

The fix is to stop screening on a rare EVENT and screen on the prevailing rating LEVEL,
which is point-in-time observable on every rated holding on every date. That lifts the
exposure from 2.63% to 5.97% of weight for sub-IG, and gives a continuous predictor
across 78,757 rated position-days instead of 11 events.

=======================  PRE-SPECIFICATION (written before running)  =======================

H1 (PRIMARY, tradeable). Among magic-formula selections, PIT credit rating notch is
   POSITIVELY related to forward return: lower-rated holdings underperform.
   Both relevant literatures predict this sign — the value-trap mechanism (Dichev &
   Piotroski) and the distress anomaly (Campbell/Hilscher/Szilagyi 2008, distressed
   stocks earn LOWER returns). The competing view is that within a value strategy the
   low-rated names ARE the deep-value names carrying the value premium, which would
   give the opposite sign. Genuinely two-sided, hence worth testing.

   Test: ONE pooled regression. Monthly, non-overlapping. Forward 1-month return,
   cross-sectionally demeaned within each month across held+rated names (removes market
   and time effects entirely), regressed on demeaned notch. SE clustered by date.
   Significance at |t| > 2. No transformations, no interactions, no threshold search.

H2 (SECONDARY, pre-specified screen). Excluding sub-IG holdings (notch < 12) improves
   risk-adjusted return. ONE threshold only, chosen for economic reasons before seeing
   any result: BBB-/Baa3 is the index-eligibility and forced-seller boundary, and the
   point where Moody's switches rating label. Paired difference test vs rated baseline.

CONFOUND. Low-rated names are systematically more levered, more cyclical, smaller and
   cheaper, and the strategy already tilts to value. H1 is therefore reported both raw
   and with log-mcap plus sector controls. If the coefficient dies under controls, the
   signal is a size/sector proxy, not credit information.

EXPLORATORY (reported, but NOT evidence — other thresholds and the forward-looking
   diagnostic, which uses future information and is not tradeable).

Pre-declared stopping rule: if H1 fails under controls, H2 is reported for completeness
and the lead is closed. No widening of thresholds to find significance.
===========================================================================================

Usage:
    python scripts/magic_credit_quality_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import summary_stats
from data import sp500_sectors
from data.rating_history import entity_ratings, rating_panel, rated_tickers
from strategies.magic_formula import EnhancedMagicConfig, enhanced_weights
from strategies.magic_formula.construct import pnl
from run_best_magic import _load

FITCH_PARQUET = ROOT / "data" / "cache" / "fitch_rating_actions.parquet"
START, END = "2012-01-01", "2026-08-23"
IG_FLOOR = 12


def build_panel_data(w: pd.DataFrame, panel: pd.DataFrame, adj: pd.DataFrame,
                     mcap: pd.DataFrame) -> pd.DataFrame:
    """One row per (month, held+rated name): forward 1m return, notch, controls.

    Monthly and non-overlapping, so observations do not share return windows and a
    date-clustered SE is the only correction needed.
    """
    month_ends = adj.resample("ME").last().index
    month_ends = [d for d in month_ends if d in adj.index]
    fwd = adj.reindex(month_ends).pct_change(fill_method=None).shift(-1)

    held = (w.abs() > 1e-12).reindex(month_ends).fillna(False)
    notch = panel.reindex(month_ends)
    lmcap = np.log(mcap.reindex(month_ends).where(lambda x: x > 0))

    rows = []
    for d in month_ends:
        names = held.columns[held.loc[d] & notch.loc[d].notna() & fwd.loc[d].notna()]
        if len(names) < 5:
            continue
        rows.append(pd.DataFrame({
            "date": d, "ticker": names,
            "fwd": fwd.loc[d, names].values,
            "notch": notch.loc[d, names].values,
            "lmcap": lmcap.loc[d, names].values,
        }))
    df = pd.concat(rows, ignore_index=True)
    # Cross-sectional demeaning within month removes market + time effects.
    for col in ("fwd", "notch", "lmcap"):
        df[col + "_dm"] = df[col] - df.groupby("date")[col].transform("mean")
    return df


def regress(df: pd.DataFrame, xs: list[str], label: str) -> tuple[float, float]:
    d = df.dropna(subset=["fwd_dm"] + xs)
    y, X = d["fwd_dm"], sm.add_constant(d[xs])
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["date"]})
    b, t = m.params["notch_dm"], m.tvalues["notch_dm"]
    print(f"  {label:44s} beta={b*100:+7.3f}%/notch/mo  t={t:+6.2f}  n={len(d):5d}")
    return b, t


def paired(net: pd.Series, base: pd.Series, idx, label: str) -> None:
    d = (net.reindex(idx).fillna(0.0) - base.reindex(idx).fillna(0.0)).dropna()
    ann, vol = d.mean() * 252, d.std() * np.sqrt(252)
    t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
    print(f"  {label:44s} ann {ann:+7.2%}  vol {vol:6.2%}  t={t:+6.2f}"
          f"   {'SIGNIFICANT' if abs(t) > 2 else 'not distinguishable from zero'}")


def main() -> None:
    R = pd.read_parquet(FITCH_PARQUET)
    cfg = EnhancedMagicConfig()
    print("[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load("sp500_pit", cfg, START, END)
    uni = sorted(adj.columns)
    rated = [t for t in rated_tickers(R, uni) if t in adj.columns]
    mask = pd.Series(adj.columns.isin(rated), index=adj.columns)
    elig_rated = base & mask

    w, _ = enhanced_weights(f, mcap, adj, elig_rated, cfg)
    panel = rating_panel(entity_ratings(R), adj.index, list(adj.columns)).reindex(
        index=adj.index, columns=adj.columns)

    d = build_panel_data(w, panel, adj, mcap)
    print(f"\n  panel: {len(d)} (month, name) observations over "
          f"{d['date'].nunique()} months, {d['ticker'].nunique()} names")
    print(f"  mean notch {d['notch'].mean():.2f}, sub-IG share {(d['notch'] < IG_FLOOR).mean():.1%}")

    print("\n" + "=" * 84)
    print("H1 (PRIMARY) — does PIT rating notch predict forward return among holdings?")
    print("  positive beta = lower-rated holdings underperform (predicted direction)")
    print("=" * 84)
    b_raw, t_raw = regress(d, ["notch_dm"], "raw")
    b_ctl, t_ctl = regress(d, ["notch_dm", "lmcap_dm"], "+ log market cap control")

    sec = sp500_sectors().reindex(d["ticker"]).values
    d2 = d.assign(sector=sec).dropna(subset=["sector"])
    dummies = pd.get_dummies(d2["sector"], prefix="s", drop_first=True, dtype=float)
    d2 = pd.concat([d2.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    b_sec, t_sec = regress(d2, ["notch_dm", "lmcap_dm"] + list(dummies.columns),
                           "+ log mcap + sector dummies")

    h1_survives = abs(t_ctl) > 2 and abs(t_sec) > 2 and np.sign(b_ctl) == np.sign(b_sec)
    print(f"\n  H1 verdict: {'SUPPORTED' if h1_survives else 'NOT SUPPORTED'}"
          f"  (needs |t|>2 under BOTH control sets, consistent sign)")

    print("\n" + "=" * 84)
    print("H2 (SECONDARY, pre-specified) — exclude sub-IG holdings (notch < 12)")
    print("=" * 84)
    net_base, turn_base = pnl(w, adj, volume, close)
    idx = net_base.replace(0.0, np.nan).dropna().index
    s = summary_stats(net_base.reindex(idx).fillna(0.0))
    print(f"  {'baseline (rated universe)':44s} ann {s['ann_return']:+7.2%}  "
          f"sharpe {s['sharpe']:+5.2f}  maxDD {s['max_drawdown']:+7.2%}")

    results = {}
    for thresh, tag, kind in [(12, "exclude sub-IG (notch<12)  [PRE-SPECIFIED]", "pre"),
                              (13, "exclude BBB- and below      [exploratory]", "exp"),
                              (14, "exclude BBB and below       [exploratory]", "exp")]:
        bad = panel.lt(thresh).reindex(index=adj.index, columns=adj.columns).fillna(False)
        w2, _ = enhanced_weights(f, mcap, adj, elig_rated & ~bad, cfg)
        net2, turn2 = pnl(w2, adj, volume, close)
        s2 = summary_stats(net2.reindex(idx).fillna(0.0))
        print(f"  {tag:44s} ann {s2['ann_return']:+7.2%}  "
              f"sharpe {s2['sharpe']:+5.2f}  maxDD {s2['max_drawdown']:+7.2%}")
        results[tag] = (net2, kind)

    print("\n  paired difference vs baseline:")
    for tag, (net2, kind) in results.items():
        paired(net2, net_base, idx, tag)

    print("\n" + "=" * 84)
    print("EXPLORATORY — forward-looking diagnostic (uses FUTURE info, NOT tradeable)")
    print("=" * 84)
    from data.rating_history import downgrade_events, map_to_tickers
    ev = map_to_tickers(downgrade_events(entity_ratings(R)), uni).dropna(subset=["ticker"])
    ev = ev[["ticker", "action_date"]].drop_duplicates()
    fut = pd.DataFrame(False, index=adj.index, columns=adj.columns)
    for tk, dt in ev.itertuples(index=False):
        if tk in fut.columns:
            fut.loc[(adj.index > dt - pd.DateOffset(months=12)) & (adj.index <= dt), tk] = True
    d3 = d.assign(pre_dg=[bool(fut.loc[r.date, r.ticker]) for r in d.itertuples()])
    g = d3.groupby("pre_dg")["fwd_dm"].agg(["mean", "std", "count"])
    print(f"  demeaned fwd 1m return, NOT within 12m before a downgrade: "
          f"{g.loc[False,'mean']*100:+.3f}%  (n={int(g.loc[False,'count'])})")
    if True in g.index:
        print(f"  demeaned fwd 1m return, WITHIN 12m before a downgrade:     "
              f"{g.loc[True,'mean']*100:+.3f}%  (n={int(g.loc[True,'count'])})")
        from scipy.stats import ttest_ind
        a = d3.loc[d3.pre_dg, "fwd_dm"].dropna()
        b = d3.loc[~d3.pre_dg, "fwd_dm"].dropna()
        t, p = ttest_ind(a, b, equal_var=False)
        print(f"  Welch t = {t:+.2f}, p = {p:.4f}   difference "
              f"{(a.mean()-b.mean())*100:+.2f}%/month (~{((1+a.mean()-b.mean())**12-1)*100:+.0f}%/yr)")
        print("\n  READ THIS AS MECHANISM, NOT SIGNAL: the window is defined by a FUTURE")
        print("  downgrade, so it cannot be traded. What it shows is WHERE the return")
        print("  damage sits — before the agency acts, not after. That is why both the")
        print("  post-downgrade screen and the rating-level screen find nothing: by")
        print("  publication the market has already repriced. Ratings lag prices.")

    out = ROOT / "results" / "credit_quality"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "panel.csv", index=False)
    print(f"\n  wrote {out}/panel.csv")


if __name__ == "__main__":
    main()
