"""Lab: does excluding recently-downgraded names improve the enhanced Magic Formula?

Motivation. Greenblatt-style screens rank on *trailing* earnings yield, so a company
whose stock has collapsed into a credit downgrade still screens cheap at exactly the
moment its fundamentals are deteriorating — the textbook value trap. Dichev & Piotroski
(1998) found ~1yr of negative abnormal equity returns after a bond downgrade, and the
distress-risk literature (Campbell/Hilscher/Szilagyi 2008) points the same way. So the
pre-specified, economically-motivated test is a NEGATIVE screen: drop any name
downgraded in the trailing N months, and let the strategy take the next-ranked one.

Note this is the opposite direction to the fallen-angel *bond* trade, where forced
selling by IG-only mandates creates a real technical discount. Equity has no such
forced seller, and the two claims on the same impaired firm move oppositely.

Design, and why it is shaped this way:

  * BOTH arms are restricted to the SAME agency-rated universe. Fitch rates ~361 of
    the ~825 names that passed through the S&P 500 since 2011, and the rated ones are
    systematically the more indebted (88% with >=$500m LT debt vs 74% of the rest).
    Comparing a screened-on-361 arm against the full-825 baseline would confound the
    screen with a leverage/sector bet.
  * Fitch is the primary source. Moody's obligor file is dominated by speculative-grade
    CFRs, so for many names its history *begins* at the fall to junk and "no downgrade"
    is missing data rather than an observation. See data/rating_history.py.
  * The overlap check runs FIRST and can stop the study. With ~30 holdings drawn from
    a 361-name pool, the screen may simply never bind often enough to be measurable.
    A statistically powerless result should be reported as such, not fitted to.

No look-ahead: rating actions are public the day they occur, so action_date is used
directly. The 12-month publication delay in the 17g-7 files affects only live trading,
not the historical dates.

Usage:
    python scripts/magic_downgrade_lab.py                  # overlap check + full study
    python scripts/magic_downgrade_lab.py --overlap-only    # just the power check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import summary_stats
from data.rating_history import (downgrade_events, entity_ratings, ig_to_hy_crossings,
                                 map_to_tickers, rated_tickers)
from strategies.magic_formula import EnhancedMagicConfig, enhanced_weights
from strategies.magic_formula.construct import pnl, size_bucket
from run_best_magic import _load

FITCH_PARQUET = ROOT / "data" / "cache" / "fitch_rating_actions.parquet"
START, END = "2012-01-01", None


def load_ratings() -> pd.DataFrame:
    if not FITCH_PARQUET.exists():
        raise SystemExit(
            f"missing {FITCH_PARQUET}\n"
            "Build it first:\n"
            "  python -c \"from data.rating_history import parse_rocr_zip; \\\n"
            "    parse_rocr_zip('data/cache/Fitch_Ratings-<date>.zip')"
            f".to_parquet('{FITCH_PARQUET}')\"")
    return pd.read_parquet(FITCH_PARQUET)


def event_table(ratings: pd.DataFrame, tickers: list[str], fallen_angels: bool) -> pd.DataFrame:
    """Deduplicated (ticker, action_date) downgrade events inside the universe.

    Parent and financing subsidiaries both map to one ticker and act on the same day
    (Ford Motor Company / Ford Motor Credit), so one corporate event counts once.
    """
    er = entity_ratings(ratings)
    ev = ig_to_hy_crossings(er) if fallen_angels else downgrade_events(er)
    ev = map_to_tickers(ev, tickers).dropna(subset=["ticker", "action_date"])
    return ev[["ticker", "action_date"]].drop_duplicates().sort_values("action_date")


def flag_frame(events: pd.DataFrame, calendar: pd.DatetimeIndex,
               tickers: list[str], months: int) -> pd.DataFrame:
    """(date x ticker) bool: downgraded within the trailing `months`."""
    flags = pd.DataFrame(False, index=calendar, columns=sorted(set(tickers)))
    window = pd.DateOffset(months=months)
    for ticker, date in events.itertuples(index=False):
        if ticker in flags.columns:
            flags.loc[(calendar >= date) & (calendar < date + window), ticker] = True
    return flags


def overlap_report(weights: pd.DataFrame, events: pd.DataFrame, months: int) -> dict:
    """Does the screen bind often enough to be measurable? Runs before any backtest."""
    held = weights.abs() > 1e-12
    flags = flag_frame(events, weights.index, list(weights.columns), months).reindex(
        index=weights.index, columns=weights.columns, fill_value=False)
    hit = held & flags

    held_days = int(held.values.sum())
    hit_days = int(hit.values.sum())
    names_held = sorted(weights.columns[held.any()])
    names_hit = sorted(weights.columns[hit.any()])
    # distinct (ticker, event) pairs where the name was held on the action date
    on_date = 0
    for ticker, date in events.itertuples(index=False):
        if ticker in held.columns:
            near = held.index[(held.index >= date - pd.Timedelta(days=5)) &
                              (held.index <= date + pd.Timedelta(days=5))]
            if len(near) and bool(held.loc[near, ticker].any()):
                on_date += 1

    weight_touched = float(weights.where(hit).abs().sum().sum() / weights.abs().sum().sum())
    print("=" * 78)
    print(f"OVERLAP / POWER CHECK   (trailing {months}m downgrade screen)")
    print("=" * 78)
    print(f"  names ever held                       {len(names_held):5d}")
    print(f"  names ever held WHILE flagged         {len(names_hit):5d}")
    print(f"  position-days held                    {held_days:7,d}")
    print(f"  position-days held while flagged      {hit_days:7,d}  ({hit_days/max(held_days,1):.2%})")
    print(f"  downgrade events hit while held       {on_date:5d}  of {len(events)} in universe")
    print(f"  share of portfolio weight touched     {weight_touched:.2%}")
    print(f"\n  names affected: {', '.join(names_hit) if names_hit else '(none)'}")

    # --- Is the effect detectable? Settle this before reading any backtest. ---
    # This is a PAIRED test: the screened arm differs from the baseline only in the
    # names it swaps, so the difference is a small long/short book of (replacement
    # minus excluded). Its t-stat is (effect / pair_vol) * sqrt(n_events) — the weight
    # touched cancels, and what governs power is purely the NUMBER OF EVENTS.
    # A two-stock long/short pair runs at roughly 40% annualised vol.
    PAIR_VOL = 0.40
    print(f"\n  power (paired test, {on_date} usable events, pair vol ~{PAIR_VOL:.0%}):")
    for assumed in (0.05, 0.10, 0.20):
        t_exp = (assumed / PAIR_VOL) * np.sqrt(max(on_date, 1))
        needed = int(np.ceil((2 * PAIR_VOL / assumed) ** 2))
        print(f"    if downgraded names underperform by {assumed:>4.0%}: expected t={t_exp:>5.2f}, "
              f"need ~{needed:4d} events for t=2  ({'OK' if on_date >= needed else 'INSUFFICIENT'})")

    return {"names_hit": len(names_hit), "hit_days": hit_days, "held_days": held_days,
            "events_hit": on_date, "weight": weight_touched,
            "events_needed_10pct": int(np.ceil((2 * PAIR_VOL / 0.10) ** 2))}


def run_arm(f, mcap, adj, volume, close, elig, cfg) -> tuple[pd.Series, float, pd.DataFrame]:
    weights, _ = enhanced_weights(f, mcap, adj, elig, cfg)
    net, turnover = pnl(weights, adj, volume, close)
    return net, turnover, weights


def show(label: str, net: pd.Series, turnover: float, idx: pd.DatetimeIndex) -> dict:
    s = summary_stats(net.reindex(idx).fillna(0.0))
    print(f"  {label:34s} ann_ret {s['ann_return']:>+7.2%}  vol {s['ann_vol']:>6.2%}  "
          f"sharpe {s['sharpe']:>+5.2f}  maxDD {s['max_drawdown']:>+7.2%}  turn {turnover:4.1f}x")
    return s


def main(overlap_only: bool = False, bucket: str = "all") -> None:
    ratings = load_ratings()
    cfg = EnhancedMagicConfig()
    print(f"[load] sp500_pit …")
    adj, close, volume, spy, base, mcap, f, label = _load("sp500_pit", cfg, START,
                                                          END or pd.Timestamp.today().strftime("%Y-%m-%d"))
    elig_all = base if bucket == "all" else size_bucket(mcap, base, 0.0, 1 / 3)
    universe = sorted(adj.columns)

    rated = [t for t in rated_tickers(ratings, universe) if t in adj.columns]
    print(f"[ratings] Fitch rates {len(rated)} of {len(universe)} universe names "
          f"({len(rated)/len(universe):.1%})")

    # Restrict BOTH arms to the rated universe so the leverage tilt cancels.
    rated_mask = pd.Series(adj.columns.isin(rated), index=adj.columns)
    elig_rated = elig_all & rated_mask

    events = event_table(ratings, universe, fallen_angels=False)
    print(f"[events]  {len(events)} deduplicated downgrade events in universe, "
          f"{events['ticker'].nunique()} distinct names\n")

    # Power check uses the RATED baseline's own holdings.
    _, _, w_rated = run_arm(f, mcap, adj, volume, close, elig_rated, cfg)
    stats = overlap_report(w_rated, events, months=12)

    if stats["events_hit"] < stats["events_needed_10pct"]:
        print(f"\n  *** UNDERPOWERED: {stats['events_hit']} usable events against the "
              f"~{stats['events_needed_10pct']} needed to resolve a 10%/yr effect. ***")
        print("      Whatever the arms below print is NOISE. Do not read it as evidence")
        print("      in either direction, and do not tune the window to improve it.")
    if overlap_only:
        return

    idx = None
    print("\n" + "=" * 78)
    print(f"WITHIN-UNIVERSE COMPARISON  (both arms restricted to the {len(rated)} rated names)")
    print("=" * 78)
    results, nets = {}, {}
    net_rated, turn_rated, _ = run_arm(f, mcap, adj, volume, close, elig_rated, cfg)
    idx = net_rated.replace(0.0, np.nan).dropna().index
    results["baseline"] = show("baseline (rated universe)", net_rated, turn_rated, idx)
    nets["baseline"] = net_rated

    for months in (6, 12, 18):
        flags = flag_frame(events, adj.index, list(adj.columns), months).reindex(
            index=adj.index, columns=adj.columns, fill_value=False)
        net, turn, _ = run_arm(f, mcap, adj, volume, close, elig_rated & ~flags, cfg)
        results[f"screen_{months}m"] = show(f"exclude downgraded, trailing {months}m",
                                            net, turn, idx)
        nets[f"screen_{months}m"] = net

    fa = event_table(ratings, universe, fallen_angels=True)
    flags_fa = flag_frame(fa, adj.index, list(adj.columns), 12).reindex(
        index=adj.index, columns=adj.columns, fill_value=False)
    net_fa, turn_fa, _ = run_arm(f, mcap, adj, volume, close, elig_rated & ~flags_fa, cfg)
    results["fallen_angels_12m"] = show(f"exclude IG->HY only, trailing 12m "
                                        f"({len(fa)} events)", net_fa, turn_fa, idx)
    nets["fallen_angels_12m"] = net_fa

    # --- PAIRED test: the arms share ~97% of their positions, so the difference
    # series has far less variance than either arm. Testing the difference directly
    # is the correct power question; comparing two standalone Sharpes is not.
    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCE vs baseline  (screened minus baseline, daily)")
    print("=" * 78)
    print(f"  {'arm':26s} {'ann_diff':>9s} {'diff_vol':>9s} {'t-stat':>7s}   verdict")
    for k, net in nets.items():
        if k == "baseline":
            continue
        d = (net.reindex(idx).fillna(0.0) - net_rated.reindex(idx).fillna(0.0)).dropna()
        ann = d.mean() * 252
        vol = d.std() * np.sqrt(252)
        t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        verdict = "significant" if abs(t) > 2 else "not distinguishable from zero"
        print(f"  {k:26s} {ann:>+8.2%} {vol:>9.2%} {t:>+7.2f}   {verdict}")
        results[k]["paired_t"] = t
        results[k]["ann_diff"] = ann

    print("\n  --- context only, NOT comparable (different universe) ---")
    net_full, turn_full, _ = run_arm(f, mcap, adj, volume, close, elig_all, cfg)
    show("unrestricted baseline (all names)", net_full, turn_full, idx)
    show("SPY", spy, 0.0, idx)

    base_sh = results["baseline"]["sharpe"]
    print("\n" + "=" * 78)
    print("SHARPE DELTA vs rated baseline")
    print("=" * 78)
    for k, v in results.items():
        if k == "baseline":
            continue
        print(f"  {k:24s} {v['sharpe'] - base_sh:>+6.2f}")
    print("\n  Interpretation: with the overlap above, treat |delta| < 0.10 as noise.")

    out = ROOT / "results" / "downgrade_screen"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(out / "summary.csv")
    events.to_csv(out / "downgrade_events.csv", index=False)
    print(f"\n  wrote {out}/summary.csv and downgrade_events.csv")


if __name__ == "__main__":
    main(overlap_only="--overlap-only" in sys.argv)
