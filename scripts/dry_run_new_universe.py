"""Dry-run the LIVE ranking on the widened universe, and diff it against the old one.

WHY. The 2026-08-28 change took the universe from 764 to 964 names by adding five eurozone
venues and all of non-eurozone Europe. Before `live` advances, the question is not whether the
code runs but what the BOOK would actually look like: how many European names reach the top 30,
which US names they displace, and whether anything ranks absurdly (the signature of a currency
or units bug surviving the fixes).

METHOD. One panel pull, two rankings off it:
    NEW = everything (US + eurozone + non-eurozone)
    OLD = the same panels restricted to US + eurozone, i.e. the universe as it stood before
Ranking the SAME panels twice makes the comparison exact — any difference is the universe
change alone, not a different pull, a different day, or a different data state.

READ-ONLY. It does not touch results/paper/ranking.json or panels.pkl (which the live runner
caches monthly and would otherwise reuse, since the cached month is already 2026-08), it places
no orders and needs no broker. Config mirrors run_paper.py exactly: use_graham=False,
min_mcap_usd=500e6.

Run: python3 scripts/dry_run_new_universe.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.universe import european_non_eur_tickers  # noqa: E402
from paper.live_data import fetch_live_panels  # noqa: E402
from paper.rank import build_eligibility, rank_report, todays_ranking  # noqa: E402
from paper.universe import paper_universe  # noqa: E402
from strategies.magic_formula import EnhancedMagicConfig  # noqa: E402

TOP_N = 30          # cfg.top_n — the names that would actually be held
MIN_MCAP = 500e6    # run_paper.py:106


def _region(t: str, non_eur: set[str]) -> str:
    if "." not in t:
        return "US"
    return "non-EUR Europe" if t in non_eur else "eurozone"


def main() -> None:
    cfg = EnhancedMagicConfig(use_graham=False)
    non_eur = set(european_non_eur_tickers())
    tickers = paper_universe()
    print(f"[load] {len(tickers)} names (US + all Europe) — the heavy pull, ~15-25 min …")
    panels = fetch_live_panels(tickers)
    live = list(panels["adj"].columns)
    print(f"[ok]   {len(live)} resolved\n")

    rank_new = todays_ranking(panels, cfg, min_mcap_usd=MIN_MCAP)

    # OLD = same panels, non-eurozone names removed. Restricting the ELIGIBILITY (not the pull)
    # is what makes this a controlled comparison.
    old_panels = dict(panels)
    keep = [t for t in live if t not in non_eur]
    old_panels["adj"] = panels["adj"][keep]
    old_panels["mcap"] = panels["mcap"][keep]
    old_panels["f"] = {k: v[[c for c in v.columns if c in keep]] for k, v in panels["f"].items()}
    rank_old = todays_ranking(old_panels, cfg, min_mcap_usd=MIN_MCAP)

    elig = build_eligibility(panels, cfg, MIN_MCAP)
    elig_today = elig.iloc[-1]
    print("=" * 96)
    print("UNIVERSE AND ELIGIBILITY")
    print("=" * 96)
    rows = {}
    for t in live:
        r = _region(t, non_eur)
        d = rows.setdefault(r, [0, 0, 0])
        d[0] += 1
        d[1] += int(bool(elig_today.get(t, False)))
        d[2] += int(t in rank_new.index)
    print(f"  {'region':17s} {'pulled':>7s} {'eligible':>9s} {'ranked':>7s}")
    for r in ("US", "eurozone", "non-EUR Europe"):
        if r in rows:
            n, e, k = rows[r]
            print(f"  {r:17s} {n:7d} {e:9d} {k:7d}")
    print(f"  {'TOTAL':17s} {len(live):7d} {int(elig_today.sum()):9d} {len(rank_new):7d}")

    print("\n" + "=" * 96)
    print(f"TOP {TOP_N} ON THE NEW UNIVERSE — this is the book that would be held")
    print("=" * 96)
    rep = rank_report(panels, cfg, top_n=TOP_N, min_mcap_usd=MIN_MCAP)
    rep["region"] = [_region(t, non_eur) for t in rep["ticker"]]
    print(rep.to_string(index=False))

    new_top = list(rank_new.index[:TOP_N])
    old_top = list(rank_old.index[:TOP_N])
    added = [t for t in new_top if t not in old_top]
    dropped = [t for t in old_top if t not in new_top]

    print("\n" + "=" * 96)
    print(f"WHAT THE WIDENING CHANGES IN THE TOP {TOP_N}")
    print("=" * 96)
    mix = pd.Series([_region(t, non_eur) for t in new_top]).value_counts()
    mix_old = pd.Series([_region(t, non_eur) for t in old_top]).value_counts()
    print(f"  {'region':17s} {'old top30':>10s} {'new top30':>10s}")
    for r in ("US", "eurozone", "non-EUR Europe"):
        print(f"  {r:17s} {int(mix_old.get(r, 0)):10d} {int(mix.get(r, 0)):10d}")
    print(f"\n  turnover this change would cause: {len(added)} of {TOP_N} names "
          f"({len(added) / TOP_N:.0%})")
    print(f"  ENTERS : {', '.join(added) or '(none)'}")
    print(f"  LEAVES : {', '.join(dropped) or '(none)'}")

    print("\n" + "=" * 96)
    print("SANITY — a currency or units bug shows up as an absurd market cap")
    print("=" * 96)
    mc = pd.Series(panels["mcap_usd_latest"]).dropna().sort_values(ascending=False)
    print(f"  largest 5 by USD mcap:  " +
          ", ".join(f"{t} ${v / 1e9:,.0f}bn" for t, v in mc.head(5).items()))
    print(f"  smallest 5 eligible:    " +
          ", ".join(f"{t} ${v / 1e9:,.2f}bn"
                    for t, v in mc[mc.index.isin(rank_new.index)].tail(5).items()))
    absurd = mc[mc > 5e12]
    print(f"  above $5tn (impossible today): {list(absurd.index) or 'none'}")
    gbp = [t for t in rank_new.index[:TOP_N] if t.endswith(".L")]
    print(f"  .L names in the top {TOP_N}: {gbp or 'none'}"
          f"{'  <-- verify units before live (scripts/check_ib_price_units.py)' if gbp else ''}")

    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "dry_run_new_universe.csv", index=False)
    print(f"\n  wrote {out}/dry_run_new_universe.csv")
    print("  NB read-only: ranking.json / panels.pkl untouched.")


if __name__ == "__main__":
    main()
