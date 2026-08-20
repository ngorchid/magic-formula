"""inv_vol and a vol-quintile universe filter, judged ON A HOLDOUT.

WHY A HOLDOUT, STRICTLY. Five experiments have now been run on this dataset (concentration,
turnover, the breadth x frequency grid, conditional IC, and this). Each one improved the answer.
That pattern is the selection problem itself: a best cell found after searching dozens is not the
same as one found once. So the rule here is that the configuration must be CHOSEN on 2012-2019
and the 2020+ column read only afterwards. If the choice flips between the two columns, the
in-sample result was noise and there is nothing here.

TWO LEVERS, both suggested by the conditional-IC result (idio-vol Q1 IC -0.0038, Q4 +0.0106):

  inv_vol   the champion DIVIDES the signal by idiosyncratic vol, which down-weights exactly the
            names where IC is highest. That is risk parity fighting alpha. One-line flag.
  vol_min   drop the quietest names from the universe entirely. NB the IC peak was Q4, NOT Q5 --
            the relation turns over at the top -- so "Q5 only" is not obviously the right cut and
            Q3+/Q4+/Q5 are all tested.

⚠ RESTRICTING THE UNIVERSE COSTS BREADTH. Q5-only leaves ~1/5 of the names, so taking the top 50
by |signal| means reaching much deeper into a smaller pool -- weaker average signal per name, and
a book concentrated in the highest-volatility stocks, where beta/sector neutralisation is doing
more work and may hold less well. Higher IC per name does not automatically mean higher Sharpe.

Fixed at the grid's best cell: 50 names, weekly (k=5), $1M. Quintiles are assigned
CROSS-SECTIONALLY PER DAY, so there is no lookahead.

Run: python3 scripts/reversal_holdout_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import summary_stats                        # noqa: E402
from data import download_ohlcv                            # noqa: E402
from data.universe import (sp1500_constituents, sp1500_sectors,  # noqa: E402
                           sp1500_tickers)
from strategies.equity_mn.neutralize import rolling_beta    # noqa: E402
from reversal_lab import Variant, build_weights             # noqa: E402
from reversal_fees_lab import concentrate                   # noqa: E402
from reversal_grid_lab import evaluate_periodic             # noqa: E402

SPLIT = "2020-01-01"
NAMES, K, BUDGET = 50, 5, 1_000_000.0


def main() -> None:
    print("loading mid+small …")
    panel = download_ohlcv(sorted(set(sp1500_tickers() + ["SPY"])), "2011-01-01", None)
    pf = panel["adj_close"].dropna(how="all", axis=1)
    vol = panel["volume"].reindex_like(pf)
    bench = pf["SPY"].pct_change(fill_method=None)
    prices = pf.drop(columns=["SPY"], errors="ignore")
    vol = vol.drop(columns=["SPY"], errors="ignore")
    r = prices.pct_change(fill_method=None)
    betas = rolling_beta(r, bench, 252)
    sectors = sp1500_sectors().reindex(prices.columns)
    idio = r.rolling(20).std()
    import yfinance as yf
    vix = (yf.download("^VIX", start="2011-01-01", auto_adjust=True,
                       progress=False)["Close"].squeeze() / 100.0)
    tier = sp1500_constituents().set_index("ticker")["tier"].reindex(prices.columns)
    ms = [c for c in prices.columns if tier.get(c) in ("mid", "small")]
    prices, vol, idio = prices[ms], vol[ms], idio[ms]

    rows = []
    for inv in (True, False):
        w_full = build_weights(
            Variant("v", horizons=(1, 3, 5, 10), news_filter=True, smooth=2,
                    inv_vol=inv, vix_scale=True),
            prices, vol, betas.reindex(columns=ms), sectors.reindex(ms), idio, vix)
        common = w_full.replace(0.0, np.nan).dropna(how="all").index
        w_full = w_full.reindex(common).fillna(0.0)
        vranks = idio.reindex(common).rank(axis=1, pct=True)      # per-day, no lookahead
        for vmin, vlab in ((0.0, "all names"), (0.4, "Q3+"), (0.6, "Q4+"), (0.8, "Q5 only")):
            w = w_full.where(vranks > vmin, 0.0) if vmin > 0 else w_full
            held = concentrate(w, NAMES // 2, NAMES // 2)
            g, net, ordd, ibyr, _ = evaluate_periodic(held, prices, vol, BUDGET, K)
            n_held = float((held != 0).sum(axis=1).mean())
            row = {"inv_vol": inv, "filter": vlab, "held": n_held, "ib": ibyr}
            for lab, sl in (("IS", slice(None, SPLIT)), ("OOS", slice(SPLIT, None))):
                row[f"g_{lab}"] = summary_stats(g.loc[sl].fillna(0.0))["sharpe"]
                row[f"n_{lab}"] = summary_stats(net.loc[sl].fillna(0.0))["sharpe"]
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"HOLDOUT — {NAMES} names, weekly, ${BUDGET:,.0f}. "
          f"CHOOSE on IS (2012-2019), then read OOS (2020+).")
    print("=" * 96)
    print(f"  {'inv_vol':9}{'filter':11}{'held':>6}{'IB%/yr':>8}"
          f"{'gross IS':>10}{'net IS':>9}   {'gross OOS':>11}{'net OOS':>10}")
    print("  " + "-" * 90)
    for _, x in df.iterrows():
        print(f"  {str(x['inv_vol']):9}{x['filter']:11}{x['held']:>6.0f}{x['ib']:>8.1%}"
              f"{x['g_IS']:>+10.2f}{x['n_IS']:>+9.2f}   {x['g_OOS']:>+11.2f}{x['n_OOS']:>+10.2f}")

    best_is = df.loc[df["n_IS"].idxmax()]
    best_oos = df.loc[df["n_OOS"].idxmax()]
    print(f"\n  BEST ON IS   : inv_vol={best_is['inv_vol']}, {best_is['filter']} "
          f"-> net IS {best_is['n_IS']:+.2f},  and OOS it delivers {best_is['n_OOS']:+.2f}")
    print(f"  best on OOS  : inv_vol={best_oos['inv_vol']}, {best_oos['filter']} "
          f"-> net OOS {best_oos['n_OOS']:+.2f}")
    if best_is["filter"] != best_oos["filter"] or best_is["inv_vol"] != best_oos["inv_vol"]:
        print("  ⚠ THE CHOICE FLIPS between the two halves — the in-sample ranking did not hold.")
    else:
        print("  the same configuration wins in both halves.")
    print("\n  Honest reading: the OOS number of the IS-CHOSEN row is the only one you could have")
    print("  actually earned. The best-OOS row is hindsight.")
    df.to_csv(ROOT / "results" / "reversal_holdout.csv", index=False)


if __name__ == "__main__":
    main()
