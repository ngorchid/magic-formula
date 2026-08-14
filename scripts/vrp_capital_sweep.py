"""What the options-VRP sleeve actually does at a REAL account size.

WHY. Every capital number for this strategy came out of `vrp_basket_mc.py` at a $100k budget
with CONTINUOUS position sizing. The live sizer does not size continuously:

    contracts = int((cfg.risk_per_trade * cfg.budget) // (max_loss * 100))   # strategy.py:368

A name whose single contract risks more than the per-position budget is not sized small — it is
not traded at all. That truncation is invisible to continuous sizing and it is the binding
constraint on a small account, so the headline Sharpe was measured in a regime the account will
not be in. This sweeps the budget with the integer floor switched on.

Two things move together as the budget falls, and they pull in OPPOSITE directions:
  - the TRADEABLE SET shrinks, because one contract of an expensive name no longer fits; and
  - the names that drop out first are the CHEAPEST to execute, because contract size scales with
    spot x IV while execution cost scales inversely with liquidity. SPY is the cheapest name in
    the basket (1.0% of credit) and one of the first to become unsizeable.

So shrinking the account does not just scale the strategy down, it forces it into the expensive
half of its own basket — where the cost guard then refuses to trade. Both effects are modelled.

    python3 scripts/vrp_capital_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vrp_basket_mc import (BASKET, MAXLOSS, POOL, POOL_CREDIT_FRAC,  # noqa: E402
                           name_cost, simulate)

BUDGETS = [15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 50_000, 75_000, 100_000, 200_000]
RISK_PER_TRADE = 0.03
# MULTI-SEED IS NOT OPTIONAL HERE. Changing the budget changes which names are unsizeable, which
# changes how many random draws the loop consumes, so a fixed seed gives each budget a DIFFERENT
# stream — single-seed runs of this sweep showed a spurious 0.35-Sharpe "CAT hurts above $133k"
# effect and overstated $50k by 0.23, both of which vanish on averaging.
SEEDS = range(24)


def main() -> None:
    pool = pd.read_csv(POOL)
    common = dict(pool=pool, years=40.0, pass_rate=0.377, rho_signal=0.72, rho_pnl=0.47,
                  n_names=13, max_positions=6, risk_per_trade=RISK_PER_TRADE, max_per_sector=2,
                  apply_constraints=True, per_name_cost=True, cheapest_first=True,
                  integer_contracts=True)

    print("=" * 100)
    print("OPTIONS-VRP — CAPITAL SWEEP WITH WHOLE-CONTRACT SIZING")
    print("=" * 100)

    print(f"\n  Per-contract max loss (strike width x 100), and the budget at which one contract"
          f"\n  first fits inside {RISK_PER_TRADE:.0%} of the sleeve:\n")
    print(f"  {'name':6}{'maxloss/ct':>12}{'cost (dict)':>13}{'cost (pool)':>13}"
          f"{'min budget':>13}")
    print("  " + "-" * 58)
    for nm in sorted(BASKET, key=lambda n: MAXLOSS[n]):
        print(f"  {nm:6}{MAXLOSS[nm]:>12,}{name_cost(nm, 'dict'):>12.1%}"
              f"{name_cost(nm, 'pool'):>13.1%}{MAXLOSS[nm]/RISK_PER_TRADE:>13,.0f}")

    for basis in ("dict", "pool"):
        lab = ("CREDIT dict as-is" if basis == "dict" else
               f"credit re-derived at {POOL_CREDIT_FRAC:.1%} of width (pool-consistent)")
        print(f"\n\n  {'='*94}\n  COST BASIS: {lab}\n  {'='*94}")
        print(f"\n  {'budget':>9}{'tradeable':>11}{'trades/yr':>11}{'ann ret':>10}{'vol':>8}"
              f"{'Sharpe':>12}{'maxDD':>9}{'avg risk':>10}{'flat':>7}{'skip':>7}")
        print("  " + "-" * 91)
        rows = []
        for b in BUDGETS:
            rs = [simulate(rng=np.random.default_rng(s), budget=float(b), credit_basis=basis,
                           **common) for s in SEEDS]
            g = lambda k: np.array([r[k] for r in rs])  # noqa: E731
            sh, an = g("sharpe"), g("ann_return")
            se = sh.std(ddof=1) / np.sqrt(len(rs))
            n_ok = 13 - len(rs[0]["unsizeable_names"])
            rows.append({"budget": b, "basis": basis, "tradeable": n_ok, "sharpe": sh.mean(),
                         "sharpe_se": se, "ann_return": an.mean(), "ann_usd": an.mean() * b,
                         "maxdd": g("maxdd").mean(), "ann_vol": g("ann_vol").mean(),
                         "trades_yr": g("trades_yr").mean(), "skip_rate": g("skip_rate").mean(),
                         "avg_risk_pct": g("avg_risk_pct").mean(),
                         "unsizeable": "|".join(rs[0]["unsizeable_names"])})
            print(f"  {b:>9,}{n_ok:>8}/13{g('trades_yr').mean():>11.1f}{an.mean():>+10.2%}"
                  f"{g('ann_vol').mean():>8.2%}{sh.mean():>+7.2f} ±{se:<4.2f}"
                  f"{g('maxdd').mean():>+9.1%}{g('avg_risk_pct').mean():>10.1%}"
                  f"{g('pct_days_flat').mean():>7.0%}{g('skip_rate').mean():>7.0%}")
            if rs[0]["unsizeable_names"]:
                print(f"            unsizeable: {', '.join(rs[0]['unsizeable_names'])}")
        pd.DataFrame(rows).to_csv(
            ROOT / "results" / "spx_vrp" / f"capital_sweep_{basis}.csv", index=False)

    print("\n  Reference: CONTINUOUS sizing at $100k scores +0.68 ±0.04 on the same 24 seeds —")
    print("  i.e. the whole-contract floor costs NOTHING once every name is sizeable. It is not")
    print("  a drag on a big account; it is a CLIFF on a small one. (An earlier single-seed run")
    print("  put continuous sizing at +0.86; that was seed noise, not a real gap.)")
    print("\n  Wrote results/spx_vrp/capital_sweep_*.csv")


if __name__ == "__main__":
    main()
