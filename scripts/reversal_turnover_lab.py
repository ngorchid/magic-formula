"""Can a LONGER-HOLDING reversal book survive real IB commissions?

WHY. The concentration test (reversal_fees_lab) showed that trading FEWER NAMES does not help:
IB charges per SHARE, not per order, so a narrower book holds proportionally bigger positions,
trades the same dollars and pays the same ~6%/yr — while gross Sharpe falls on the sqrt(N)
breadth law. Concentration is strictly harmful under that fee schedule.

But cost/yr = TURNOVER x cost-per-turn, and the per-turn cost is genuinely tiny: ~$1.23 on a
~$5,000 position is ~2.5bp. What makes it 6%/yr is paying it ~250 times a year. So the untested
lever is not WHICH names but HOW OFTEN — reduce turnover and the cost falls proportionally,
without giving up breadth.

Three ways to hold longer, all already supported by `Variant`:
  smooth    average the target book over N days (damps day-to-day signal noise)
  band      no-trade band: leave a name alone unless its target moves more than band x avg weight.
            This is what rescued the trend overlay (Sharpe 0.655 -> 0.801 at $100k).
  horizons  drop the fast legs. A 1-day reversal must be re-traded daily by construction; a
            10-day one need not be.

⚠ THE TRADE-OFF IS REAL AND MAY NOT PAY. Reversal alpha decays fast: holding 5 days instead of 1
captures less than 5x the alpha per trade, so turnover and gross return fall TOGETHER. This
measures whether net improves, not whether turnover falls (it obviously does).

Judged on NET-OF-IB Sharpe, the only number that matters here, at both $1M and $100k.

Run: python3 scripts/reversal_turnover_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from backtest import summary_stats                      # noqa: E402
from data import download_ohlcv                          # noqa: E402
from data.universe import (sp1500_constituents, sp1500_sectors,  # noqa: E402
                           sp1500_tickers)
from strategies.equity_mn.neutralize import rolling_beta  # noqa: E402
from reversal_lab import Variant, build_weights           # noqa: E402
from reversal_fees_lab import concentrate                 # noqa: E402


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
    prices, vol = prices[ms], vol[ms]
    bt, sc, iv = betas.reindex(columns=ms), sectors.reindex(ms), idio.reindex(columns=ms)

    # The champion, then progressively slower variants of it.
    BASE = dict(horizons=(1, 3, 5, 10), news_filter=True, inv_vol=True, vix_scale=True)
    variants = [
        ("CHAMPION  smooth2, no band", dict(smooth=2)),
        ("smooth 3", dict(smooth=3)),
        ("smooth 5", dict(smooth=5)),
        ("smooth 10", dict(smooth=10)),
        ("smooth 2 + band 0.5", dict(smooth=2, band=0.5)),
        ("smooth 2 + band 1.0", dict(smooth=2, band=1.0)),
        ("smooth 2 + band 2.0", dict(smooth=2, band=2.0)),
        ("smooth 5 + band 1.0", dict(smooth=5, band=1.0)),
        ("slow horizons (5,10)", dict(smooth=2, horizons=(5, 10))),
        ("slow horizons + band 1.0", dict(smooth=2, horizons=(5, 10), band=1.0)),
        ("very slow (10,21)", dict(smooth=5, horizons=(10, 21), band=1.0)),
        # --- PERIODIC REBALANCE: trade on 1 day in k, nothing on the others ---
        ("REBAL every 2d", dict(smooth=2, rebal_k=2)),
        ("REBAL every 3d", dict(smooth=2, rebal_k=3)),
        ("REBAL weekly (5d)", dict(smooth=2, rebal_k=5)),
        ("REBAL weekly, slow (5,10)", dict(smooth=2, horizons=(5, 10), rebal_k=5)),
        ("REBAL weekly, slow+band", dict(smooth=2, horizons=(5, 10), band=1.0, rebal_k=5)),
        ("REBAL every 10d, slow (10,21)", dict(smooth=2, horizons=(10, 21), rebal_k=10)),
        ("REBAL monthly (21d), (10,21)", dict(smooth=2, horizons=(10, 21), rebal_k=21)),
    ]

    # PERIODIC REBALANCE — the thing the variants above do NOT test. They all re-target every
    # day; a slower SIGNAL still moves a little daily, and any move past the band is an order,
    # which is why order count barely fell. Rebalancing every k days places ZERO orders on the
    # other k-1 days, so order count falls by construction rather than by hoping the target
    # sits still.
    # NB weights are held FIXED between rebalances rather than drifting with prices. Over a
    # 5-10 day hold the difference is small, but it flatters neutrality slightly.
    def evaluate_periodic(w, prices, volume, budget, k, spread_bps=1.5):
        """Cost and return when SHARES are held fixed between rebalances.

        ⚠ The shared `evaluate()` cannot be used for this. It derives shares as
        weight x budget / price and diffs THAT — so a drifting price changes the implied share
        count every day even when the target weight is frozen, producing a phantom order daily.
        It models continuous rebalancing-to-weight, which is exactly what a periodic-rebalance
        test must not do. Reported 162 orders/day for a MONTHLY rebalance before this was fixed.

        Here shares are set on rebalance days and CARRIED. Between rebalances the position value
        drifts with price (as it must), P&L is shares x price change, and there are genuinely
        zero orders.
        """
        px = prices.reindex_like(w).ffill()
        rebal = pd.Series(False, index=w.index)
        rebal.iloc[::k] = True
        tgt = (w * budget).div(px).round().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        shares = tgt.where(rebal, np.nan).ffill().fillna(0.0)      # set, then CARRY
        traded = (shares - shares.shift(1)).abs().fillna(shares.abs())
        # P&L from holding those shares, as a fraction of budget.
        pnl = (shares.shift(1) * px.diff()).sum(axis=1) / budget
        notional = traded * px
        adv = (prices * volume).rolling(21).mean().reindex_like(w).ffill()
        adv = adv.fillna(adv.median().median())
        from backtest import LinearCostModel
        pct = LinearCostModel(spread_bps, 10.0).charge(notional, adv) / budget
        per = np.minimum(0.005 * traded, 0.01 * notional)
        per = per.where(traded > 0, 0.0).clip(lower=0.0)
        per = per.mask((traded > 0) & (per < 1.0), 1.0)
        ib = per.sum(axis=1) / budget
        orders = (traded > 0).sum(axis=1)
        turn = float((notional.sum(axis=1) / budget).mean() * 252)
        return pnl, pnl - pct - ib, float(orders.mean()), float(ib.mean() * 252), turn

    rows = []
    for label, kw in variants:
        cfg = {**BASE, **kw}
        k = cfg.pop("rebal_k", 1)
        w = build_weights(Variant(label, **cfg), prices, vol, bt, sc, iv, vix)
        held = concentrate(w, 100, 100)                  # breadth fixed; vary SPEED and CADENCE
        common = held.replace(0.0, np.nan).dropna(how="all").index
        held = held.reindex(common).fillna(0.0)
        out = {"label": label}
        for budget, tag in ((1_000_000.0, "1M"), (100_000.0, "100k")):
            g, net, ordd, ib_yr, turn = evaluate_periodic(held, prices, vol, budget, k)
            out["turnover"] = turn
            out[f"gross_{tag}"] = summary_stats(g.fillna(0.0))["sharpe"]
            out[f"net_{tag}"] = summary_stats(net.fillna(0.0))["sharpe"]
            out[f"ibyr_{tag}"] = ib_yr
            out[f"ord_{tag}"] = ordd
        rows.append(out)

    print("\n" + "=" * 104)
    print("TURNOVER REDUCTION — breadth held at 100L/100S; only the HOLDING SPEED varies")
    print("=" * 104)
    print(f"  {'variant':28}{'turn/yr':>9}{'gross':>8}"
          f"{'$1M net':>10}{'IB%/yr':>8}{'ord/d':>7}"
          f"{'$100k net':>11}{'IB%/yr':>8}{'ord/d':>7}")
    print("  " + "-" * 100)
    for x in rows:
        print(f"  {x['label']:28}{x['turnover']:>9.0f}{x['gross_1M']:>+8.2f}"
              f"{x['net_1M']:>+10.2f}{x['ibyr_1M']:>8.1%}{x['ord_1M']:>7.0f}"
              f"{x['net_100k']:>+11.2f}{x['ibyr_100k']:>8.1%}{x['ord_100k']:>7.0f}")
    print("\n  turn/yr = two-way turnover as a multiple of gross book. cost/yr ~= turnover x")
    print("  cost-per-turn, so this is the quantity that has to fall for the strategy to live.")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "reversal_turnover.csv", index=False)
    print(f"  wrote {ROOT}/results/reversal_turnover.csv")


if __name__ == "__main__":
    main()
