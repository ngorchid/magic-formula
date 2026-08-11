"""Trend-overlay paper runner — compute the target futures book and (optionally) trade it.

Default is a DRY RUN (no IB connection): pulls the ETF-proxy history, computes the blended
TSMOM signal, sizes target contracts to the budget, and prints the book + the order plan vs.
current positions. Add `--live` to connect to the IB *paper* gateway and place the orders
(front-month futures, market orders). Run this weekly; it also rolls (front-month resolver
skips contracts near expiry).

    python scripts/run_trend_paper.py                    # dry run, default budget
    python scripts/run_trend_paper.py --budget 300000 --mult 0.5
    python scripts/run_trend_paper.py --live             # execute on IB paper (needs gateway)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import download_ohlcv
from strategies.trend_futures.contracts import BY_MARKET, FUTURES, PROXY_ETFS
from strategies.trend_futures.execution import (
    FuturesBroker,
    HeldPosition,
    TrendPaperConfig,
    compute_targets,
    plan_roll_orders,
    safety_closes,
)


def _selftest() -> None:
    """Offline check of roll + hard safety logic with fabricated positions (no IB)."""
    today = "2026-07-10"
    targets = {"equity_us": 4, "oil": 2, "gold": -1}
    held = [
        HeldPosition("equity_us", "MES", "20260619", 3),   # old front -> should roll
        HeldPosition("oil",       "MCL", "20260721", 2),    # 11d to expiry, buffer 14 -> SAFETY
        HeldPosition("gold",      "MGC", "20260828", -1),   # == front -> no trade
    ]
    front = {"equity_us": "20260918", "oil": "20260820", "gold": "20260828"}
    print("SELF-TEST (today 2026-07-10)  — combined live flow, deduped")
    safety = safety_closes(held, BY_MARKET, today)
    done = {(o.ib_symbol, o.expiry) for o in safety}
    held_left = [h for h in held if (h.ib_symbol, h.expiry) not in done]
    rolls = plan_roll_orders(targets, held_left, front, BY_MARKET, True, today)
    print("  [SAFETY force-closes]")
    for o in safety:
        print(f"    {o.action} {o.qty} {o.ib_symbol} {o.expiry}  <- {o.reason}")
    print("  [ROLL + RECONCILE]")
    for o in rolls:
        print(f"    {o.action} {o.qty} {o.ib_symbol} {o.expiry}  <- {o.reason}")
    print("  (note: no contract appears in BOTH lists -> no double-close)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=200_000.0)
    ap.add_argument("--mult", type=float, default=1.0, help="overlay multiple (0.5, 1.0, ...)")
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--live", action="store_true", help="connect to IB paper and place orders")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=6)
    ap.add_argument("--selftest", action="store_true", help="offline roll/safety logic check")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.selftest:
        _selftest(); return

    cfg = TrendPaperConfig(budget=args.budget, target_vol=args.target_vol, overlay_multiple=args.mult)

    print(f"[1/3] proxy history for {len(PROXY_ETFS)} markets …")
    start = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    px = download_ohlcv(PROXY_ETFS, start)["adj_close"].dropna(how="all", axis=1)

    print("[2/3] target book …")
    tgt = compute_targets(px, cfg)
    gross = float(tgt["notional_used"].abs().sum())
    print("\n" + "=" * 78)
    print(f"TREND OVERLAY TARGET  (budget ${cfg.budget:,.0f}, {cfg.target_vol:.0%} vol × {cfg.overlay_multiple:g}, "
          f"{'micro' if cfg.use_micro else 'standard'})")
    print("=" * 78)
    print(f"  {'market':11s} {'sym':5s} {'signal':>7s} {'ann_vol':>8s} {'contracts':>10s} {'notional':>12s}")
    for m, r in tgt.iterrows():
        print(f"  {m:11s} {r['ib_symbol']:5s} {r['signal']:>+7.2f} {r['ann_vol']:>8.1%} "
              f"{int(r['contracts']):>10d} {r['notional_used']:>+12,.0f}")
    print(f"  {'GROSS notional':11s} {'':5s} {'':>7s} {'':>8s} {'':>10s} {gross:>12,.0f}")
    print(f"  net notional = {tgt['notional_used'].sum():+,.0f}   "
          f"gross/budget = {gross/cfg.budget:.1f}x")

    print("\n[3/3] order plan (roll + safety + reconcile) …")
    targets = {m: int(r["contracts"]) for m, r in tgt.iterrows()}
    if not args.live:
        print("  dry run: roll/safety planning needs live IB contract data (front months + held\n"
              "  expiries). Run `--selftest` to verify the roll logic offline, or `--live` to plan\n"
              "  against real positions. Target reconcile from flat:")
        for m, c in targets.items():
            if c:
                print(f"    {'BUY' if c > 0 else 'SELL'} {abs(c)} {tgt.loc[m, 'ib_symbol']} (front)")
        print("\n  (dry run — nothing sent)")
        return

    broker = FuturesBroker(port=args.port, client_id=args.client_id, dry_run=False)
    if not broker.connect():
        print("  IB connect failed — aborting live run."); return
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    front = broker.front_expiries(FUTURES, cfg.use_micro)
    held = broker.held_positions(FUTURES)

    # 1. HARD safety: force-close anything inside its delivery buffer.
    safety = safety_closes(held, BY_MARKET, today)
    done = {(o.ib_symbol, o.expiry) for o in safety}
    # 2. Roll + reconcile on what's left (safety-closed contracts treated as flat, so no
    #    double-close; roll still opens the target in the safe front for those markets).
    held_left = [h for h in held if (h.ib_symbol, h.expiry) not in done]
    rolls = plan_roll_orders(targets, held_left, front, BY_MARKET, cfg.use_micro, today)

    for label, batch in (("SAFETY", safety), ("ROLL+RECONCILE", rolls)):
        print(f"  [{label}] {len(batch)} orders")
        broker.execute(batch, BY_MARKET, cfg.use_micro)
    broker.disconnect()
    print("\n  live orders submitted.")


if __name__ == "__main__":
    main()
