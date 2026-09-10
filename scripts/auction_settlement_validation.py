"""Does the CMT-based backtest actually describe the FUTURES it claims to trade?

THE GAP THIS CLOSES. The auction sleeve's edge is measured on FRED CMT yields -- a fitted
daily par curve -- with P&L modelled as belly_DV01 x change-in-fly. That is a MODEL of what
the futures butterfly would have earned, never a measurement of what it did. Four things can
break the mapping:

  cheapest-to-deliver   a future tracks whichever deliverable bond is cheapest, and that
                        changes. ZN is already known to behave like a ~7y, not a 10y.
  DV01 drift            futures DV01 = CTD DV01 / conversion factor, and it moves with yield
                        levels. The fly's neutrality depends on those weights being right.
  rolls                 quarterly expiry; the CTD can change and the price series jumps. The
                        CMT curve has no such discontinuity.
  basis and carry       a futures price is not a spot bond price.

If the model tracks the real thing, the +1.56bp gross / +0.81bp net stands. If it does not,
the sleeve's economics change.

WHAT THIS DOES NOT VALIDATE: slippage. A settlement price is an end-of-day mark, not a fill.
Nothing in a daily historical series says whether you could transact near it. Only live fills
at size do -- which is precisely why this is worth running BEFORE buying tick data, not after.

PER-CONTRACT, NOT CONTINUOUS. yfinance's ZN=F et al. are continuous front-month series with
roll gaps; measured elsewhere in this repo, roll-window days run ~1.85x normal volatility.
Splicing those into a 5-day event window would inject a fake move exactly where precision is
needed. So each event is priced on the SPECIFIC contract month the sleeve would have held,
chosen by the live `resolve_contract` -- the same function that picks the real trade's month.

REQUEST BUDGET. IB paces historical data at roughly 60 requests per 10 minutes. Naively this
would be 3 legs x hundreds of events; instead each (symbol, contract-month) is fetched ONCE
over its whole life and cached to disk, then sliced per event. Re-runs cost nothing.

Usage:
  python scripts/auction_settlement_validation.py --plan          # offline: what it would fetch
  python scripts/auction_settlement_validation.py --since 2019    # connect to IB and validate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Contract specs come from the DEPLOYED sleeve, never a copy. A duplicated DV01 table would
# drift silently against the thing being validated, which is the failure mode the
# documentation's "repo drift" section exists to prevent.
TA_REPO = ROOT.parent / "treasury-auction"
if not TA_REPO.exists():
    raise SystemExit(f"treasury-auction repo not found at {TA_REPO}")
sys.path.insert(0, str(TA_REPO))

from treasury_auction.contracts import CONTRACTS, FLIES, _solve_wings  # noqa: E402
from treasury_auction.execution import resolve_contract  # noqa: E402
from treasury_auction.sizing import SizingConfig, solve_sizing  # noqa: E402
from treasury_auction_lab import (  # noqa: E402
    Legs, build_measures, event_panel, fetch_auctions, fetch_yields, leg_pnl, parse_tenor,
)

CACHE = ROOT / "data" / "cache" / "treasury" / "ib_settlements"
OUT = ROOT / "results" / "auction_settlement"
BELLY_DV01_TARGET = 135.0    # fallback only, if the live sizer skips a tenor
BUDGET = 25_000.0            # the deployed budget (see the sleeve's .env)
PACE_SECONDS = 11.0          # ~60 requests / 10 min, with headroom


_LOTS_CACHE: dict[int, dict[str, int]] = {}


def fly_lots(tenor: int) -> dict[str, int]:
    """Integer lots per leg, from the LIVE sizer at the deployed budget.

    Uses `solve_sizing` rather than a local DV01 target so the validation prices the fly the
    sleeve would actually have held. A hand-rolled target got this wrong: aiming at $135/bp
    (the 5y's DV01, quoted in the doc's cost note) produced a 2-lot TN belly for the 10y
    against the live 5, and a smaller fly carries proportionally more DV01 residual from
    integer rounding -- which would have shown up as tracking error and been misread as the
    CMT model failing.

    Positive = long. The snapback leg is long the belly and short both wings.
    """
    if tenor in _LOTS_CACHE:
        return _LOTS_CACHE[tenor]
    spec = FLIES[tenor]
    sized = solve_sizing(SizingConfig(budget=BUDGET))
    s = sized.get(tenor)
    if s is None or not getattr(s, "belly_lots", 0):
        # Tenor not tradeable at this budget (the doc's $10k row skips the 10y). Fall back to
        # the DV01 target so the mapping can still be measured, and say so.
        n_belly = max(int(round(BELLY_DV01_TARGET / CONTRACTS[spec.belly].dv01)), 1)
        n_lo, n_hi, _ = _solve_wings(spec, n_belly * CONTRACTS[spec.belly].dv01)
        print(f"  NOTE {tenor}y not sized by the live sizer at ${BUDGET:,.0f}; "
              f"falling back to a {BELLY_DV01_TARGET:.0f}$/bp belly")
    else:
        n_belly = int(s.belly_lots)
        n_lo, n_hi, _ = _solve_wings(spec, n_belly * CONTRACTS[spec.belly].dv01)
    out = {spec.belly: n_belly, spec.lo: -n_lo, spec.hi: -n_hi}
    _LOTS_CACHE[tenor] = out
    return out


def event_windows(since: str | None) -> pd.DataFrame:
    """One row per new-issue 5y/10y auction with the entry/exit dates and CMT-model P&L."""
    auctions = fetch_auctions()
    yields = fetch_yields()
    auctions["tenor"] = auctions["securityTerm"].map(parse_tenor)
    auctions = auctions.dropna(subset=["tenor"])
    auctions["tenor"] = auctions["tenor"].astype(int)
    auctions["reopen"] = auctions["reopening"].astype(str).str.strip().str.lower().eq("yes")
    auctions["size_z"] = 0.0
    a = auctions[(~auctions["reopen"]) & auctions["tenor"].isin([5, 10])].copy()
    if since:
        a = a[a["auctionDate"] >= pd.Timestamp(since)]

    measures = build_measures(yields)
    panel = event_panel(measures, a)
    if panel.empty:
        return pd.DataFrame()
    pnl = leg_pnl(panel[panel["measure"] == "fly"], Legs())
    if pnl.empty:
        return pd.DataFrame()

    # Business-day entry/exit around the auction, matching Legs(): pivot T-1 -> exit T+3
    idx = measures[5].index
    rows = []
    for _, r in pnl.iterrows():
        adate = pd.Timestamp(r["auctionDate"])
        loc = idx.searchsorted(adate)
        if loc - 1 < 0 or loc + 3 >= len(idx):
            continue
        entry, exit_ = idx[loc + Legs().pivot], idx[loc + Legs().exit]
        tenor = int(r["tenor"])
        belly_dv01 = fly_lots(tenor)[FLIES[tenor].belly] * CONTRACTS[FLIES[tenor].belly].dv01
        rows.append({
            "tenor": tenor, "cusip": r["cusip"], "auctionDate": adate,
            "entry": entry, "exit": exit_,
            "model_bp": float(r["snapback"]),
            "model_usd": float(r["snapback"]) * belly_dv01,
        })
    return pd.DataFrame(rows)


def needed_contracts(ev: pd.DataFrame) -> dict[tuple[str, str], list]:
    """{(symbol, YYYYMM) -> [event indices]} using the LIVE month-resolution rule."""
    need: dict[tuple[str, str], list] = {}
    for i, r in ev.iterrows():
        for sym in fly_lots(int(r["tenor"])):
            try:
                mth = resolve_contract(sym, r["exit"], today=r["entry"])
            except RuntimeError:
                continue
            need.setdefault((sym, mth), []).append(i)
    return need


_MONTHMAP: dict[str, dict[str, str]] = {}


def contract_months(ib, symbol: str) -> dict[str, str]:
    """{YYYYMM -> the contract's actual lastTradeDate YYYYMMDD} for everything IB still knows.

    Two things this exists for. `lastTradeDateOrContractMonth` is a full DATE (20260930), not
    a month, so passing YYYYMM fails to qualify at all. And IB's retention for EXPIRED futures
    is about ONE YEAR -- measured 2026-09-10: ZT/ZF back to 20250930, ZN/TN/ZB to 20251219 --
    so enumerating tells us up front which events are priceable instead of discovering it as
    hundreds of silent misses.
    """
    if symbol in _MONTHMAP:
        return _MONTHMAP[symbol]
    from ib_insync import Future
    c = Future(symbol=symbol, exchange=CONTRACTS[symbol].exchange,
               currency="USD", includeExpired=True)
    out: dict[str, str] = {}
    try:
        for d in ib.reqContractDetails(c):
            ltd = d.contract.lastTradeDateOrContractMonth
            if len(ltd) >= 6:
                out[ltd[:6]] = ltd
    except Exception as exc:  # noqa: BLE001
        print(f"  {symbol}: contract enumeration failed: {exc}")
    _MONTHMAP[symbol] = out
    return out


def fetch_history(ib, symbol: str, yyyymm: str) -> pd.DataFrame | None:
    """Daily settlement history for ONE expired contract month, cached to disk.

    `includeExpired=True` is what makes an expired future resolvable at all. IB's retention
    for expired contracts is limited and undocumented, so a miss is expected rather than an
    error -- coverage is reported instead of assumed.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol}_{yyyymm}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return None if df.empty else df.set_index("date")

    from ib_insync import Future
    ltd = contract_months(ib, symbol).get(yyyymm)
    if ltd is None:
        # Beyond IB's ~1y expired retention, or not a listed month. Cache the miss so a
        # re-run does not pay the round trip again.
        pd.DataFrame(columns=["date"]).to_csv(path, index=False)
        return None
    c = Future(symbol=symbol, lastTradeDateOrContractMonth=ltd,
               exchange=CONTRACTS[symbol].exchange, currency="USD", includeExpired=True)
    try:
        qual = ib.qualifyContracts(c)
        if not qual:
            pd.DataFrame(columns=["date"]).to_csv(path, index=False)
            return None
        # endDateTime must be the CONTRACT'S OWN expiry, not "now". An empty string asks for
        # data ending today, which returns nothing for a contract that stopped trading years
        # ago. 1 Y back from expiry covers the whole period a quarterly Treasury future is
        # actively quoted.
        end = (pd.Timestamp(ltd) + pd.Timedelta(days=1)).to_pydatetime()
        bars = ib.reqHistoricalData(
            qual[0], endDateTime=end, durationStr="1 Y", barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True, formatDate=1)
    except Exception as exc:  # noqa: BLE001
        print(f"    {symbol} {yyyymm}: {type(exc).__name__} {exc}")
        return None
    finally:
        time.sleep(PACE_SECONDS)

    if not bars:
        # DO NOT cache this as a miss. An empty result here is usually TRANSIENT -- most often
        # IB error 162 "Trading TWS session is connected from a different IP address", which
        # means the account's market-data session is bound to another machine (e.g. the
        # Windows box running the live sleeves) and blocks historical data everywhere else.
        # Caching it would make a later, working run silently return nothing. Only a month
        # ABSENT from `contract_months` is a permanent miss, and that is cached above.
        print(f"    {symbol} {yyyymm}: qualified but returned no bars — not cached, "
              f"will retry (check IB error 162 / market-data session)")
        return None
    df = pd.DataFrame([{"date": pd.Timestamp(b.date), "close": float(b.close)} for b in bars])
    df.to_csv(path, index=False)
    return df.set_index("date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="offline: report the request budget")
    ap.add_argument("--since", default="2015-01-01", help="earliest auction date")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=18)
    args = ap.parse_args()

    ev = event_windows(args.since)
    if ev.empty:
        print("no events"); return
    need = needed_contracts(ev)

    print(f"  {len(ev)} new-issue 5y/10y events from {args.since}")
    print(f"  lots per fly: 5y {fly_lots(5)},  10y {fly_lots(10)}")
    print(f"  distinct (symbol, contract-month) to fetch: {len(need)}")
    print(f"  cached already: {sum((CACHE / f'{s}_{m}.csv').exists() for s, m in need)}")
    est = (len(need) - sum((CACHE / f'{s}_{m}.csv').exists() for s, m in need)) * PACE_SECONDS
    print(f"  estimated fetch time at IB pacing: {est / 60:.0f} min")
    if args.plan:
        print("\n  --plan: nothing fetched. Start IB Gateway/TWS and re-run without --plan.")
        return

    from ib_insync import IB
    ib = IB()
    try:
        ib.connect("127.0.0.1", args.port, clientId=args.client_id, timeout=15)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  IB connect failed ({exc!r}). Is Gateway/TWS running on port "
              f"{args.port}? Use --plan to inspect scope offline.")
        return

    hist: dict[tuple[str, str], pd.DataFrame] = {}
    for k, (sym, mth) in enumerate(sorted(need), 1):
        print(f"  [{k}/{len(need)}] {sym} {mth}", end="\r")
        h = fetch_history(ib, sym, mth)
        if h is not None:
            hist[(sym, mth)] = h
    ib.disconnect()
    print(f"\n  fetched/cached {len(hist)} of {len(need)} contract months")

    rows = []
    for i, r in ev.iterrows():
        lots = fly_lots(int(r["tenor"]))
        usd, ok = 0.0, True
        for sym, n in lots.items():
            try:
                mth = resolve_contract(sym, r["exit"], today=r["entry"])
            except RuntimeError:
                ok = False; break
            h = hist.get((sym, mth))
            if h is None:
                ok = False; break
            e0 = h["close"].reindex([r["entry"]]).iloc[0] if r["entry"] in h.index else np.nan
            e1 = h["close"].reindex([r["exit"]]).iloc[0] if r["exit"] in h.index else np.nan
            if not (np.isfinite(e0) and np.isfinite(e1)):
                ok = False; break
            usd += n * (e1 - e0) * CONTRACTS[sym].point_value
        if ok:
            rows.append({**r.to_dict(), "actual_usd": usd})

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    if res.empty:
        print("\n  NO EVENTS PRICED — IB returned no usable history for the needed months.")
        print("  Expired-future retention is the likely cause; try a more recent --since.")
        return
    res.to_csv(OUT / "settlement_validation.csv", index=False)

    print("\n" + "=" * 88)
    print("CMT MODEL vs REAL FUTURES SETTLEMENTS — snapback leg, per event, USD")
    print("=" * 88)
    print(f"  events priced on real contracts: {len(res)} of {len(ev)} "
          f"({len(res) / len(ev):.0%})")
    for tenor, g in list(res.groupby("tenor")) + [("ALL", res)]:
        if len(g) < 5:
            continue
        b, a_ = np.polyfit(g["model_usd"], g["actual_usd"], 1)
        corr = float(g["model_usd"].corr(g["actual_usd"]))
        print(f"\n  --- {tenor} ---   n={len(g)}")
        print(f"    model  mean ${g['model_usd'].mean():>+9,.0f}   sd ${g['model_usd'].std():>8,.0f}")
        print(f"    actual mean ${g['actual_usd'].mean():>+9,.0f}   sd ${g['actual_usd'].std():>8,.0f}")
        print(f"    correlation {corr:>+.3f}   regression slope {b:>+.3f} (1.00 = model is right)")
        print(f"    mean tracking error ${(g['actual_usd'] - g['model_usd']).abs().mean():>8,.0f}")
        print(f"    sign agreement {float((np.sign(g['actual_usd']) == np.sign(g['model_usd'])).mean()):.0%}")

    print("\n  READ: slope ~1 and high correlation means the CMT backtest describes the")
    print("  futures it claims to trade, and the measured edge stands. A slope well below 1")
    print("  means the fly's real DV01 weights under-deliver the modelled move. Low")
    print("  correlation with a fine mean means the model is right on average and noisy per")
    print("  event, which matters for sizing but not for the edge.")
    print(f"\nWrote {OUT / 'settlement_validation.csv'}")


if __name__ == "__main__":
    main()
