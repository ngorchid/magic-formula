"""Ask IB what it actually calls a ticker, instead of guessing the format.

WHY. `qualify` tries three formatting variants of the yfinance ticker (hyphen, space,
concatenated). On 2026-08-30 ALL THREE failed for VOLV-B.ST, NOVO-B.CO and RYA.IR — so the
share-class separator is not the problem, and inventing a fourth variant would be guessing
again. IB can be asked directly:

  * `reqMatchingSymbols(pattern)` is IB's own symbol search. It returns every instrument whose
    symbol or company name matches, with its primary exchange and currency. That is the
    authoritative answer to "what does IB call Volvo B".
  * The second hypothesis this tests is ROUTING, not naming. `qualify` builds
    `Stock(sym, "SMART", ccy, primaryExchange=exch)`. SMART does not route every European
    venue, and where it does not, the contract must name the exchange DIRECTLY
    (`Stock(sym, "SFB", "SEK")`). A contract can be perfectly well named and still fail to
    qualify through the wrong route, which looks identical from the outside.

So for each failing ticker this prints (a) what IB's search returns, and (b) whether the
contract qualifies via SMART, via the direct exchange, or not at all.

NO ORDERS ARE PLACED. Symbol search and contract qualification are metadata requests.

Run on the box, where TWS is reachable:
    python3 scripts/ib_symbol_probe.py                 # the known failures
    python3 scripts/ib_symbol_probe.py VOLV-B.ST FOO.ST

Feed the verdict back into `IB_SYMBOL_FIX` (naming) or `SUFFIX_MAP` (routing) in
paper/broker.py — only entries VERIFIED here, since a wrong symbol trades the wrong stock.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.broker import ib_contract_spec, ib_symbol_candidates  # noqa: E402

DEFAULT = ["VOLV-B.ST", "NOVO-B.CO", "RYA.IR"]


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from ib_insync import IB, Stock

    tickers = sys.argv[1:] or DEFAULT
    ib = IB()
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "7497"))
    try:
        ib.connect(host, port, clientId=19, timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"could not connect to {host}:{port} — {e}")
        return 2

    for t in tickers:
        base, ccy, exch = ib_contract_spec(t)
        print("\n" + "=" * 88)
        print(f"{t}   (we build: symbol={base!r} currency={ccy} primaryExchange={exch})")
        print("=" * 88)

        # (a) WHAT DOES IB CALL IT? Search on the root, before any share-class suffix, since
        # that is the part we are confident about.
        root = base.split("-")[0].split(" ")[0]
        try:
            matches = ib.reqMatchingSymbols(root) or []
        except Exception as e:  # noqa: BLE001
            matches = []
            print(f"  symbol search failed: {e}")
        rows = []
        for m in matches:
            c = m.contract
            if c.secType != "STK":
                continue
            if ccy and c.currency != ccy:
                continue          # keep the venue we are actually trying to trade
            rows.append((c.symbol, c.primaryExchange, c.currency, getattr(m, "description", "")))
        if rows:
            print(f"  IB symbol search for {root!r} ({ccy}):")
            for sym, pex, cur, desc in rows[:12]:
                print(f"    symbol={sym:<12} primaryExchange={pex:<10} {cur}  {desc[:38]}")
        else:
            print(f"  IB symbol search for {root!r} returned nothing in {ccy}"
                  f"{' (try the company name)' if matches else ''}")

        # (b) ROUTING. Same symbols, two routes. A contract can be named correctly and still
        # fail to qualify because SMART does not cover its venue.
        print("  qualification attempts:")
        cands = list(dict.fromkeys(ib_symbol_candidates(t) + [r[0] for r in rows]))
        for sym in cands:
            for route, kw in (("SMART", {"primaryExchange": exch}), (exch, {})):
                if not route:
                    continue
                try:
                    q = ib.qualifyContracts(Stock(sym, route, ccy, **kw))
                except Exception:
                    q = []
                mark = "OK  <-- USE THIS" if q else "no"
                print(f"    symbol={sym:<12} exchange={route:<10} {mark}")

    print("\n" + "=" * 88)
    print("Encode a VERIFIED result: naming -> IB_SYMBOL_FIX, routing -> SUFFIX_MAP")
    print("(paper/broker.py). A wrong entry trades the WRONG INSTRUMENT — verify first.")
    ib.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
