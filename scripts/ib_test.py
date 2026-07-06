"""Safe IB paper-connection smoke test — verifies the plumbing WITHOUT trading.

Checks the things the strategy needs but doesn't run the strategy: connection, US +
European contract resolution (the main unknown), and account read. It writes NO state
and places NO strategy orders, so there is nothing to undo.

  python scripts/ib_test.py            # connect + resolve contracts + read account
  python scripts/ib_test.py --order    # additionally place 1 share and immediately flatten
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.broker import Broker, ib_contract_spec  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# a spread of US + European names — the EU ones exercise the exchange/currency mapping
TEST_TICKERS = ["AAPL", "NVDA", "SAP.DE", "MC.PA", "ASML.AS", "SIE.DE", "NESN.SW", "SHEL.L"]


def main(do_order: bool = False) -> None:
    broker = Broker(host="127.0.0.1", port=7497, client_id=5, dry_run=False)
    if not broker.connect():
        logging.error("Could not connect. Is IB Gateway (paper) running on port 7497 "
                      "with the API enabled and 127.0.0.1 trusted?")
        return
    try:
        print("\n=== Account ===")
        nl = broker.net_liq()
        print(f"  NetLiquidation: {f'{nl[0]:,.0f} {nl[1]}' if nl else 'n/a'}  (commingled/base ccy — not used by strategy)")
        print(f"  Open positions: {broker.ib_positions() or '(none)'}")

        print("\n=== Contract resolution (US + Europe) ===")
        ok, bad = [], []
        for t in TEST_TICKERS:
            sym, ccy, exch = ib_contract_spec(t)
            c = broker.qualify(t)
            px = broker.price(t) if c else None
            if c and px:
                print(f"  ✓ {t:9s} -> {sym} {ccy}@{exch or 'SMART'}  last {px}")
                ok.append(t)
            else:
                print(f"  ✗ {t:9s} -> {sym} {ccy}@{exch or 'SMART'}  UNRESOLVED")
                bad.append(t)
        print(f"\n  resolved {len(ok)}/{len(TEST_TICKERS)}"
              + (f"; FAILED: {bad} (need mapping fixes)" if bad else " — all good"))

        if do_order and ok:
            t = "AAPL" if "AAPL" in ok else ok[0]
            print(f"\n=== Order path test: buy 1 {t}, then flatten ===")
            if broker.order(t, "BUY", 1):
                broker.ib.sleep(2)
                print(f"  positions after buy: {broker.ib_positions()}")
                broker.order(t, "SELL", 1)   # flatten immediately
                broker.ib.sleep(2)
                print(f"  positions after flatten: {broker.ib_positions()}")
    finally:
        broker.disconnect()
        print("\nDone. (No strategy state written.)")


if __name__ == "__main__":
    main(do_order="--order" in sys.argv)
