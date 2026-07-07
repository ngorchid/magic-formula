"""Safe IB paper-connection smoke test — verifies the plumbing WITHOUT trading.

Checks the things the strategy needs but doesn't run the strategy: connection, US +
European contract resolution (the main unknown), and account read. It writes NO state
and places NO strategy orders, so there is nothing to undo.

  python scripts/ib_test.py            # connect + resolve contracts + read account
  python scripts/ib_test.py --order    # additionally place 1 share and immediately flatten
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from paper.broker import Broker, ib_contract_spec  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv(ROOT / ".env")

# a spread of US + European names — the EU ones exercise the exchange/currency mapping
TEST_TICKERS = ["AAPL", "NVDA", "SAP.DE", "MC.PA", "ASML.AS", "SIE.DE", "NESN.SW", "SHEL.L"]


def main(do_order: bool = False) -> None:
    broker = Broker(host=os.getenv("IB_HOST", "127.0.0.1"),
                    port=int(os.getenv("IB_PORT", "7497")),
                    client_id=int(os.getenv("IB_CLIENT_ID", "5")),
                    dry_run=False)
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
        print("  (qualify = what the strategy needs for orders; IB price is informational —")
        print("   strategy marks come from yfinance, so a missing IB price is fine)")
        ok, bad = [], []
        for t in TEST_TICKERS:
            sym, ccy, exch = ib_contract_spec(t)
            c = broker.qualify(t)                       # the real test: does the contract resolve?
            if c:
                px = broker.price(t)                    # informational — may fail if MD session busy
                pxs = f"IB last {px}" if px else "IB price n/a (market-data session busy — OK)"
                print(f"  ✓ {t:9s} -> {sym} {ccy}@{exch or 'SMART'}  qualified; {pxs}")
                ok.append(t)
            else:
                print(f"  ✗ {t:9s} -> {sym} {ccy}@{exch or 'SMART'}  QUALIFY FAILED")
                bad.append(t)
        print(f"\n  qualified {len(ok)}/{len(TEST_TICKERS)}"
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
