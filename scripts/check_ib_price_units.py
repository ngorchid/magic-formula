"""Does IB quote LSE stocks in the same units as yfinance? Answer it without trading.

THE QUESTION. Position sizing divides a USD slot by the yfinance mark, but the position is
recorded at the IB fill price. Those are only comparable if both are denominated identically,
and for London that is genuinely in doubt: the LSE quotes in PENCE while the IB contract
currency reads GBP. IB ships `ContractDetails.priceMagnifier` specifically to reconcile
execution prices with market data -- "allows execution and strike prices to be reported
consistently with market data, historical data and the order price" -- so the discrepancy is
real enough to have a dedicated field. But priceMagnifier only makes IB self-consistent. It
says nothing about agreeing with yfinance, which is the comparison that decides whether our
sizing is right.

WHY IT MATTERS. If IB reports pounds while the mark is pence, `cost_usd` is 100x too small.
Cash barely moves, the sizer thinks it still has budget, and it keeps buying. Nothing raises.

NO ORDER IS PLACED. A historical bar answers the units question exactly as well as a fill
does, with no capital at risk. This script only reads.

Run it wherever TWS/IB Gateway is reachable (the Windows box):
    python3 scripts/check_ib_price_units.py

Reads IB_HOST / IB_PORT from .env. Uses a distinct client id so it cannot collide with the
live runners (magic-formula 5, trend 6, options-vrp 7, treasury 8).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# One name per venue that the European universe can now select from.
PROBES = ["SHEL.L", "AZN.L", "HSBA.L",          # pence — the case under test
          "NESN.SW", "VOLV-B.ST", "NOVO-B.CO", "EQNR.OL",   # non-EUR majors
          "SAP.DE", "ABI.BR", "RYA.IR"]                     # EUR control
TOL = 5.0


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "7497"))

    import yfinance as yf
    from paper.broker import Broker, ib_contract_spec

    br = Broker(host=host, port=port, client_id=17, dry_run=False)
    if not br.connect():
        print(f"could not connect to TWS/Gateway at {host}:{port} — is it running, and is the "
              f"API enabled with 127.0.0.1 trusted?")
        return 2

    print(f"\n{'ticker':11s} {'IB ccy':7s} {'magnif':>7s} {'IB price':>11s} {'yfinance':>11s} "
          f"{'ratio':>8s}  verdict")
    bad = 0
    for t in PROBES:
        _, ccy, _ = ib_contract_spec(t)
        try:
            fi = yf.Ticker(t).fast_info
            yf_px, yf_ccy = fi.get("lastPrice"), fi.get("currency")
        except Exception:
            yf_px, yf_ccy = None, None
        ib_px = br.price(t)
        # priceMagnifier is on ContractDetails, not the Contract — fetch it for the record.
        mag = ""
        try:
            c = br.qualify(t)
            if c is not None:
                cd = br.ib.reqContractDetails(c)
                if cd:
                    mag = str(getattr(cd[0], "priceMagnifier", ""))
        except Exception:
            pass
        if not ib_px or not yf_px:
            print(f"  {t:9s} {ccy:7s} {mag:>7s} {str(ib_px):>11s} {str(yf_px):>11s} "
                  f"{'—':>8s}  no data")
            continue
        r = ib_px / yf_px
        ok = (1 / TOL) <= r <= TOL
        if not ok:
            bad += 1
        note = "OK" if ok else ("IB in MAJOR units, mark in MINOR" if r < 1
                                else "IB in MINOR units, mark in MAJOR")
        print(f"  {t:9s} {ccy:7s} {mag:>7s} {ib_px:11.2f} {yf_px:11.2f} {r:8.4f}  "
              f"{note}{'' if ok else '  <-- 100x RISK'}  (yf ccy {yf_ccy})")

    print("\n" + "=" * 92)
    if bad:
        print(f"{bad} name(s) DISAGREE beyond {TOL:.0f}x. Sizing uses the yfinance mark and")
        print("records the IB fill, so those two must match. Do NOT enable those venues live")
        print("until the broker price is normalised (divide by priceMagnifier).")
    else:
        print(f"All probed names agree within {TOL:.0f}x — IB and yfinance use the same units,")
        print("so entry_price and the mark are directly comparable. The pre-trade guard")
        print("`price_units_agree` in paper/orchestrator.py stays as the standing safety net.")
    br.disconnect()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
