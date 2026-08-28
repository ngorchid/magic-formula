"""Is fundamentals coverage the blocker for widening the European universe?

THE QUESTION. The live magic formula draws its EU leg from five national indices (DAX,
CAC 40, AEX, IBEX 35, FTSE MIB). Widening it raises two options:
  (a) the five EUROZONE exchanges the broker already supports but the universe never
      scrapes -- .BR .HE .IR .LS .VI -- at zero incremental FX exposure, or
  (b) non-EUR Europe (.L .SW .ST .CO .OL), which adds five currencies.

Recollection was that European fundamentals are often unavailable, which would make the
question moot. That is mechanically plausible -- live fundamentals come from yfinance
LABEL MATCHING, not EDGAR, and `combine_ranks` intersects validity masks so a name missing
ANY required item drops out of the ranking entirely -- but it had never been measured.

WHAT THIS MEASURES. Exactly the live extraction path: `tk.financials / balance_sheet /
cashflow`, resolved through `ITEM_LABELS` with `_first_row` (first matching label wins).
A name counts as USABLE only if every item its live factor set needs is present.

Graham is OFF in production (`run_paper.py` sets `use_graham=False`), so `net_income` and
`total_equity` are NOT required -- they feed only the Graham family. That leaves 10
required items, and measuring against 12 would understate coverage.

Run: python scripts/eu_fundamentals_coverage.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.live_data import ITEM_LABELS, _ABS_ITEMS, _first_row  # noqa: E402

# Items the DEPLOYED factor set needs (Graham off).
#   fcf_ev_yield          -> ocf, capex, shares_diluted, st_debt, lt_debt, cash
#   fcf_return_on_capital -> ocf, capex, curr_assets, cash, curr_liabs, st_debt, ppe_net
#   revenue_growth        -> revenue
#   fcf_growth            -> ocf, capex
#   residual_momentum     -> prices only
REQUIRED = ["revenue", "operating_cash_flow", "capex", "shares_diluted",
            "short_term_debt", "long_term_debt", "cash",
            "total_current_assets", "total_current_liabilities", "ppe_net"]
GRAHAM_ONLY = ["net_income", "total_equity"]

SAMPLE: dict[str, list[str]] = {
    "EUR: current 5 indices": ["SAP.DE", "ASML.AS", "MC.PA", "SAN.MC", "ENI.MI", "AIR.PA"],
    "EUR: Brussels (.BR)":    ["ABI.BR", "KBC.BR", "UCB.BR", "SOLB.BR"],
    "EUR: Helsinki (.HE)":    ["NOKIA.HE", "SAMPO.HE", "NESTE.HE", "KNEBV.HE"],
    "EUR: Dublin (.IR)":      ["RYA.IR", "KRZ.IR", "BIRG.IR"],
    "EUR: Lisbon (.LS)":      ["GALP.LS", "EDP.LS", "JMT.LS"],
    "EUR: Vienna (.VI)":      ["OMV.VI", "EBS.VI", "VER.VI"],
    "non-EUR: UK (.L)":       ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L"],
    "non-EUR: Swiss (.SW)":   ["NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW"],
    "non-EUR: Nordic":        ["VOLV-B.ST", "ERIC-B.ST", "NOVO-B.CO", "EQNR.OL"],
    "US control":             ["AAPL", "JPM", "XOM"],
}


def probe(ticker: str) -> dict:
    """Replicate the live extraction and report which items resolve."""
    out = {"ticker": ticker, "ok": False, "n_req": 0, "n_graham": 0,
           "missing": [], "err": ""}
    try:
        tk = yf.Ticker(ticker)
        inc, bs, cf = tk.financials, tk.balance_sheet, tk.cashflow
        found = set()
        for it, labels in ITEM_LABELS.items():
            src = cf if it in ("operating_cash_flow", "capex") else \
                  inc if it in ("revenue", "net_income", "shares_diluted") else bs
            s = _first_row(src, labels)
            if s is not None and len(s):
                found.add(it)
        out["n_req"] = sum(i in found for i in REQUIRED)
        out["n_graham"] = sum(i in found for i in GRAHAM_ONLY)
        out["missing"] = [i for i in REQUIRED if i not in found]
        out["ok"] = out["n_req"] == len(REQUIRED)
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}"
    return out


def main() -> None:
    n = sum(len(v) for v in SAMPLE.values())
    print(f"probing {n} names against the live yfinance extraction path "
          f"({len(REQUIRED)} required items, Graham off)\n")
    rows = []
    for group, tickers in SAMPLE.items():
        for t in tickers:
            r = probe(t)
            r["group"] = group
            rows.append(r)
            flag = "OK " if r["ok"] else "-- "
            miss = ",".join(r["missing"][:4]) + ("…" if len(r["missing"]) > 4 else "")
            print(f"  {flag}{t:12s} {r['n_req']:2d}/{len(REQUIRED)} req  "
                  f"{r['n_graham']}/2 graham  {r['err'] or miss}")
    df = pd.DataFrame(rows)

    print("\n" + "=" * 88)
    print("COVERAGE BY REGION — a name is USABLE only if all 10 required items resolve")
    print("=" * 88)
    print(f"  {'group':26s} {'n':>3s} {'usable':>7s} {'rate':>7s} {'mean req':>9s} {'graham':>8s}")
    for group in SAMPLE:
        g = df[df["group"] == group]
        if not len(g):
            continue
        print(f"  {group:26s} {len(g):3d} {int(g['ok'].sum()):7d} {g['ok'].mean():7.0%} "
              f"{g['n_req'].mean():8.1f}/10 {g['n_graham'].mean():7.1f}/2")

    print("\n  most-missed required items:")
    miss = pd.Series([m for ms in df["missing"] for m in ms]).value_counts()
    for item, c in miss.head(6).items():
        print(f"    {item:28s} missing on {c} of {len(df)} names")

    eur_new = df[df["group"].str.startswith("EUR:") & ~df["group"].str.contains("current")]
    non_eur = df[df["group"].str.startswith("non-EUR")]
    cur = df[df["group"].str.contains("current")]
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    for lbl, g in (("current 5 indices", cur), ("(a) new EUROZONE exchanges", eur_new),
                   ("(b) non-EUR Europe", non_eur)):
        if len(g):
            print(f"  {lbl:28s} usable {g['ok'].mean():5.0%}  ({int(g['ok'].sum())}/{len(g)})")
    print("\n  If (a) matches the current indices, fundamentals are NOT the blocker and the")
    print("  five missing eurozone exchanges can be added on their merits.")

    out = ROOT / "results" / "eu_coverage"
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["missing"]).to_csv(out / "coverage.csv", index=False)
    print(f"\n  wrote {out}/coverage.csv")


if __name__ == "__main__":
    main()
