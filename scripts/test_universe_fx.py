"""Tests for the European universe scrape and the FX conversion that sizes it.

WHY THIS EXISTS. Widening the universe to all of Europe on 2026-08-28 introduced two failure
modes that are dangerous precisely because they are SILENT — neither raises, and both produce
a plausible-looking book:

  1. A MIS-NORMALISED TICKER simply fails to download, and `fetch_live_panels` drops any
     ticker with no data. So a broken normaliser does not error; the venue just quietly
     contributes nothing. Six of ten indices were in exactly that state before this change
     (Wikipedia publishes "Euronext Brussels:\xa0ABI", not "ABI.BR") and the only symptom was
     a ticker count nobody was checking.

  2. THE PENCE TRAP. LSE equities report currency "GBp" and quote in pence. Yahoo's FX
     endpoint is case-insensitive, so `GBpUSD=X` returns the POUND rate instead of failing —
     measured that day, GBp and GBP both came back 1.3528. Market caps for every UK name
     would be 100x too large, and mcap is not cosmetic here: it feeds the size floor and
     eligibility, so all ~100 FTSE names would clear the large-cap filter and swamp the rank.
     A silent 100x is far worse than a NaN, which would at least drop the name.

These are pure functions over fixed inputs, so they are tested offline with no network. The
resolution rate against live yfinance (454/461 = 98% on 2026-08-28) is a separate, networked
check — see the docstring in data/universe.py.

Run: python3 scripts/test_universe_fx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.universe import (EURO_INDEX_PAGES, NON_EUR_INDEX_PAGES,  # noqa: E402
                           _ALL_SUFFIXES, _normalise_ticker)
from paper.live_data import _MINOR_UNITS  # noqa: E402

fails, ran = [], 0


def expect(label: str, ok: bool, detail: str = "") -> None:
    global ran
    ran += 1
    if not ok:
        fails.append(f"{label}: {detail}")
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:66}{'' if ok else '  | ' + detail}")


def eq(label: str, got, want) -> None:
    expect(label, got == want, f"got {got!r}, want {want!r}")


print("=" * 92)
print("_normalise_ticker — the three cell shapes actually observed on Wikipedia")
print("=" * 92)

# Exchange prefix + non-breaking space. This is the shape that silently killed Brussels,
# Dublin, Lisbon and Oslo: the old code kept only cells ALREADY ending in a venue suffix.
eq("Euronext prefix + nbsp -> stripped", _normalise_ticker("Euronext Brussels:\xa0ABI", ".BR"), "ABI.BR")
eq("Euronext Dublin prefix", _normalise_ticker("Euronext Dublin:\xa0A5G", ".IR"), "A5G.IR")
eq("OSE prefix, ordinary space", _normalise_ticker("OSE: AKRBP", ".OL"), "AKRBP.OL")
eq("bare symbol gets its venue suffix", _normalise_ticker("NOVN", ".SW"), "NOVN.SW")

# Share classes. Two different separators, one target form.
eq("Nordic share class: space -> hyphen", _normalise_ticker("MAERSK B", ".CO"), "MAERSK-B.CO")
eq("UK share class: dot -> hyphen", _normalise_ticker("BT.A", ".L"), "BT-A.L")

# REGRESSION: the dot rule must not eat the venue suffix of an already-formed ticker, or the
# original five indices (which DO publish yfinance form) would all be corrupted to "ADS-DE".
eq("already-suffixed ticker untouched", _normalise_ticker("ADS.DE", ".DE"), "ADS.DE")
eq("already-suffixed, different venue col", _normalise_ticker("ABN.AS", ".AS"), "ABN.AS")

# Junk rejection: a misaligned or merged cell must yield None, not a fake ticker.
eq("empty cell rejected", _normalise_ticker("", ".L"), None)
eq("literal nan rejected", _normalise_ticker("nan", ".L"), None)
eq("prose rejected", _normalise_ticker("Some long company name", ".L"), None)
eq("footnote marker rejected", _normalise_ticker("see note [4]", ".L"), None)

print()
print("=" * 92)
print("index page config")
print("=" * 92)

overlap = set(EURO_INDEX_PAGES) & set(NON_EUR_INDEX_PAGES)
expect("eurozone and non-eurozone index sets are disjoint", not overlap, str(overlap))

eur_sfx = {s for _, s in EURO_INDEX_PAGES.values()}
non_sfx = {s for _, s in NON_EUR_INDEX_PAGES.values()}
expect("suffix sets are disjoint", not (eur_sfx & non_sfx), str(eur_sfx & non_sfx))
expect("every configured suffix is in _ALL_SUFFIXES",
       (eur_sfx | non_sfx) <= set(_ALL_SUFFIXES), str((eur_sfx | non_sfx) - set(_ALL_SUFFIXES)))
for pages in (EURO_INDEX_PAGES, NON_EUR_INDEX_PAGES):
    for name, (url, sfx) in pages.items():
        expect(f"{name}: (url, suffix) well-formed",
               url.startswith("https://") and sfx.startswith(".") and len(sfx) >= 2,
               f"{url!r}, {sfx!r}")

# Vienna is absent ON PURPOSE (no ticker column on either Wikipedia edition). If someone adds
# a .VI page later this fails, which is the prompt to confirm the source really has tickers.
expect("Vienna (.VI) still deliberately excluded", ".VI" not in (eur_sfx | non_sfx))

print()
print("=" * 92)
print("FX minor units — the pence trap")
print("=" * 92)

eq("GBp maps to GBP / 100", _MINOR_UNITS.get("GBp"), ("GBP", 100.0))
expect("GBP itself is NOT in the minor-unit table", "GBP" not in _MINOR_UNITS,
       "would divide the major rate by 100 twice over")

# The conversion arithmetic, with a stubbed rate so this stays offline and deterministic.
GBPUSD = 1.3528
major, div = _MINOR_UNITS["GBp"]
fx_pence = GBPUSD / div
expect("pence factor is exactly 1/100 of the pound factor",
       abs(GBPUSD / fx_pence - 100.0) < 1e-9, f"ratio {GBPUSD / fx_pence}")

# SHEL.L quoted 3344.5 pence on 2026-08-28 = GBP 33.445 ~ USD 45.
usd = 3344.5 * fx_pence
expect("SHEL.L 3344.5p converts to a sane per-share USD value",
       40.0 < usd < 50.0, f"got {usd:.2f} (the bug gives ~{3344.5 * GBPUSD:.0f})")

# The bug's signature, asserted directly: treating pence as pounds inflates by exactly 100x.
expect("mis-handling GBp would inflate market cap exactly 100x",
       abs((3344.5 * GBPUSD) / usd - 100.0) < 1e-9)

# Majors must pass through untouched — a currency not in the table divides by 1.0.
for ccy in ("EUR", "CHF", "SEK", "DKK", "NOK", "USD"):
    m, d = _MINOR_UNITS.get(ccy, (ccy, 1.0))
    expect(f"{ccy} passes through unscaled", m == ccy and d == 1.0, f"{m}, {d}")

print()
print("=" * 92)
print("marketCap uses the MAJOR-unit rate — the subtle half of the pence fix")
print("=" * 92)

# Verified against live data 2026-08-28: marketCap / (price x shares) was exactly 0.0100 for
# all four UK names and exactly 1.0000 for every non-pence venue. So `marketCap` is in POUNDS
# while `price` is in PENCE, and the two need rates that differ by the subdivision. Applying
# the quoted rate to marketCap understates every UK company by a factor of 100.
def mcap_rate(ccy: str, quoted: float) -> float:
    """Mirror of the live derivation: major-unit rate = quoted rate x subdivision."""
    return quoted * _MINOR_UNITS.get(ccy, (ccy, 1.0))[1]


GBPUSD = 1.3533
gbp_quoted = GBPUSD / 100.0                      # what _fx_to_usd returns for "GBp"
expect("GBp marketCap rate is the POUND rate, not the pence rate",
       abs(mcap_rate("GBp", gbp_quoted) - GBPUSD) < 1e-12,
       f"got {mcap_rate('GBp', gbp_quoted)}, want {GBPUSD}")

# SHEL.L: marketCap 184.11bn (pounds) -> ~249bn USD. The bug gives ~1.84bn.
shel = 184_111_431_680 * mcap_rate("GBp", gbp_quoted) / 1e9
expect("SHEL.L marketCap converts to ~249bn USD", 240 < shel < 260, f"got {shel:.1f}bn")
expect("using the quoted rate instead would understate SHEL.L 100x",
       abs(shel / (184_111_431_680 * gbp_quoted / 1e9) - 100.0) < 1e-9)

# A non-minor currency must be unaffected by the major-unit step, or every EUR/NOK/CHF name
# would break in exchange for fixing the UK ones.
for ccy, rate in (("EUR", 1.1586), ("NOK", 0.1066), ("USD", 1.0)):
    expect(f"{ccy} marketCap rate equals its quoted rate",
           abs(mcap_rate(ccy, rate) - rate) < 1e-12)

# EQNR.OL: quoted NOK, reports USD. Tagging it USD (the old behaviour) left 917bn NOK
# untouched and reported a ~$917bn company; the real figure is ~$98bn.
eqnr_nok = 917_068_644_352
expect("EQNR.OL 917bn NOK converts to ~98bn USD",
       90 < eqnr_nok * 0.1066 / 1e9 < 105, f"got {eqnr_nok * 0.1066 / 1e9:.1f}bn")
expect("the old financialCurrency tag would have overstated EQNR.OL ~9.4x",
       abs(1.0 / 0.1066 - 9.38) < 0.05)


print()
print("=" * 92)
print("IB SYMBOL CANDIDATES — yfinance and IB disagree on share-class separators")
print("=" * 92)

# Measured against live IB 2026-08-30 by sweeping all 32 hyphenated European names: IB uses a
# DOT for the share-class separator (VOLV.B), 29 of 32; a SPACE for one (NDA-FI -> "NDA FI") and
# NONE for two (NDA-SE -> NDASE, ROCK-B -> ROCKB). It never uses the hyphen yfinance writes. The
# old candidate list omitted the dot entirely, so every share class failed SILENTLY — qualify()
# returned None, the order was refused, and the name sat in the universe unable to ever trade.
from paper.broker import IB_SYMBOL_FIX, ib_symbol_candidates  # noqa: E402

_c = ib_symbol_candidates("VOLV-B.ST")
expect("hyphen ticker offers the DOT form (IB's measured convention)", "VOLV.B" in _c, str(_c))
expect("...the DOT form comes FIRST, so it resolves on the first probe", _c[0] == "VOLV.B", str(_c))
expect("...still offers the SPACE form (NDA-FI needs it)", "VOLV B" in _c, str(_c))
expect("...and the concatenated form (NDA-SE, ROCK-B need it)", "VOLVB" in _c, str(_c))
expect("...the raw hyphen form is still tried (never IB's, but harmless)", "VOLV-B" in _c, str(_c))
expect("suffix is stripped before generating candidates",
       all(".ST" not in x for x in _c), str(_c))

# A plain ticker must not generate junk alternatives — every extra candidate is a wasted IB
# round trip on the monthly pull over ~960 names.
eq("plain ticker yields exactly one candidate", ib_symbol_candidates("SAP.DE"), ["SAP"])
eq("US ticker yields exactly one candidate", ib_symbol_candidates("AAPL"), ["AAPL"])
expect("no duplicates when the forms coincide",
       len(ib_symbol_candidates("BT-A.L")) == len(set(ib_symbol_candidates("BT-A.L"))),
       str(ib_symbol_candidates("BT-A.L")))

# IB_SYMBOL_FIX must take precedence — it exists for tickers that DIFFER (Dublin lists Ryanair
# as RY4C, yfinance as RYA), which no formatting rule can derive.
import paper.broker as _b  # noqa: E402
_b.IB_SYMBOL_FIX["RYA.IR"] = "RY4C"
try:
    expect("IB_SYMBOL_FIX overrides and comes FIRST",
           ib_symbol_candidates("RYA.IR")[0] == "RY4C", str(ib_symbol_candidates("RYA.IR")))
finally:
    _b.IB_SYMBOL_FIX.pop("RYA.IR", None)
expect("IB_SYMBOL_FIX ships EMPTY (entries must be verified against live IB first)",
       IB_SYMBOL_FIX == {}, str(IB_SYMBOL_FIX))

# Euronext Dublin dropped 2026-08-30: 0/20 Irish names qualified at IB (no Dublin permission).
# Guard against a silent re-add — every .IR name would re-enter the universe and fail every run,
# re-feeding the "universe shrinking" alerts this venue was removed to stop.
expect("ISEQ 20 is OUT of EURO_INDEX_PAGES", "ISEQ 20" not in EURO_INDEX_PAGES,
       str(list(EURO_INDEX_PAGES)))
expect("the .IR suffix is no longer in the scraped universe", ".IR" not in _ALL_SUFFIXES,
       str(_ALL_SUFFIXES))

print("\n" + "=" * 92)
if fails:
    print(f"{len(fails)} FAILURE(S) of {ran}:")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print(f"all {ran} universe/FX checks behaved as expected")
