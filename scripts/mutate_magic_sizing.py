"""Mutation-test `scripts/test_magic_sizing.py`: seed real faults, demand the suite catches them.

WHY. A test suite that cannot fail is worse than none, because it converts absence of evidence
into apparent evidence. On 2026-08-14 four assertions in `test_risk_guard.py` were found to have
been vacuous since they were written — including the gross-cap check — because they were handed
bare `type("C", (), {...})()` objects, which define no `__bool__` and are therefore always
truthy. Nothing in the ordinary run of the suite could have revealed that. Mutation testing can:
break the code on purpose and see whether anything notices.

Each entry below is a fault we either HAVE shipped or specifically fear. Every one must be
CAUGHT. A surviving mutation means the corresponding case is decoration, and the exit status is
non-zero so this can gate a release.

Run: python3 scripts/mutate_magic_sizing.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "paper" / "orchestrator.py"
SUITE = ["python3", "scripts/test_magic_sizing.py"]

# (find, replace, description). Ordered roughly by the three historical regressions first.
MUTATIONS = [
    # --- REGRESSION 1: tilts no longer average 1, so gross drifts off budget ---
    ("    return t / m if (m and m == m and m > 0) else pd.Series(1.0, index=vol.index)",
     "    return t",
     "tilts NOT normalised (gross drifts off budget)"),
    ("    m = float(np.nanmean(t.values))",
     "    m = 1.0",
     "tilt normaliser hard-coded to 1 (same effect, subtler)"),

    # --- REGRESSION 2: cash cap applied BEFORE the tilt, so a high tilt overdraws ---
    ("    target_usd = min(slot_usd * tilt, max(float(cash_usd), 0.0))",
     "    target_usd = min(slot_usd, max(float(cash_usd), 0.0)) * tilt",
     "cash cap applied BEFORE the tilt (overdraws; cash went to -$2,370)"),
    ("    target_usd = min(slot_usd * tilt, max(float(cash_usd), 0.0))",
     "    target_usd = slot_usd * tilt",
     "cash cap removed entirely"),
    ("    if target_usd <= 0 or not denom or denom <= 0 or denom != denom:",
     "    if not denom or denom <= 0 or denom != denom:",
     "non-positive guard removed (emits a BUY for negative shares)"),

    # --- REGRESSION 3: a full book aims the whole idle gap at ONE order ---
    ("    return min(slot, cfg.max_entry_mult * equal_weight)",
     "    return slot",
     "entry cap removed (full book -> ~16% of NAV in one order)"),
    ("    equal_weight = nav * gross_scalar / max(cfg.top_n, 1)",
     "    equal_weight = nav * gross_scalar",
     "equal-weight computed without dividing by top_n (cap never binds)"),

    # --- other sizing invariants ---
    ("    remaining = max(nav * gross_scalar - invested, 0.0)",
     "    remaining = nav * gross_scalar - invested",
     "remaining not floored at 0 (negative slot when over-invested)"),
    ("    nav = state.nav(marks, fx)",
     "    nav = cfg.budget",
     "slot sized off cfg.budget instead of NAV (tops up with money not held)"),
    ("    return float(np.clip(cfg.vol_target / est_book_vol, 0.0, 1.0))",
     "    return float(cfg.vol_target / est_book_vol)",
     "gross_scalar unclipped (permits leverage > 1)"),
    ("    return int(target_usd // denom)",
     "    return int(-(-target_usd // denom))",
     "shares rounded UP instead of down (overdraws by a fraction of a share)"),
    ("    t = pd.Series(np.clip(ref / vol, *cfg.inv_vol_clip), index=vol.index).replace(",
     "    t = pd.Series(ref / vol, index=vol.index).replace(",
     "inverse-vol clip removed (one low-vol name can dominate)"),

    # --- stale prices (wired 2026-08-16) ---
    ("            if t in stale_px:",
     "            if False:",
     "stale-price skip removed (buys size off a frozen price)"),
    ("    stale_px, _ = stale_columns(adj, pd.Timestamp(today),",
     "    stale_px, _ = ({}, None) or stale_columns(adj, pd.Timestamp(today),",
     "stale detection returns nothing (every name looks live)"),
]


# --- FX CASH SWEEP (added 2026-08-28) ---------------------------------------------------
# The sweep PLACES ORDERS, so a fault here spends money rather than mis-reporting. The
# direction mutations matter most: an inverted action does not fail loudly, it doubles the
# exposure it was meant to close.
BROKER = ROOT / "paper" / "broker.py"
MUTATIONS += [
    ("        plan[ccy] = -bal          # trade the negative of the balance to reach zero",
     "        plan[ccy] = bal",
     "sweep direction inverted (DOUBLES the balance instead of closing it)"),
    ("        if abs(bal * rate) < min_usd:",
     "        if abs(bal) < min_usd:",
     "threshold on UNITS not USD (never sweeps SEK/NOK; sweeps tiny GBP)"),
    ("        if not rate or rate <= 0 or not np.isfinite(rate):",
     "        if False:",
     "missing/NaN rate no longer skipped (sizes an order off a bad rate)"),
    ("        if ccy == \"USD\" or not bal:",
     "        if not bal:",
     "USD itself swept (would trade USDUSD / corrupt the plan)"),
    ("        action, qty = (\"BUY\" if amount_ccy > 0 else \"SELL\"), abs(amount_ccy)",
     "        action, qty = (\"SELL\" if amount_ccy > 0 else \"BUY\"), abs(amount_ccy)",
     "ccy-as-base direction inverted (EUR/GBP traded the wrong way)", BROKER),
    ("        action = \"SELL\" if amount_ccy > 0 else \"BUY\"",
     "        action = \"BUY\" if amount_ccy > 0 else \"SELL\"",
     "USD-as-base direction NOT inverted (CHF/SEK/DKK/NOK wrong way)", BROKER),
    ("        qty = abs(amount_ccy) * (rate_usd or 0.0)",
     "        qty = abs(amount_ccy)",
     "USD-base order sized in foreign units, not USD (SEK ~10x too large)", BROKER),
]


def main() -> int:
    originals = {f: f.read_text() for f in {TARGET, BROKER}}
    results = []
    print("=" * 92)
    print(f"MUTATION TEST — orchestrator.py + broker.py against {SUITE[1]}")
    print("=" * 92)
    print(f"  {len(MUTATIONS)} seeded faults; every one must be CAUGHT\n")
    try:
        for mut in MUTATIONS:
            find, repl, why = mut[0], mut[1], mut[2]
            tgt = mut[3] if len(mut) > 3 else TARGET
            base = originals[tgt]
            if find not in base:
                results.append((why, None))
                print(f"  [ ?? ] {why:70} PATTERN MISSING")
                continue
            tgt.write_text(base.replace(find, repl, 1))
            for pyc in ROOT.rglob("*.pyc"):
                pyc.unlink(missing_ok=True)
            r = subprocess.run(SUITE, cwd=ROOT, capture_output=True, text=True)
            caught = r.returncode != 0
            results.append((why, caught))
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:70} "
                  f"{'CAUGHT' if caught else '*** SURVIVED ***'}")
    finally:
        for f, text in originals.items():
            f.write_text(text)
        for pyc in ROOT.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)

    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 92)
    if missing:
        print(f"{len(missing)} mutation(s) could not be applied — the code moved, update this file:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
        return 1
    if missing:
        return 1
    # Restoring must leave the suite green, or the harness itself corrupted the file.
    r = subprocess.run(SUITE, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("RESTORE FAILED — the suite does not pass on the original file")
        return 1
    print(f"all {len(MUTATIONS)} seeded faults were caught; suite restored and green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
