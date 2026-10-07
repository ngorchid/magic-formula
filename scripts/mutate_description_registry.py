"""Mutation-test `scripts/test_description_registry.py`: the unseen-description registry.

Seeds real faults into a TEMP COPY of download_ib_records.py (via IB_RECORDS_MODULE; the real
script is never edited) and demands the suite catches every one. Engine: _mutate_records_core.py.

Run: python scripts/mutate_description_registry.py
"""
from __future__ import annotations

import sys

from _mutate_records_core import run

MUTATIONS = [
    ('    warn += check_descriptions()\n',
     '',
     'registry never consulted (new descriptions silent again)'),
    ('        if n in reg:\n            continue\n',
     '',
     'known descriptions re-alerted every day'),
    ('        reg[n] = {"first_seen": d, "example": raw, "sleeve": sl}\n    reg_file',
     '    reg_file',
     'a new description is never remembered (alerts daily)'),
    ('        if not marker.exists():\n',
     '        if False:\n',
     'the very first seed alerts (false alarm on deployment)'),
    ('        else:\n            return [f"description registry was MISSING',
     '        elif False:\n            return [f"description registry was MISSING',
     'a lost registry re-seeds silently'),
    ('            marker.write_text(datetime.now().isoformat(timespec="seconds"))',
     '            pass',
     'the seeded-marker is never written (loss never detected)'),
    ('REGISTRY_CATEGORIES = {"fee", "interest", "other", "adjustment"}',
     'REGISTRY_CATEGORIES = {"fee", "interest", "other", "adjustment", "dividend"}',
     'dividends registered as fee/interest lines'),
    ('REGISTRY_CATEGORIES = {"fee", "interest", "other", "adjustment"}',
     'REGISTRY_CATEGORIES = {"fee", "interest", "adjustment"}',
     'unknown row types never reported'),
    ('    s = re.sub(rf"\\b(?:{_MONTHS})[A-Z]*\\.?(?:[-/ ]?\\d{{2,4}})?\\b", " ", s)  # SEP-2026, OCT 26\n',
     '',
     'month names not stripped (every month looks new)'),
    ('    s = re.sub(r"[^A-Z ]", " ", s)                                        # amounts, ids, punct.',
     '    s = re.sub(r"[^A-Z0-9 ]", " ", s)',
     'amounts not stripped'),
    ('    if symbol:\n        s = s.replace(symbol.upper(), " ")\n',
     '',
     'symbols not stripped'),
    ('    s = re.sub(r"\\b[A-Z]{2}[A-Z0-9]{9}\\d\\b", " ", s)                       # ISIN\n',
     '',
     'ISINs not stripped'),
    ('f"NEW fee/interest description, assigned to {sl or \'?\'}: \'{raw}\' — "',
     'f"NEW fee/interest description: \'{raw}\' — "',
     'the alert no longer says where the line went'),
]

if __name__ == "__main__":
    sys.exit(run("test_description_registry.py", MUTATIONS))
