"""Mutation-test `scripts/test_inception_twr.py`: the inception-to-date TWR and its S&P 500 benchmark.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_inception_twr.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

F = "scripts/inception_twr.py"
MUTATIONS = [
    (F, "MAX_DAYS = 365 ", "MAX_DAYS = 366 ", "a window one day over IB's limit"),
    (F, "        e = min(end, s + timedelta(days=MAX_DAYS - 1))",
     "        e = min(end, s + timedelta(days=MAX_DAYS))", "windows too long"),
    (F, "        s = e + timedelta(days=1)", "        s = e + timedelta(days=2)", "a day skipped between windows"),
    (F, "        g *= 1 + r / 100", "        g += r / 100", "segment TWRs added instead of chained"),
    (F, "    if account and accts != {account}:", "    if False:", "another account's statement accepted"),
    (F, "    if len(rows) != 1 or rows[0].get(\"twr\") in (None, \"\"):",
     "    if not rows:", "a statement without a twr accepted"),
    (F, '    return {"fromDate": navs[0]["fromDate"]', '    return {"fromDate": navs[-1]["fromDate"]',
     "the window starts at the last segment"),
    (F, '"endingValue": navs[-1].get("endingValue", "")', '"endingValue": navs[0].get("endingValue", "")',
     "the first segment's ending value"),
    (F, '    a = [c for c in closes if c[0] <= frm]', '    a = [c for c in closes if c[0] >= frm]',
     "the benchmark starts after inception"),
    (F, '    b = [c for c in closes if c[0] <= to]', '    b = closes', "the benchmark runs past IB's toDate"),
    (F, "    if not a or not b or b[-1][0] <= a[-1][0]:", "    if not a or not b:",
     "an empty benchmark window reads as 0%"),
    (F, "    row, data = itd(fetch, _d(INCEPTION), today - timedelta(days=1), account)",
     "    row, data = itd(fetch, _d(INCEPTION), today, account)", "today requested (no statement yet)"),
    (F, "    except Exception as e:  # noqa: BLE001\n        print(f\"benchmark",
     "    except ZeroDivisionError as e:  # noqa: BLE001\n        print(f\"benchmark",
     "a failed benchmark download loses the TWR"),
    (F, '    (out / "latest.xml").write_bytes(data)', "    pass", "the ITD statement is not kept"),
    (F, '    with open(hist, "a", newline=""', '    with open(hist, "w", newline=""', "history overwritten"),
    (F, "        return rows[-1] if rows else None", "        return rows[0] if rows else None",
     "the oldest row is reported"),
    (F, "    except (OSError, csv.Error):\n        return None", "    except csv.Error:\n        return None",
     "a missing history crashes"),
]

if __name__ == "__main__":
    sys.exit(run("test_inception_twr.py", MUTATIONS))
