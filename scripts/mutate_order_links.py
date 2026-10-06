"""Mutation-test `scripts/test_order_links.py`: seed real faults, demand the suite catches them.

WHY. The order -> IB-record links (orderRef tag, execution ids) fail SILENTLY: an untagged order
still fills, a dropped execution id still books. The first version of test_order_links.py handed
the broker its tag and the ledger its ids by hand, so deleting the runner's one tagging line, or
dropping the ids on the way into the ledger, survived with the suite green (measured 2026-10-06).
A passing suite is only evidence if breaking each link on purpose makes it fail.

Never edits the real files: the repo's code (no data, results, .env or .git) is copied to a temp
dir once and each mutant is written there, so an interrupted run cannot leave a broken file for
the scheduled task.

Non-zero exit if any fault survives, a pattern no longer matches exactly once (the code moved:
update this file), or a mutant does not compile (it would be "caught" by the SyntaxError alone).

Run: python scripts/mutate_order_links.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = [sys.executable, "scripts/test_order_links.py"]
IGNORE = shutil.ignore_patterns(".git", "data", "results", "__pycache__", ".idea", ".claude",
                                ".env", ".venv", "venv", "*.parquet", "*.pkl")

# (file, find, replace, what the fault means)
MUTATIONS = [
    ('paper/broker.py',
     '            order.orderRef = ref\n',
     '            pass\n',
     'broker never stamps orderRef on the order'),
    ('paper/broker.py',
     'ref = getattr(self, "order_ref", None)',
     'ref = self.order_ref',
     'a broker without a tag raises (a missing tag BLOCKS the trade)'),
    ('paper/broker.py',
     'while not getattr(trade, "fills", None) and waited < wait:',
     'while not getattr(trade, "fills", None) and waited < 0:',
     "no wait for execution details that trail 'Filled'"),
    ('paper/broker.py',
     'return [f.execution.execId for f in (getattr(trade, "fills", None) or [])]',
     'return []',
     'execution ids never collected'),
    ('paper/broker.py',
     'return {"ok": True, "status": st, "fill_price": float(fill) if fill else None,\n                        "exec_ids": ex, "order_ref": order.orderRef}',
     'return {"ok": True, "status": st, "fill_price": float(fill) if fill else None,\n                        "exec_ids": [], "order_ref": order.orderRef}',
     'a normal fill drops its execution ids'),
    ('paper/broker.py',
     'return {"ok": True, "status": st, "fill_price": float(fill) if fill else None,\n                        "exec_ids": ex, "order_ref": order.orderRef}',
     'return {"ok": True, "status": st, "fill_price": float(fill) if fill else None,\n                        "exec_ids": ex, "order_ref": ""}',
     'a normal fill drops its orderRef'),
    ('paper/broker.py',
     '                            "exec_ids": ex, "order_ref": order.orderRef}\n                st = trade',
     '                            "exec_ids": [], "order_ref": order.orderRef}\n                st = trade',
     'a cancel-race fill drops its execution ids'),
    ('paper/broker.py',
     'order = self._tag(MarketOrder(action, qty))',
     'order = MarketOrder(action, qty)',
     'FX sweep orders go out untagged'),
    ('paper/broker.py',
     '"filled": float(amount_ccy), "exec_ids": ex}',
     '"filled": float(amount_ccy), "exec_ids": []}',
     'FX sweep drops its execution ids'),
    ('scripts/run_paper.py',
     '    b.order_ref = ORDER_REF\n',
     '',
     'RUNNER: make_broker never sets the tag'),
    ('scripts/run_paper.py',
     '    broker = make_broker(host=',
     '    broker = Broker(host=',
     'RUNNER: main() bypasses make_broker (untagged broker)'),
    ('scripts/run_paper.py',
     'ORDER_REF = f"magic-formula:{RUN_ID}"',
     'ORDER_REF = f"magic:{RUN_ID}"',
     'RUNNER: tag names the wrong strategy'),
    ('paper/orchestrator.py',
     'entry_order_ref=res.get("order_ref", ""),',
     'entry_order_ref="",',
     "orchestrator drops a BUY's orderRef"),
    ('paper/orchestrator.py',
     'entry_exec_ids=list(res.get("exec_ids") or [])))',
     'entry_exec_ids=[]))',
     "orchestrator drops a BUY's execution ids"),
    ('paper/orchestrator.py',
     '                                     order_ref=res.get("order_ref", ""),',
     '                                     order_ref="",',
     "orchestrator drops a SELL's orderRef"),
    ('paper/orchestrator.py',
     'exec_ids=res.get("exec_ids"))',
     'exec_ids=None)',
     "orchestrator drops a SELL's execution ids"),
    ('paper/state.py',
     '"entry_order_ref": pos.entry_order_ref, "entry_exec_ids": list(pos.entry_exec_ids),',
     '"entry_order_ref": "", "entry_exec_ids": [],',
     'closed trade loses the ENTRY links'),
    ('paper/state.py',
     '"exit_order_ref": order_ref, "exit_exec_ids": list(exec_ids or [])}',
     '"exit_order_ref": order_ref, "exit_exec_ids": []}',
     'closed trade loses the EXIT execution ids'),
    ('paper/state.py',
     'entry_exec_ids: list[str] = field(default_factory=list)',
     'entry_exec_ids: list[str] = None',
     'positions saved before the fields existed load with None ids'),
    ('paper/state.py',
     'd["positions"] = [Position(**x) for x in d.get("positions", [])]',
     'd["positions"] = [Position(**{k: v for k, v in x.items() if "order_ref" not in k and "exec_ids" not in k}) for x in d.get("positions", [])]',
     'load() drops the links (save/load round trip)'),
]


def main() -> int:
    base = subprocess.run(SUITE, cwd=ROOT, capture_output=True, text=True)
    if base.returncode != 0:
        print("the suite does not pass on the ORIGINAL code — fix that first")
        print(base.stdout[-2000:])
        return 1
    print("=" * 100)
    print(f"  {len(MUTATIONS)} seeded faults; every one must be CAUGHT\n")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT, work, ignore=IGNORE)
        for rel, find, repl, why in MUTATIONS:
            f = work / rel
            orig = f.read_text(encoding="utf-8")
            n = orig.count(find)
            if n != 1:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} PATTERN {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}")
                continue
            mutant = orig.replace(find, repl, 1)
            try:                    # a mutant that does not even compile is not a test of anything
                compile(mutant, rel, "exec")
            except SyntaxError as e:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} INVALID MUTANT ({e.msg})")
                continue
            f.write_text(mutant, encoding="utf-8")
            try:
                r = subprocess.run(SUITE, cwd=work, capture_output=True, text=True)
            finally:
                f.write_text(orig, encoding="utf-8")
            caught = r.returncode != 0
            results.append((why, caught))
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:80} "
                  f"{'CAUGHT' if caught else '*** SURVIVED ***'}")
    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 100)
    if missing:
        print(f"{len(missing)} mutation(s) could not be applied (pattern moved or mutant invalid) "
              "— update this file:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
    if survived or missing:
        return 1
    print(f"all {len(MUTATIONS)} seeded faults were caught; the real files were never modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
