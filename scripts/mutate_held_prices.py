"""Mutation-test `scripts/test_held_prices.py`: held names that left the universe keep a price.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_held_prices.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

R = "scripts/run_paper.py"
MUTATIONS = [
    (R, '    panels = _ensure_held_prices(panels, {p.ticker: p.currency for p in state.positions})\n',
     '', 'main() never adds the held names (the original zero-valuation bug)'),
    (R, '    missing = sorted(t for t in held if t and t not in adj.columns)',
     '    missing = sorted(t for t in held if t)',
     'every held name is re-downloaded, also those already in the panel'),
    (R, '    missing = sorted(t for t in held if t and t not in adj.columns)',
     '    missing = []', 'nothing is ever treated as missing'),
    (R, '        raw = yf.download(missing, start=adj.index[0], auto_adjust=True, progress=False,',
     '        raw = yf.download(missing, period="5d", auto_adjust=True, progress=False,',
     'only 5 days of history (no vol / staleness history for the added name)'),
    (R, '            adj[t] = px[t].reindex(adj.index)',
     '            adj = adj.reindex(adj.index.union(px.index)); adj[t] = px[t]',
     'download dates added as panel rows (other columns get NaN rows)'),
    (R, '            panels["currency"].setdefault(t, held[t])\n', '',
     'the added name gets no currency (valued at a wrong FX)'),
    (R, '            if list(px.columns) == ["Close"]:\n                px.columns = [missing[0]]\n', '',
     'the flat single-ticker download shape is not recognised'),
    (R, '        if t in px.columns and px[t].notna().any():',
     '        if t in px.columns:', 'an all-NaN download is added as if it were a price'),
    (R, '    if lost:\n', '    if False:\n', 'a held name left without a price is silent'),
    (R, '    if added:\n        logging.warning(', '    if False:\n        logging.warning(',
     'added names are not reported'),
    (R, '    except Exception as e:  # noqa: BLE001\n        logging.warning("held-name price download failed (%s)", e)',
     '    except ImportError as e:\n        logging.warning("held-name price download failed (%s)", e)',
     'a failed download crashes the run'),
]

if __name__ == "__main__":
    sys.exit(run("test_held_prices.py", MUTATIONS))
