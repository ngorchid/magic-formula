"""Pre-market warm-up: rebuild ONLY the monthly universe/fundamentals cache (panels.pkl +
ranking.json), so the 16:00 live run hits a warm cache instead of doing the ~30-minute pull
inline and trading late (or missing Oslo's 16:20 close entirely on the 1st of the month).

It reuses the live runner's OWN `_refresh_ranking`, so the cache it writes is exactly what the
16:00 run expects — no divergence risk. Deliberately does nothing else: no IB connection, no
orders, no state write, no email. Idempotent — within a month it just reloads the existing
cache and exits in seconds; it only does the heavy pull when the month rolls over.

    python scripts/warmup_universe.py
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))   # so `import run_paper` resolves
sys.path.insert(0, str(ROOT))

import run_paper  # noqa: E402  — module-level only sets sys.path, loads .env, configures logging
from strategies.magic_formula import EnhancedMagicConfig  # noqa: E402


def main() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    cfg_mf = EnhancedMagicConfig(use_graham=False)   # identical to run_paper.main()
    ranking, _panels = run_paper._refresh_ranking(today, cfg_mf)
    logging.info("warm-up complete: %d names cached for %s", len(ranking), today[:7])


if __name__ == "__main__":
    main()
