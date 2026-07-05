# algo_trading

A quantitative equity research framework. Multi-signal alpha on US equities, with planned extension to derivatives and FX. Tooling biased toward institutional patterns: cross-sectional panel signals, IC-weighted combination, cvxpy factor-neutral optimization, realistic transaction costs.

## Layout

```
data/         # ingestion + storage (yfinance now, IBKR later, DuckDB cache)
signals/      # one file per signal family
combination/  # winsorize / z-score / decay / IC-weighted combine
backtest/     # vectorized engine + cost model + metrics
portfolio/    # sizing + cvxpy factor-neutral optimizer
risk/         # exposure monitoring + drawdown controls
research/     # notebooks for exploratory work
scripts/      # CLI entry points (e.g. `run_momentum_backtest.py`)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# First end-to-end run: 12-1 momentum on S&P 500, 10y backtest
python scripts/run_momentum_backtest.py
```

Results (cumulative return PNG, summary CSV) are written to `results/`.

## Design notes

- **Panel-first.** Signals operate on wide DataFrames `[date × ticker]`, never per-ticker loops. This scales to thousands of names.
- **Point-in-time universe.** The S&P 500 list as-of-today suffers from severe survivorship bias. The universe loader is structured so that historical constituents can be slotted in once a real source (CRSP, Sharadar, Norgate) is wired up.
- **Costs are not optional.** The backtester always charges spread + a simple square-root market-impact term. Strategies that look good gross often vanish net.
