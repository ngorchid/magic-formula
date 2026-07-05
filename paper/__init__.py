"""Live paper-trading layer for the enhanced Magic Formula (US + Europe).

Separate from the backtest framework: it pulls CURRENT fundamentals/prices from
yfinance (deep PIT history not needed live), reuses the validated `enhanced_rank`
to pick today's targets, and runs a staggered daily IB paper-trading loop.
"""
