"""Alt-data signal family (placeholder).

Candidates: news sentiment, SEC EDGAR filings text, options skew, short interest,
Google Trends. Each needs its own ingestion path; design alt-data signals so they
reduce to the same `[date × ticker]` panel shape as price-based signals.
"""
from __future__ import annotations
