"""Microstructure signal family (placeholder).

Targets: Amihud illiquidity, bid-ask bounce, intraday volume profile, kyle's lambda.
Most of these need TAQ-level or at least minute-bar data — wire up after the IBKR
historical bar feed is online.
"""
from __future__ import annotations


def amihud_illiquidity(*args, **kwargs):
    raise NotImplementedError
