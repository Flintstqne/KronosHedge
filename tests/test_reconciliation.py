"""Tests for SignalReconciler."""

import pytest
from reconciliation.merger import SignalReconciler


def test_reconcile_basic():
    rec = SignalReconciler(qlib_weight=0.6, agent_weight=0.4)
    qlib = {"AAPL": 0.5, "MSFT": 0.5}
    agents = {
        "AAPL": {"action": "BUY", "quantity_pct": 0.05},
        "MSFT": {"action": "HOLD", "quantity_pct": 0.0},
    }
    result = rec.reconcile(qlib, agents)
    assert all(v >= 0 for v in result.values())
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_reconcile_sell_blocked():
    rec = SignalReconciler(qlib_weight=0.6, agent_weight=0.4)
    qlib = {"AAPL": 1.0}
    agents = {"AAPL": {"action": "SELL", "quantity_pct": 0.05}}
    result = rec.reconcile(qlib, agents)
    # SELL converts to negative agent weight, but long-only zeroes it
    assert result.get("AAPL", 0) >= 0


def test_weight_normalization():
    rec = SignalReconciler(qlib_weight=0.6, agent_weight=0.4)
    qlib = {"A": 0.3, "B": 0.3, "C": 0.3}
    agents = {k: {"action": "BUY", "quantity_pct": 0.05} for k in "ABC"}
    result = rec.reconcile(qlib, agents)
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-6
