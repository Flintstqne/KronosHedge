"""Tests for PerformanceMetrics using synthetic audit records."""

import pytest
from monitoring.metrics import PerformanceMetrics


def _make_records(equities: list[float]) -> list[dict]:
    records = []
    for i, eq in enumerate(equities):
        records.append({
            "timestamp": f"2024-01-{i+1:02d}T09:30:00Z",
            "portfolio_equity": eq,
            "kronos_signals": {},
            "orders": [],
        })
    return records


def test_equity_curve():
    m = PerformanceMetrics(_make_records([10000, 10100, 10050, 10200]))
    ec = m.equity_curve()
    assert len(ec) == 4
    assert list(ec["equity"]) == [10000, 10100, 10050, 10200]


def test_cumulative_return():
    m = PerformanceMetrics(_make_records([10000, 11000]))
    assert abs(m.cumulative_return() - 0.10) < 1e-6


def test_max_drawdown():
    m = PerformanceMetrics(_make_records([10000, 12000, 9000, 11000]))
    dd = m.max_drawdown()
    assert dd < 0  # must be negative
    assert abs(dd - (-0.25)) < 1e-6  # 9000/12000 - 1 = -0.25


def test_empty_records():
    m = PerformanceMetrics([])
    assert m.cumulative_return() == 0.0
    assert m.sharpe_ratio() == 0.0
    assert m.max_drawdown() == 0.0
