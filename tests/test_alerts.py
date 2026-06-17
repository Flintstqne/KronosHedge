"""Tests for the alert system."""

import os
from unittest.mock import MagicMock, patch

import pytest

from monitoring.alerts import (
    AlertRule,
    _send_discord,
    check_drawdown,
    check_kronos_accuracy,
    check_no_orders,
)


def test_drawdown_no_alert():
    assert check_drawdown(98_000, 100_000, threshold=-0.05) is None


def test_drawdown_warning():
    alert = check_drawdown(94_000, 100_000, threshold=-0.05)
    assert alert is not None
    assert alert.severity in ("warning", "critical")
    assert "6.00%" in alert.message


def test_drawdown_critical():
    alert = check_drawdown(88_000, 100_000, threshold=-0.05)
    assert alert is not None
    assert alert.severity == "critical"


def test_accuracy_no_alert():
    assert check_kronos_accuracy(0.65) is None


def test_accuracy_warning():
    alert = check_kronos_accuracy(0.40)
    assert alert is not None
    assert "40%" in alert.message


def test_no_orders_alert():
    alert = check_no_orders([], equity=100_000)
    assert alert is not None
    assert alert.severity == "info"


def test_no_orders_no_alert_when_orders_exist():
    from execution.base import Order
    orders = [Order(ticker="AAPL", side="buy", notional_usd=1000)]
    assert check_no_orders(orders, equity=100_000) is None


# ── Discord tests ─────────────────────────────────────────────────────────────

def test_discord_skipped_when_no_url(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with patch("monitoring.alerts.httpx.post") as mock_post:
        _send_discord(AlertRule("Test", "msg", "warning"), {})
        mock_post.assert_not_called()


def test_discord_posts_embed(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch("monitoring.alerts.httpx.post", return_value=mock_resp) as mock_post:
        _send_discord(
            AlertRule("DrawdownBreach", "Portfolio down 6%", "critical"),
            {"equity": 94000, "peak": 100000},
        )
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        embed = payload["embeds"][0]
        assert embed["color"] == 0xef5350          # critical = red
        assert "DrawdownBreach" in embed["title"]
        assert "🚨" in embed["title"]
        assert embed["description"] == "Portfolio down 6%"
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "equity" in fields


def test_discord_warning_color(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch("monitoring.alerts.httpx.post", return_value=mock_resp) as mock_post:
        _send_discord(AlertRule("AccuracyLow", "Accuracy 40%", "warning"), {})
        payload = mock_post.call_args.kwargs["json"]
        assert payload["embeds"][0]["color"] == 0xffa726   # warning = amber


def test_discord_info_color(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch("monitoring.alerts.httpx.post", return_value=mock_resp) as mock_post:
        _send_discord(AlertRule("NoOrders", "No orders placed", "info"), {})
        payload = mock_post.call_args.kwargs["json"]
        assert payload["embeds"][0]["color"] == 0x42a5f5   # info = blue
