"""
Alert system. Fires when portfolio crosses configured thresholds.
Writes to ./logs/alerts.jsonl and optionally POSTs to a webhook URL.

Supported channels:
  - File log (always)
  - Discord webhook (set DISCORD_WEBHOOK_URL in .env)
  - Generic webhook (set ALERT_WEBHOOK_URL in .env)
  - Email via SMTP (set ALERT_SMTP_* vars in .env)
"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

ALERT_LOG = Path("./logs/alerts.jsonl")


class AlertRule:
    def __init__(self, name: str, message: str, severity: str = "warning"):
        self.name     = name
        self.message  = message
        self.severity = severity  # info | warning | critical


def _write_log(alert: AlertRule, context: dict) -> None:
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name":      alert.name,
        "severity":  alert.severity,
        "message":   alert.message,
        **context,
    }
    with ALERT_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    log.warning("alert_fired", **record)


_SEVERITY_COLORS = {
    "info":     0x42a5f5,   # blue
    "warning":  0xffa726,   # amber
    "critical": 0xef5350,   # red
}

_SEVERITY_EMOJI = {
    "info":     "ℹ️",
    "warning":  "⚠️",
    "critical": "🚨",
}


def _send_discord(alert: AlertRule, context: dict) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return

    emoji   = _SEVERITY_EMOJI.get(alert.severity, "📢")
    color   = _SEVERITY_COLORS.get(alert.severity, 0x78909c)
    fields  = [
        {"name": k, "value": str(v), "inline": True}
        for k, v in context.items()
        if k not in ("timestamp",)
    ]

    payload = {
        "username":   "Kronos Hedge",
        "avatar_url": "https://em-content.zobj.net/source/twitter/376/chart-increasing_1f4c8.png",
        "embeds": [{
            "title":       f"{emoji}  {alert.name}",
            "description": alert.message,
            "color":       color,
            "fields":      fields,
            "footer":      {"text": f"Kronos Hedge • {alert.severity.upper()}"},
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }],
    }
    try:
        r = httpx.post(url, json=payload, timeout=5)
        r.raise_for_status()
    except Exception as exc:
        log.error("discord_failed", error=str(exc))


def _send_webhook(alert: AlertRule, context: dict) -> None:
    url = os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        return
    payload = {
        "text":     f"[{alert.severity.upper()}] {alert.name}: {alert.message}",
        "severity": alert.severity,
        **context,
    }
    try:
        httpx.post(url, json=payload, timeout=5)
    except Exception as exc:
        log.error("webhook_failed", error=str(exc))


def _send_email(alert: AlertRule) -> None:
    host     = os.getenv("ALERT_SMTP_HOST")
    port     = int(os.getenv("ALERT_SMTP_PORT", "587"))
    user     = os.getenv("ALERT_SMTP_USER")
    password = os.getenv("ALERT_SMTP_PASSWORD")
    to_addr  = os.getenv("ALERT_EMAIL_TO")

    if not all([host, user, password, to_addr]):
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = f"[Kronos Hedge {alert.severity.upper()}] {alert.name}"
        msg["From"]    = user
        msg["To"]      = to_addr
        msg.set_content(alert.message)
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
    except Exception as exc:
        log.error("email_failed", error=str(exc))


def fire(alert: AlertRule, context: dict | None = None) -> None:
    ctx = context or {}
    _write_log(alert, ctx)
    _send_discord(alert, ctx)
    _send_webhook(alert, ctx)
    if alert.severity == "critical":
        _send_email(alert)


# ── Pre-built alert checks ────────────────────────────────────────────────────

def check_drawdown(
    current_equity: float,
    peak_equity: float,
    threshold: float = -0.05,
) -> AlertRule | None:
    if peak_equity <= 0:
        return None
    dd = (current_equity - peak_equity) / peak_equity
    if dd <= threshold:
        return AlertRule(
            name="DrawdownBreach",
            message=(
                f"Portfolio drawdown {dd:.2%} exceeded threshold {threshold:.2%}. "
                f"Current equity: ${current_equity:,.2f}, peak: ${peak_equity:,.2f}."
            ),
            severity="critical" if dd <= threshold * 2 else "warning",
        )
    return None


def check_kronos_accuracy(accuracy: float, threshold: float = 0.45) -> AlertRule | None:
    if accuracy < threshold:
        return AlertRule(
            name="KronosAccuracyLow",
            message=(
                f"Kronos directional accuracy {accuracy:.0%} fell below threshold {threshold:.0%}. "
                "Consider retraining or reviewing model inputs."
            ),
            severity="warning",
        )
    return None


def check_no_orders(orders: list, equity: float) -> AlertRule | None:
    if not orders and equity > 0:
        return AlertRule(
            name="NoOrdersPlaced",
            message="Cycle completed but no orders were placed. Check signal strength or minimum order size.",
            severity="info",
        )
    return None


def check_orders_placed(orders: list, equity: float) -> AlertRule | None:
    if not orders:
        return None
    buys  = [o for o in orders if o.side == "buy"]
    sells = [o for o in orders if o.side == "sell"]
    parts = []
    if buys:
        parts.append(f"Bought: {', '.join(o.ticker for o in buys)}")
    if sells:
        parts.append(f"Sold: {', '.join(o.ticker for o in sells)}")
    total_notional = sum(abs(o.notional_usd) for o in orders)
    return AlertRule(
        name="OrdersPlaced",
        message=f"{len(orders)} order(s) executed. {' | '.join(parts)}. "
                f"Total notional: ${total_notional:,.0f}. Equity: ${equity:,.2f}.",
        severity="info",
    )


def check_regime_change(current_regime: str, previous_regime: str | None) -> AlertRule | None:
    if previous_regime is None or current_regime == previous_regime:
        return None
    severity = "warning" if current_regime == "risk_off" else "info"
    return AlertRule(
        name="RegimeChange",
        message=f"Market regime shifted: {previous_regime} → {current_regime}. "
                f"Position sizing rules have updated accordingly.",
        severity=severity,
    )


def run_all_checks(
    current_equity: float,
    peak_equity: float,
    orders: list,
    kronos_accuracy: float | None,
    cfg: dict,
    regime: str | None = None,
    previous_regime: str | None = None,
) -> list[AlertRule]:
    fired: list[AlertRule] = []
    threshold = cfg.get("monitoring", {}).get("alert_drawdown_threshold", -0.05)

    dd_alert = check_drawdown(current_equity, peak_equity, threshold)
    if dd_alert:
        fire(dd_alert, {"equity": current_equity, "peak": peak_equity})
        fired.append(dd_alert)

    if kronos_accuracy is not None:
        acc_alert = check_kronos_accuracy(kronos_accuracy)
        if acc_alert:
            fire(acc_alert, {"accuracy": kronos_accuracy})
            fired.append(acc_alert)

    no_order_alert = check_no_orders(orders, current_equity)
    if no_order_alert:
        fire(no_order_alert, {})
        fired.append(no_order_alert)

    orders_alert = check_orders_placed(orders, current_equity)
    if orders_alert:
        fire(orders_alert, {"count": len(orders)})
        fired.append(orders_alert)

    regime_alert = check_regime_change(regime or "unknown", previous_regime)
    if regime_alert:
        fire(regime_alert, {"from": previous_regime, "to": regime})
        fired.append(regime_alert)

    return fired
