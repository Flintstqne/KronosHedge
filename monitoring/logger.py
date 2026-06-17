"""Structured JSON audit logger. One file per run cycle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kronos_bridge import ForecastSignal
    from execution.base import Order


class AuditLogger:
    def __init__(self, log_dir: str = "./logs/audit"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        run_id: str,
        kronos_signals: dict[str, ForecastSignal],
        agent_state: dict[str, Any],
        qlib_weights: dict[str, float],
        final_weights: dict[str, float],
        orders: list[Order],
        portfolio_equity: float,
        news_context: dict | None = None,
        signal_attribution: dict | None = None,
    ) -> Path:
        order_rows = []
        slippage_bps_list = []
        for o in orders:
            slip = None
            if o.intended_price > 0 and o.fill_price > 0:
                slip = round((o.fill_price / o.intended_price - 1) * 10_000, 2)
                slippage_bps_list.append(slip)
            order_rows.append({
                "ticker": o.ticker,
                "side": o.side,
                "notional_usd": o.notional_usd,
                "order_id": o.order_id,
                "status": o.status,
                "intended_price": o.intended_price,
                "fill_price": o.fill_price,
                "slippage_bps": slip,
            })

        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_equity": portfolio_equity,
            "kronos_signals": {
                t: s.model_dump() for t, s in kronos_signals.items()
            },
            "signal_attribution": signal_attribution or {},
            "slippage_summary": {
                "mean_bps": round(sum(slippage_bps_list) / len(slippage_bps_list), 2) if slippage_bps_list else None,
                "n_fills": len(slippage_bps_list),
            },
            "technical_signals": agent_state.get("technical_signals", {}),
            "investor_signals": agent_state.get("investor_signals", {}),
            "fundamental_signals": agent_state.get("fundamental_signals", {}),
            "sentiment_signals": agent_state.get("sentiment_signals", {}),
            "valuation_signals": agent_state.get("valuation_signals", {}),
            "news_sentiment_signals": agent_state.get("news_sentiment_signals", {}),
            "news_context": {
                "ticker_news": (news_context or {}).get("ticker_news", {}),
                "earnings": (news_context or {}).get("earnings", []),
                "macro_events": (news_context or {}).get("macro_events", []),
                "as_of": (news_context or {}).get("as_of", ""),
            },
            "risk_assessments": agent_state.get("risk_assessments", {}),
            "portfolio_decisions": agent_state.get("portfolio_decisions", {}),
            "qlib_weights": qlib_weights,
            "final_weights": final_weights,
            "orders": order_rows,
        }

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        path = self._dir / f"run_{date_str}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        return path

    def load_all(self) -> list[dict]:
        records = []
        for f in sorted(self._dir.glob("run_*.json")):
            try:
                records.append(json.loads(f.read_text()))
            except Exception:
                pass
        return records

    def load_latest(self) -> dict | None:
        files = sorted(self._dir.glob("run_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text())
