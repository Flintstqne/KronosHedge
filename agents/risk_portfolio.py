"""
Risk Manager and Portfolio Manager agents.
Risk Manager: checks signals against drawdown/volatility limits.
Portfolio Manager: aggregates all signals → final position dict.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kronos_bridge import ForecastSignal
from .llm import get_llm
from .prompts import load as load_prompt


def _aggregate_signals(state: dict[str, Any], ticker: str) -> dict:
    """Flatten all signal dicts into a per-ticker summary for the LLM."""
    sources = {
        "technical": state.get("technical_signals", {}).get(ticker),
        "fundamental": state.get("fundamental_signals", {}).get(ticker),
        "sentiment": state.get("sentiment_signals", {}).get(ticker),
        "valuation": state.get("valuation_signals", {}).get(ticker),
    }
    investor_block = state.get("investor_signals", {}).get(ticker, {})
    for name, sig in investor_block.items():
        sources[name] = sig

    # Filter None
    return {k: v for k, v in sources.items() if v is not None}


def risk_manager_agent(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm(state.get("llm_provider", "anthropic"), state.get("llm_model", "claude-sonnet-4-6"))
    max_position_pct: float = state.get("max_position_pct", 0.05)
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        signals = _aggregate_signals(state, ticker)
        signals_str = json.dumps(signals, indent=2)

        risk_prompt = (
            load_prompt("risk_manager") + f"\nMax position size: {max_position_pct:.0%}. "
            f'Output JSON: {{"approved": true|false, "max_position_pct": <float 0-{max_position_pct}>, "reasoning": "<str>"}}'
        )
        response = llm.invoke([
            SystemMessage(content=risk_prompt),
            HumanMessage(
                content=(
                    f"Ticker: {ticker}\n"
                    f"Kronos predicted volatility: {forecast.predicted_volatility:.2%}\n"
                    f"Kronos predicted max drawdown: {forecast.predicted_max_drawdown:.2%}\n"
                    f"Agent signals:\n{signals_str}"
                )
            ),
        ])
        try:
            results[ticker] = json.loads(response.content)
        except json.JSONDecodeError:
            results[ticker] = {
                "approved": True,
                "max_position_pct": max_position_pct / 2,
                "reasoning": "Defaulted to half max on parse error.",
            }

    return {**state, "risk_assessments": results}


def portfolio_manager_agent(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm(state.get("llm_provider", "anthropic"), state.get("llm_model", "claude-sonnet-4-6"))
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        risk = state.get("risk_assessments", {}).get(ticker, {})
        if not risk.get("approved", True):
            results[ticker] = {
                "action": "HOLD",
                "quantity_pct": 0.0,
                "reasoning": "Blocked by risk manager.",
            }
            continue

        signals = _aggregate_signals(state, ticker)
        bullish = sum(1 for s in signals.values() if s and s.get("signal") == "bullish")
        bearish = sum(1 for s in signals.values() if s and s.get("signal") == "bearish")
        total = max(len(signals), 1)

        bull_pct = bullish / total
        bear_pct = bearish / total
        max_pos = risk.get("max_position_pct", 0.05)

        if bull_pct >= 0.6:
            action = "BUY"
            qty_pct = max_pos * bull_pct
        elif bear_pct >= 0.6:
            action = "SELL"
            qty_pct = max_pos * bear_pct
        else:
            action = "HOLD"
            qty_pct = 0.0

        results[ticker] = {
            "action": action,
            "quantity_pct": round(qty_pct, 4),
            "bullish_votes": bullish,
            "bearish_votes": bearish,
            "total_agents": total,
            "reasoning": (
                f"{bullish}/{total} agents bullish, {bearish}/{total} bearish. "
                f"Kronos: {forecast.directional_signal} {forecast.predicted_return:+.2%}."
            ),
        }

    return {**state, "portfolio_decisions": results}
