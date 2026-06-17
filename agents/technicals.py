"""
Technicals agent: Kronos-powered replacement for TA-library signal.
Input:  state["kronos_signals"] dict[ticker, ForecastSignal]
Output: state["technical_signals"] dict[ticker, {signal, confidence, reasoning}]
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kronos_bridge import ForecastSignal, render_for_llm
from .llm import get_llm
from .prompts import load as load_prompt


def technicals_agent(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm(
        provider=state.get("llm_provider", "anthropic"),
        model=state.get("llm_model", "claude-sonnet-4-6"),
    )
    kronos_signals: dict[str, ForecastSignal] = state["kronos_signals"]
    results: dict[str, dict] = {}

    for ticker, forecast in kronos_signals.items():
        context = render_for_llm(forecast)
        response = llm.invoke([
            SystemMessage(content=load_prompt("technicals")),
            HumanMessage(content=f"{context}\n\nProvide your signal for {ticker}:"),
        ])
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = {
                "signal": forecast.directional_signal.lower().replace("buy", "bullish")
                          .replace("sell", "bearish").replace("hold", "neutral"),
                "confidence": int(forecast.confidence * 100),
                "reasoning": "Parsed directly from Kronos directional signal.",
            }
        results[ticker] = parsed

    return {**state, "technical_signals": results}
