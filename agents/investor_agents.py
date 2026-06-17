"""
Investor philosophy agents (Warren Buffett, Peter Lynch, Charlie Munger, etc.)
Each receives Kronos context + ticker and outputs a signal.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kronos_bridge import render_for_llm, ForecastSignal
from .llm import get_llm

INVESTOR_PERSONAS = {
    "warren_buffett": (
        "You are Warren Buffett. You focus on competitive moats, earnings power, "
        "and buying wonderful companies at fair prices. You distrust short-term price forecasts "
        "but use them as one data point among many."
    ),
    "peter_lynch": (
        "You are Peter Lynch. You look for 'ten-baggers' — companies growing fast that "
        "Wall Street hasn't noticed yet. You favor growth at a reasonable price (GARP)."
    ),
    "charlie_munger": (
        "You are Charlie Munger. You apply mental models across disciplines. "
        "You avoid businesses you don't understand and focus on quality at any price."
    ),
    "benjamin_graham": (
        "You are Benjamin Graham. You focus purely on margin of safety and intrinsic value. "
        "You are deeply skeptical of momentum signals but note price trends for risk context."
    ),
    "george_soros": (
        "You are George Soros. You look for reflexivity — market narratives that become "
        "self-fulfilling. Price momentum and trend shifts are highly relevant to you."
    ),
}

SIGNAL_SCHEMA = (
    '{"signal": "bullish" | "bearish" | "neutral", "confidence": <0-100 int>, '
    '"reasoning": "<1 sentence>"}'
)


def _run_investor(
    name: str,
    persona: str,
    ticker: str,
    forecast: ForecastSignal,
    llm: Any,
) -> dict:
    context = render_for_llm(forecast)
    response = llm.invoke([
        SystemMessage(content=f"{persona}\n\nOutput JSON only: {SIGNAL_SCHEMA}"),
        HumanMessage(content=f"{context}\n\nWhat is your signal for {ticker}?"),
    ])
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"signal": "neutral", "confidence": 50, "reasoning": "Parse error fallback."}


def investor_agents(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm(
        provider=state.get("llm_provider", "anthropic"),
        model=state.get("llm_model", "claude-sonnet-4-6"),
    )
    kronos_signals: dict[str, ForecastSignal] = state["kronos_signals"]
    enabled: list[str] = [
        k for k in state.get("enabled_agents", list(INVESTOR_PERSONAS.keys()))
        if k in INVESTOR_PERSONAS
    ]

    investor_signals: dict[str, dict[str, dict]] = {}

    for ticker, forecast in kronos_signals.items():
        investor_signals[ticker] = {}
        for name in enabled:
            investor_signals[ticker][name] = _run_investor(
                name, INVESTOR_PERSONAS[name], ticker, forecast, llm
            )

    return {**state, "investor_signals": investor_signals}
