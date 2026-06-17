"""
Agent pipeline orchestrator.
Runs: technicals → investor_agents → fundamentals → sentiment → valuation
       → risk_manager → portfolio_manager
Returns final AgentState with all signals populated.
"""

from __future__ import annotations

from typing import Any, TypedDict

from kronos_bridge import ForecastSignal

from .technicals import technicals_agent
from .investor_agents import investor_agents
from .fundamentals import fundamentals_agent, sentiment_agent, valuation_agent
from .risk_portfolio import risk_manager_agent, portfolio_manager_agent
from .news_sentiment_agent import news_sentiment_agent


class AgentState(TypedDict, total=False):
    tickers: list[str]
    kronos_signals: dict[str, ForecastSignal]
    llm_provider: str
    llm_model: str
    enabled_agents: list[str]
    max_position_pct: float
    news_context: dict                    # injected by run_agent_pipeline
    # Outputs filled stage by stage
    technical_signals: dict[str, dict]
    investor_signals: dict[str, dict[str, dict]]
    fundamental_signals: dict[str, dict]
    sentiment_signals: dict[str, dict]
    valuation_signals: dict[str, dict]
    news_sentiment_signals: dict[str, dict]
    risk_assessments: dict[str, dict]
    portfolio_decisions: dict[str, dict]


_PIPELINE = [
    news_sentiment_agent,   # runs first — enriches context for downstream agents
    technicals_agent,
    investor_agents,
    fundamentals_agent,
    sentiment_agent,
    valuation_agent,
    risk_manager_agent,
    portfolio_manager_agent,
]


def run_agent_pipeline(
    kronos_signals: dict[str, ForecastSignal],
    llm_provider: str = "anthropic",
    llm_model: str = "claude-sonnet-4-6",
    enabled_agents: list[str] | None = None,
    max_position_pct: float = 0.05,
    news_context: dict | None = None,
) -> AgentState:
    state: AgentState = {
        "tickers": list(kronos_signals.keys()),
        "kronos_signals": kronos_signals,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "enabled_agents": enabled_agents or [],
        "max_position_pct": max_position_pct,
        "news_context": news_context or {},
    }
    for agent_fn in _PIPELINE:
        state = agent_fn(state)
    return state
