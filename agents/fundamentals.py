"""
Fundamentals, Valuation, and Sentiment agents.
Pull data from yfinance; combine with Kronos context.
"""

import json
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, SystemMessage

from kronos_bridge import render_for_llm, ForecastSignal
from .llm import get_llm

SCHEMA = '{"signal": "bullish"|"bearish"|"neutral", "confidence": <0-100>, "reasoning": "<str>"}'


def _fetch_fundamentals(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return (
            f"P/E: {info.get('trailingPE', 'N/A')}  "
            f"P/B: {info.get('priceToBook', 'N/A')}  "
            f"Revenue growth YoY: {info.get('revenueGrowth', 'N/A')}  "
            f"Profit margin: {info.get('profitMargins', 'N/A')}  "
            f"Debt/Equity: {info.get('debtToEquity', 'N/A')}  "
            f"Forward P/E: {info.get('forwardPE', 'N/A')}"
        )
    except Exception:
        return "Fundamental data unavailable."


def fundamentals_agent(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm(state.get("llm_provider", "anthropic"), state.get("llm_model", "claude-sonnet-4-6"))
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        fundamentals = _fetch_fundamentals(ticker)
        context = render_for_llm(forecast)
        response = llm.invoke([
            SystemMessage(
                content=f"You are a fundamental analyst. Evaluate {ticker} using financial "
                        f"ratios and the Kronos price forecast as context. Output JSON: {SCHEMA}"
            ),
            HumanMessage(content=f"Fundamentals:\n{fundamentals}\n\n{context}"),
        ])
        try:
            results[ticker] = json.loads(response.content)
        except json.JSONDecodeError:
            results[ticker] = {"signal": "neutral", "confidence": 50, "reasoning": "Parse error."}

    return {**state, "fundamental_signals": results}


def sentiment_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Lightweight sentiment: uses recent news headlines from yfinance."""
    llm = get_llm(state.get("llm_provider", "anthropic"), state.get("llm_model", "claude-sonnet-4-6"))
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        try:
            news = yf.Ticker(ticker).news or []
            headlines = "\n".join(
                f"- {n.get('content', {}).get('title', '')}" for n in news[:8]
            )
        except Exception:
            headlines = "No news available."

        context = render_for_llm(forecast)
        response = llm.invoke([
            SystemMessage(
                content=f"You are a sentiment analyst. Evaluate market sentiment for {ticker} "
                        f"from recent news combined with the Kronos price forecast. Output JSON: {SCHEMA}"
            ),
            HumanMessage(content=f"Recent headlines:\n{headlines}\n\n{context}"),
        ])
        try:
            results[ticker] = json.loads(response.content)
        except json.JSONDecodeError:
            results[ticker] = {"signal": "neutral", "confidence": 50, "reasoning": "Parse error."}

    return {**state, "sentiment_signals": results}


def valuation_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Compares current price to Kronos predicted price for a simple valuation signal."""
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        ret = forecast.predicted_return
        if ret > 0.03:
            signal, conf = "bullish", min(90, int(abs(ret) * 1000))
        elif ret < -0.03:
            signal, conf = "bearish", min(90, int(abs(ret) * 1000))
        else:
            signal, conf = "neutral", 50

        results[ticker] = {
            "signal": signal,
            "confidence": conf,
            "reasoning": (
                f"Kronos predicts {ret:+.2%} over {forecast.horizon}d. "
                f"Current price ${forecast.last_close:.2f}."
            ),
        }

    return {**state, "valuation_signals": results}
