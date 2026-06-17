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


def _format_options(ticker: str, options_data: dict) -> str:
    d = options_data.get(ticker, {})
    if not d or "error" in d:
        return ""
    parts = []
    if d.get("iv_rank") is not None:
        parts.append(f"IV rank: {d['iv_rank']:.0%}")
    if d.get("pcr") is not None:
        skew = "bearish skew" if d["pcr"] > 1.2 else ("bullish skew" if d["pcr"] < 0.8 else "neutral skew")
        parts.append(f"Put/Call ratio: {d['pcr']:.2f} ({skew})")
    if d.get("iv_30d") is not None:
        parts.append(f"30d implied vol: {d['iv_30d']:.0%}")
    return "Options market: " + "  |  ".join(parts) if parts else ""


def _format_insider(ticker: str, insider_data: list) -> str:
    count = sum(1 for t in insider_data if t.get("ticker") == ticker)
    if count == 0:
        return ""
    return f"Insider Form 4 filings (last 60d): {count} filing{'s' if count > 1 else ''}"


def fundamentals_agent(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm(state.get("llm_provider", "anthropic"), state.get("llm_model", "claude-sonnet-4-6"))
    options_data = state.get("options_data", {})
    insider_data = state.get("insider_data", [])
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        fundamentals = _fetch_fundamentals(ticker)
        options_ctx  = _format_options(ticker, options_data)
        insider_ctx  = _format_insider(ticker, insider_data)
        extra = "\n".join(filter(None, [options_ctx, insider_ctx]))
        context = render_for_llm(forecast)
        response = llm.invoke([
            SystemMessage(
                content=f"You are a fundamental analyst. Evaluate {ticker} using financial "
                        f"ratios, options market signals, insider activity, and the Kronos price "
                        f"forecast as context. Output JSON: {SCHEMA}"
            ),
            HumanMessage(content=f"Fundamentals:\n{fundamentals}\n{extra}\n\n{context}"),
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
