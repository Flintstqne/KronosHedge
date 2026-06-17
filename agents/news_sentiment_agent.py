"""
News sentiment agent.
Synthesizes recent headlines + macro events + Kronos price forecast into:
  - per-ticker news_sentiment signal (bullish/bearish/neutral)
  - 'kronos_news_view': one sentence combining price model + news context
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kronos_bridge import render_for_llm
from .llm import get_llm

_SCHEMA = (
    '{"signal":"bullish"|"bearish"|"neutral",'
    '"confidence":<0-100>,'
    '"reasoning":"<2-3 sentences>",'
    '"kronos_news_view":"<one sentence: what price model + news mean together>"}'
)


def _macro_summary(news_context: dict) -> str:
    lines = []
    for ev in news_context.get("macro_events", []):
        d = ev["days_away"]
        when = "TODAY" if d == 0 else (f"in {d}d" if d > 0 else f"{abs(d)}d ago")
        lines.append(f"  - {ev['event']} {when} [risk: {ev['risk_level']}]")
    return "\n".join(lines) if lines else "  None in next 21 days."


def _earnings_note(ticker: str, news_context: dict) -> str:
    for ev in news_context.get("earnings", []):
        if ev["ticker"] == ticker:
            d = ev["days_away"]
            if d == 0:
                return "EARNINGS TODAY — extreme uncertainty."
            if d > 0:
                return f"Earnings in {d} day(s) — elevated uncertainty."
            return f"Earnings reported {abs(d)} day(s) ago."
    return ""


def news_sentiment_agent(state: dict[str, Any]) -> dict[str, Any]:
    news_context = state.get("news_context")
    if not news_context:
        return {**state, "news_sentiment_signals": {}}

    llm = get_llm(state.get("llm_provider", "anthropic"), state.get("llm_model", "claude-sonnet-4-6"))
    macro = _macro_summary(news_context)
    results: dict[str, dict] = {}

    for ticker, forecast in state["kronos_signals"].items():
        items = news_context.get("ticker_news", {}).get(ticker, [])
        if not items:
            continue

        headlines = "\n".join(f"- {n['title']} [{n['publisher']}]" for n in items[:8])
        earnings = _earnings_note(ticker, news_context)
        kronos_ctx = render_for_llm(forecast)

        system = (
            f"You are a senior market analyst. Synthesize recent news for {ticker} with "
            f"the Kronos quantitative price forecast. Factor in macro events. "
            f"Be concise and direct. Output ONLY valid JSON: {_SCHEMA}"
        )
        user = (
            f"Headlines for {ticker}:\n{headlines}"
            + (f"\n\n{earnings}" if earnings else "")
            + f"\n\nMacro context:\n{macro}\n\n{kronos_ctx}"
        )

        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            # Strip markdown fences if model wraps output
            raw = resp.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "signal": "neutral",
                "confidence": 50,
                "reasoning": "Analysis unavailable.",
                "kronos_news_view": "",
            }
        results[ticker] = parsed

    return {**state, "news_sentiment_signals": results}
