"""Render ForecastSignal as LLM-readable text injected into agent prompts."""

from .signal import ForecastSignal


def render_for_llm(signal: ForecastSignal) -> str:
    candle_rows = "\n".join(
        f"  {c.timestamp[:10]}  O:{c.open:.2f}  H:{c.high:.2f}  "
        f"L:{c.low:.2f}  C:{c.close:.2f}  Vol:{c.volume:,.0f}"
        for c in signal.predicted_candles
    )
    return f"""
=== KRONOS FOUNDATION MODEL FORECAST — {signal.ticker} ===
Generated : {signal.generated_at[:19]} UTC
Horizon   : {signal.horizon} trading days ahead
Last close: ${signal.last_close:.2f}

Predicted return      : {signal.predicted_return:+.2%}
Predicted volatility  : {signal.predicted_volatility:.2%}  (intra-candle range std)
Predicted max drawdown: {signal.predicted_max_drawdown:.2%}
Directional signal    : {signal.directional_signal}
Confidence            : {signal.confidence:.0%}

Predicted candles:
{candle_rows}

NOTE: Kronos forecasts price structure only. Incorporate fundamentals,
news, and macro context through other agents before final decision.
=================================================================
""".strip()


def render_batch_summary(signals: dict[str, ForecastSignal]) -> str:
    """One-liner per ticker — used in Director agent system prompt."""
    lines = []
    for ticker, s in signals.items():
        bar = "▲" if s.directional_signal == "BUY" else ("▼" if s.directional_signal == "SELL" else "─")
        lines.append(
            f"  {bar} {ticker:<6}  {s.predicted_return:+.2%}  "
            f"conf={s.confidence:.0%}  vol={s.predicted_volatility:.2%}"
        )
    return "KRONOS BATCH FORECAST SUMMARY:\n" + "\n".join(lines)
