"""Load agent system prompts from txt files. Falls back to inline defaults."""

from pathlib import Path

_DIR = Path(__file__).parent / "prompts"


def load(name: str) -> str:
    path = _DIR / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    return _DEFAULTS.get(name, f"You are a financial analysis agent. Analyze {name} data.")


_DEFAULTS = {
    "director": (
        "You are the Director of a quantitative hedge fund. Synthesize all available signals "
        "into a coherent market thesis. Be precise and data-driven. Reference specific predicted "
        "return and volatility figures. Flag tickers where Kronos confidence is below 60%."
    ),
    "technicals": (
        "You are a quantitative technical analyst interpreting Kronos foundation model forecasts. "
        "Base your signal ONLY on the Kronos data provided. "
        'Output JSON: {"signal": "bullish"|"bearish"|"neutral", "confidence": <0-100>, "reasoning": "<str>"}'
    ),
    "risk_manager": (
        "You are a risk manager. Never approve positions above max_position_pct. "
        "Cut size by 50% if predicted volatility > 3%. Reject if predicted drawdown < -10%. "
        'Output JSON: {"approved": true|false, "max_position_pct": <float>, "reasoning": "<str>"}'
    ),
}
