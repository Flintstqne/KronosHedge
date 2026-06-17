"""
Seed demo audit logs so the dashboard can be explored without running a live cycle.
Generates 30 days of realistic synthetic data where Kronos predictions are
partially correlated with actual next-period price moves (~65% directional accuracy).

Usage: python scripts/seed_demo_data.py
       python scripts/seed_demo_data.py --days 60 --accuracy 0.70
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO",   # Technology
    "AMZN", "TSLA", "HD", "MCD",                         # Consumer Discretionary
    "UNH", "LLY", "JNJ", "ABBV",                         # Healthcare
    "JPM", "V", "MA", "GS",                              # Financials
    "XOM", "CVX",                                         # Energy
    "CAT", "RTX", "HON",                                  # Industrials
    "COST", "WMT", "PG",                                  # Consumer Staples
    "NFLX", "T",                                          # Communication
    "LIN",                                                # Materials
    "NEE",                                                # Utilities
    "SPCX",                                               # ETF
]
INVESTORS = ["warren_buffett", "peter_lynch", "charlie_munger", "benjamin_graham", "george_soros"]
LOG_DIR   = Path("logs/audit")

BASE_PRICES = {
    "AAPL": 213.0, "MSFT": 430.0, "NVDA":  135.0, "GOOGL": 178.0,
    "META": 600.0, "AVGO": 185.0, "AMZN":  215.0, "TSLA":  340.0,
    "HD":   350.0, "MCD":  295.0, "UNH":   590.0, "LLY":   880.0,
    "JNJ":  155.0, "ABBV": 175.0, "JPM":   235.0, "V":     285.0,
    "MA":   500.0, "GS":   540.0, "XOM":    115.0, "CVX":   155.0,
    "CAT":  365.0, "RTX":  130.0, "HON":   230.0, "COST":  920.0,
    "WMT":   95.0, "PG":   170.0, "NFLX":  680.0, "T":      20.0,
    "LIN":  460.0, "NEE":   75.0, "SPCX":  28.0,
}


def rand_signal(bias: float = 0.0) -> str:
    weights = [max(0.05, 0.33 + bias), max(0.05, 0.34 - bias * 0.5), max(0.05, 0.33 - bias * 0.5)]
    return random.choices(["bullish", "neutral", "bearish"], weights=weights)[0]


def make_candles(base_price: float, trend: float, horizon: int = 5) -> list[dict]:
    candles, price = [], base_price
    for i in range(horizon):
        change = trend + random.gauss(0, 0.006)
        price *= (1 + change)
        spread = price * random.uniform(0.005, 0.012)
        candles.append({
            "timestamp": (date.today() + timedelta(days=i + 1)).isoformat(),
            "open":   round(price * random.uniform(0.998, 1.002), 2),
            "high":   round(price + spread, 2),
            "low":    round(price - spread, 2),
            "close":  round(price, 2),
            "volume": random.randint(10_000_000, 80_000_000),
        })
    return candles


def seed(n_days: int = 365, target_accuracy: float = 0.65,
         start_equity: float = 100_000.0) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for f in LOG_DIR.glob("run_demo_*.json"):
        f.unlink()

    prices = dict(BASE_PRICES)
    equity = start_equity
    # Store actual next-day moves so we can back-fill with correlated predictions
    day_moves: list[dict[str, float]] = []   # actual % change per ticker per day

    # ── Pass 1: generate actual price path ───────────────────────────────────
    # n_days is calendar days; stop at today so we never generate future dates.
    calendar: list[date] = []
    d = date.today() - timedelta(days=n_days)
    while d <= date.today():
        if d.weekday() < 5:
            calendar.append(d)
        d += timedelta(days=1)

    price_history: list[dict[str, float]] = []
    for _ in calendar:
        snapshot = {}
        for t in TICKERS:
            chg = random.gauss(0.0005, 0.012)
            prices[t] *= (1 + chg)
            snapshot[t] = round(prices[t], 2)
        price_history.append(snapshot)

    # ── Pass 2: build audit records ──────────────────────────────────────────
    prices = dict(BASE_PRICES)   # reset to beginning
    equity = start_equity

    for i, d in enumerate(calendar):
        prev_prices = dict(prices)
        # Advance prices to this day
        for t in TICKERS:
            prices[t] = price_history[i][t]

        daily_return = random.gauss(0.0008, 0.007)
        equity *= (1 + daily_return)

        # Kronos prediction: with probability `target_accuracy`, predict the
        # correct direction of the NEXT day's move; otherwise random.
        next_prices = price_history[i + 1] if i + 1 < len(calendar) else prices
        kronos_signals: dict = {}
        for t in TICKERS:
            actual_next_ret = (next_prices[t] - prices[t]) / prices[t]
            if random.random() < target_accuracy:
                # Predict correct direction, noisy magnitude
                pred_ret = actual_next_ret * random.uniform(0.4, 1.6) + random.gauss(0, 0.003)
            else:
                # Wrong direction
                pred_ret = -actual_next_ret * random.uniform(0.4, 1.2) + random.gauss(0, 0.003)

            vol    = abs(random.gauss(0.01, 0.003))
            conf   = random.uniform(0.50, 0.92)
            direction = "BUY" if pred_ret > 0.003 else ("SELL" if pred_ret < -0.003 else "HOLD")
            trend  = pred_ret / 5  # per-candle trend

            kronos_signals[t] = {
                "ticker": t,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "horizon": 5,
                "predicted_candles": make_candles(prices[t], trend),
                "predicted_return":  round(pred_ret, 5),
                "predicted_volatility": round(vol, 5),
                "predicted_max_drawdown": round(-abs(random.gauss(0.004, 0.002)), 5),
                "directional_signal": direction,
                "confidence": round(conf, 4),
                "last_close": prices[t],
            }

        # Agent signals biased by Kronos prediction direction
        def agent_sig(ticker: str) -> dict:
            kron_ret = kronos_signals[ticker]["predicted_return"]
            bias = 0.3 if kron_ret > 0 else (-0.3 if kron_ret < 0 else 0)
            return {
                "signal": rand_signal(bias),
                "confidence": random.randint(45, 88),
                "reasoning": "Demo signal.",
            }

        def investor_sigs() -> dict:
            out: dict = {}
            for inv in INVESTORS:
                out[inv] = {t: agent_sig(t) for t in TICKERS}
            return out

        # Weights proportional to positive predicted returns
        raw_w = {t: max(0, kronos_signals[t]["predicted_return"]) for t in TICKERS}
        total_w = sum(raw_w.values()) or 1.0
        final_weights = {t: round(v / total_w, 4) for t, v in raw_w.items()}

        orders = []
        for t, w in final_weights.items():
            if w > 0.06 and random.random() > 0.4:
                orders.append({
                    "ticker": t, "side": "buy",
                    "notional_usd": round(equity * w, 2),
                    "order_id": f"DEMO-{random.randint(100000,999999)}",
                    "status": "filled",
                })

        ts = datetime(d.year, d.month, d.day, 14, 30, tzinfo=timezone.utc)

        record = {
            "run_id":              f"demo_{d.isoformat()}",
            "timestamp":           ts.isoformat(),
            "portfolio_equity":    round(equity, 2),
            "kronos_signals":      kronos_signals,
            "technical_signals":   {t: agent_sig(t) for t in TICKERS},
            "fundamental_signals": {t: agent_sig(t) for t in TICKERS},
            "sentiment_signals":   {t: agent_sig(t) for t in TICKERS},
            "valuation_signals":   {t: agent_sig(t) for t in TICKERS},
            "investor_signals":    investor_sigs(),
            "risk_assessments":    {
                t: {"approved": True, "max_position_pct": 0.05, "reasoning": "Demo."}
                for t in TICKERS
            },
            "portfolio_decisions": {
                t: {
                    "action": kronos_signals[t]["directional_signal"],
                    "quantity_pct": round(final_weights.get(t, 0), 4),
                    "reasoning": "Demo.",
                }
                for t in TICKERS
            },
            "qlib_weights":  final_weights,
            "final_weights": final_weights,
            "orders":        orders,
        }

        fname = f"run_demo_{d.isoformat()}_{i:03d}.json"
        (LOG_DIR / fname).write_text(json.dumps(record, indent=2))

    print(f"Seeded {n_days} records  ({calendar[0]} to {calendar[-1]})  → {LOG_DIR}/")
    print(f"Start equity: ${start_equity:,.2f}  Final: ${equity:,.2f}  "
          f"({(equity/start_equity - 1):+.2%})")
    print(f"Target Kronos accuracy: ~{target_accuracy:.0%} directional")
    print("Run:  .venv/bin/streamlit run monitoring/dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",     type=int,   default=365)
    parser.add_argument("--accuracy", type=float, default=0.65)
    parser.add_argument("--cash",     type=float, default=100_000.0)
    args = parser.parse_args()
    seed(n_days=args.days, target_accuracy=args.accuracy, start_equity=args.cash)
