"""Macro regime classifier: risk_on / transition / risk_off."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

log = structlog.get_logger()

_REGIME_TICKERS = ["^TNX", "^IRX", "HYG", "SPY"]


def fetch_regime_inputs(lookback_days: int = 400) -> pd.DataFrame:
    raw = yf.download(
        _REGIME_TICKERS, period=f"{lookback_days}d",
        progress=False, auto_adjust=True,
    )
    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).normalize()

    df = pd.DataFrame(index=close.index)

    # Yield curve: 10Y minus 3M, both in percent already
    if "^TNX" in close.columns and "^IRX" in close.columns:
        df["yield_spread"] = (close["^TNX"] - close["^IRX"]).ffill()
    else:
        df["yield_spread"] = 1.0  # bullish default on data failure

    # Credit stress: HYG vs 20-day SMA
    if "HYG" in close.columns:
        hyg = close["HYG"].ffill()
        df["hyg_mom"] = (hyg / hyg.rolling(20).mean() - 1).fillna(0)
    else:
        df["hyg_mom"] = 0.0

    # Market trend: SPY vs 200-day SMA
    if "SPY" in close.columns:
        spy = close["SPY"].ffill()
        df["spy_vs_200"] = (spy / spy.rolling(200).mean() - 1).fillna(0)
    else:
        df["spy_vs_200"] = 0.0

    return df.dropna(how="all")


def classify_regime(row: pd.Series) -> str:
    bearish = 0
    if float(row.get("yield_spread", 1.0)) < -0.5:
        bearish += 1
    if float(row.get("hyg_mom", 0.0)) < -0.02:
        bearish += 1
    if float(row.get("spy_vs_200", 0.0)) < -0.05:
        bearish += 1
    if bearish == 0:
        return "risk_on"
    if bearish == 1:
        return "transition"
    return "risk_off"


def get_regime_series(lookback_days: int = 400) -> pd.Series:
    try:
        df = fetch_regime_inputs(lookback_days)
        return df.apply(classify_regime, axis=1).rename("regime")
    except Exception as e:
        log.warning("regime_fetch_failed", error=str(e))
        # Return empty series — callers default to risk_on
        return pd.Series(dtype=str, name="regime")


def apply_regime_filter(
    weights: dict[str, float],
    regime: str,
    spy_in_prices: bool = True,
) -> dict[str, float]:
    if regime == "risk_on":
        return weights
    scale = 0.80 if regime == "transition" else 0.50
    result = {t: w * scale for t, w in weights.items()}
    if regime == "risk_off" and spy_in_prices:
        result.pop("SPY", None)
        result["SPY"] = 0.30
    return result


if __name__ == "__main__":
    series = get_regime_series()
    if len(series):
        current = series.iloc[-1]
        print(f"Current regime ({series.index[-1].date()}): {current}")
        print(series.tail(10).to_string())
    else:
        print("No regime data available.")
