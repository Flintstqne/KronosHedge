"""
Data fetching layer. Uses yfinance as primary provider with Qlib as optional backend.
Returns OHLCV DataFrames in Kronos-compatible format.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import pandas as pd
import yfinance as yf


def fetch_ohlcv_universe(
    tickers: list[str],
    lookback_days: int = 60,
    end_date: date | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for each ticker. Returns dict of DataFrames
    with columns [open, high, low, close, volume, amount].
    """
    end = end_date or date.today()
    start = end - timedelta(days=lookback_days + 30)  # buffer for weekends/holidays

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                warnings.warn(f"No data for {ticker}")
                continue

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df["amount"] = df["close"] * df["volume"]
            df = df.dropna()
            df = df.tail(lookback_days)

            results[ticker] = df
        except Exception as exc:
            warnings.warn(f"Data fetch failed for {ticker}: {exc}")

    return results


def fetch_benchmark(ticker: str = "SPY", lookback_days: int = 252) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=lookback_days + 30)
    raw = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                      progress=False, auto_adjust=True)
    df = raw[["Close"]].rename(columns={"Close": "close"})
    return df.dropna().tail(lookback_days)
