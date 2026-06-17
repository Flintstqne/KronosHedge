"""Intraday price fetch and market-hours utilities."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf

_ET = ZoneInfo("America/New_York")


def market_is_open() -> bool:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def is_past_signal_time(signal_time: str = "10:00") -> bool:
    h, m = map(int, signal_time.split(":"))
    now = datetime.now(_ET)
    threshold = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return now >= threshold


def fetch_intraday_prices(
    tickers: list[str],
    as_of_time: str = "10:00",
) -> dict[str, float]:
    """Latest 5-min bar close at or before as_of_time ET for each ticker."""
    h, m = map(int, as_of_time.split(":"))
    result: dict[str, float] = {}
    try:
        raw = yf.download(
            tickers, period="1d", interval="5m",
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            return result
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, :]
        if hasattr(close, "columns"):
            # Multi-ticker download
            for col in close.columns:
                ticker = str(col)
                series = close[col].dropna()
                series = series[
                    series.index.tz_convert(_ET).map(
                        lambda ts: ts.hour < h or (ts.hour == h and ts.minute <= m)
                    )
                ]
                if not series.empty:
                    result[ticker] = float(series.iloc[-1])
        else:
            # Single-ticker (Series)
            series = close.dropna()
            series = series[
                series.index.tz_convert(_ET).map(
                    lambda ts: ts.hour < h or (ts.hour == h and ts.minute <= m)
                )
            ]
            if not series.empty and tickers:
                result[tickers[0]] = float(series.iloc[-1])
    except Exception:
        pass
    return result
