"""Collect short interest data for the universe via yfinance."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

SHORT_PATH = Path("logs/short_interest.json")


def fetch_short_interest(tickers: list[str]) -> dict:
    import yfinance as yf

    results = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            short_pct   = info.get("shortPercentOfFloat")
            days_cover  = info.get("shortRatio")
            shares_short = info.get("sharesShort")
            results[ticker] = {
                "short_pct_float": round(float(short_pct), 4) if short_pct is not None else None,
                "days_to_cover":   round(float(days_cover), 2) if days_cover is not None else None,
                "shares_short":    int(shares_short) if shares_short is not None else None,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}

    return {"date": str(date.today()), "tickers": results}


def run() -> dict:
    cfg = yaml.safe_load(open("config/settings.yaml"))
    data = fetch_short_interest(cfg["universe"]["tickers"])
    SHORT_PATH.parent.mkdir(exist_ok=True)
    SHORT_PATH.write_text(json.dumps(data, indent=2))
    ok = sum(1 for v in data["tickers"].values() if "error" not in v)
    print(f"Short interest: {ok}/{len(data['tickers'])} tickers OK")
    return data


if __name__ == "__main__":
    run()
