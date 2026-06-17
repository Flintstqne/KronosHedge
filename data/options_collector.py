"""Collect options IV rank and put/call ratio for the universe. Saves to logs/options_data.json."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

OPT_PATH = Path("logs/options_data.json")


def _iv_rank(hist_closes, current_iv: float) -> float | None:
    if hist_closes is None or len(hist_closes) < 20:
        return None
    daily_returns = hist_closes.pct_change().dropna()
    roll_vol = daily_returns.rolling(20).std() * np.sqrt(252)
    iv_min, iv_max = float(roll_vol.min()), float(roll_vol.max())
    if iv_max <= iv_min:
        return None
    return round(max(0.0, min(1.0, (current_iv - iv_min) / (iv_max - iv_min))), 3)


def fetch_options_snapshot(tickers: list[str]) -> dict:
    import yfinance as yf

    results = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            exps = t.options
            if not exps:
                continue

            chain = t.option_chain(exps[0])
            calls, puts = chain.calls, chain.puts

            call_vol = float(calls["volume"].fillna(0).sum())
            put_vol  = float(puts["volume"].fillna(0).sum())
            pcr = round(put_vol / call_vol, 3) if call_vol > 0 else None

            hist = t.history(period="1y", progress=False)
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

            atm = calls.iloc[(calls["strike"] - spot).abs().argsort()].head(1)
            iv_30d = float(atm["impliedVolatility"].iloc[0]) if not atm.empty else None

            iv_rank = _iv_rank(hist["Close"] if not hist.empty else None, iv_30d) if iv_30d else None

            results[ticker] = {
                "iv_30d":      round(iv_30d, 4) if iv_30d else None,
                "iv_rank":     iv_rank,
                "pcr":         pcr,
                "call_volume": int(call_vol),
                "put_volume":  int(put_vol),
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}

    return {"date": str(date.today()), "tickers": results}


def run() -> dict:
    cfg = yaml.safe_load(open("config/settings.yaml"))
    data = fetch_options_snapshot(cfg["universe"]["tickers"])
    OPT_PATH.parent.mkdir(exist_ok=True)
    OPT_PATH.write_text(json.dumps(data, indent=2))
    ok = sum(1 for v in data["tickers"].values() if "error" not in v)
    print(f"Options snapshot: {ok}/{len(data['tickers'])} tickers OK")
    return data


if __name__ == "__main__":
    run()
