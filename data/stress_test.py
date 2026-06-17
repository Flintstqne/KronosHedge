"""Replay current portfolio weights against historical stress scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STRESS_PATH = Path("logs/stress_test.json")

SCENARIOS: dict[str, tuple[str, str]] = {
    "COVID Crash (Feb–Mar 2020)":   ("2020-02-19", "2020-03-23"),
    "COVID Recovery (Mar–Aug 2020)": ("2020-03-23", "2020-08-28"),
    "2022 Rate Hike Bear":           ("2022-01-03", "2022-10-13"),
    "2018 Q4 Selloff":               ("2018-10-01", "2018-12-24"),
}


def run(weights: dict[str, float] | None = None) -> dict:
    import yfinance as yf

    cfg = yaml.safe_load(open("config/settings.yaml"))
    tickers = cfg["universe"]["tickers"]

    if not weights:
        weights = {t: 1 / len(tickers) for t in tickers}

    active = {t: w for t, w in weights.items() if w > 0}
    total = sum(active.values())
    active = {t: w / total for t, w in active.items()}

    scenarios_out: dict = {}

    import warnings as _w

    for name, (start, end) in SCENARIOS.items():
        try:
            # Use Ticker.history() per symbol — always returns a flat DataFrame
            # with a plain "Close" column regardless of yfinance version.
            ticker_closes: dict[str, pd.Series] = {}
            for ticker in list(active.keys()):
                try:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        hist = yf.Ticker(ticker).history(start=start, end=end)
                    if hist.empty or "Close" not in hist.columns:
                        continue
                    s = hist["Close"].dropna()
                    if len(s) > 1:
                        ticker_closes[ticker] = s
                except Exception:
                    continue

            if not ticker_closes:
                scenarios_out[name] = {"error": "no tickers returned data for this period"}
                continue

            # Align on a common date index, forward-fill gaps
            closes = pd.DataFrame(ticker_closes).ffill()
            returns = closes.pct_change().iloc[1:].fillna(0)

            if returns.empty:
                scenarios_out[name] = {"error": "returns empty after cleaning"}
                continue

            # Normalise weights to tickers that have data
            avail_w = {t: w for t, w in active.items() if t in returns.columns}
            if not avail_w:
                scenarios_out[name] = {"error": "none of the weighted tickers have data"}
                continue
            total_avail = sum(avail_w.values())
            avail_w = {t: w / total_avail for t, w in avail_w.items()}

            port_ret = pd.Series(0.0, index=returns.index)
            for t, w in avail_w.items():
                port_ret += w * returns[t].fillna(0)

            equity = [100.0]
            for r in port_ret:
                rv = float(r) if np.isfinite(float(r)) else 0.0
                equity.append(round(equity[-1] * (1 + rv), 2))

            dates = [start] + [str(d.date()) for d in returns.index]
            eq_arr = np.array(equity)
            peak = np.maximum.accumulate(eq_arr)
            max_dd = float(np.min((eq_arr - peak) / peak))

            scenarios_out[name] = {
                "start":        start,
                "end":          end,
                "tickers_used": len(avail_w),
                "total_return": round(float(equity[-1] / equity[0] - 1), 4),
                "max_drawdown": round(max_dd, 4),
                "equity_curve": [{"date": d, "value": e} for d, e in zip(dates, equity)],
            }
        except Exception as e:
            scenarios_out[name] = {"error": str(e)}

    result = {
        "computed_at":  str(pd.Timestamp.now()),
        "weights_used": active,
        "scenarios":    scenarios_out,
    }
    STRESS_PATH.parent.mkdir(exist_ok=True)
    STRESS_PATH.write_text(json.dumps(result, indent=2, allow_nan=False,
                                      default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)))
    return result


if __name__ == "__main__":
    out = run()
    for name, s in out["scenarios"].items():
        if "error" in s:
            print(f"{name}: ERROR {s['error']}")
        else:
            print(f"{name}: {s['total_return']:+.1%} return, {s['max_drawdown']:.1%} max DD")
