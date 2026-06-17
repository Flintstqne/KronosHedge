"""
Forward simulation: generates N stochastic Kronos forecast paths for each
ticker, simulates portfolio equity over the horizon, and saves to JSON.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SIM_PATH = Path("logs/forward_sim.json")


def _fs_cfg() -> dict:
    return yaml.safe_load(open("config/settings.yaml")).get("forward_sim", {})


def run(n_paths: int | None = None, horizon: int | None = None) -> dict:
    _cfg    = _fs_cfg()
    n_paths = n_paths or _cfg.get("n_paths", 64)
    horizon = horizon or _cfg.get("horizon", 21)
    from qlib_pipeline.data import fetch_ohlcv_universe
    from kronos_bridge.signal import KronosSignalBridge
    from monitoring.logger import AuditLogger

    cfg        = yaml.safe_load(open("config/settings.yaml"))
    tickers    = cfg["universe"]["tickers"]
    kronos_cfg = cfg["kronos"]

    print("Fetching OHLCV…")
    ohlcv = fetch_ohlcv_universe(tickers, lookback_days=120)

    print("Loading Kronos model…")
    bridge = KronosSignalBridge(
        model_size=kronos_cfg["model_size"],
        horizon=horizon,
        device=kronos_cfg["device"],
        model_source=kronos_cfg.get("model_source", "chronos"),
    )
    bridge._load()
    predictor = bridge._predictor

    today     = date.today()
    sim_date  = str(today)

    # ── Generate per-ticker forecast paths ───────────────────────────────────
    print(f"Generating {n_paths} paths × {len(ohlcv)} tickers (horizon={horizon}d)…")
    ticker_data: dict = {}

    for ticker, df in ohlcv.items():
        if df is None or len(df) < 30:
            continue

        paths: list[list[dict]] = []
        for _ in range(n_paths):
            try:
                pred = predictor.predict(df, pred_len=horizon)
                path = [
                    {
                        "date":  str(dt.date() if hasattr(dt, "date") else dt)[:10],
                        "open":  round(float(row["open"]),  4),
                        "high":  round(float(row["high"]),  4),
                        "low":   round(float(row["low"]),   4),
                        "close": round(float(row["close"]), 4),
                    }
                    for dt, row in pred.iterrows()
                ]
                paths.append(path)
            except Exception as e:
                print(f"  {ticker} path failed: {e}")

        if not paths:
            continue

        closes = np.array([[c["close"] for c in p] for p in paths])
        opens  = np.array([[c["open"]  for c in p] for p in paths])
        highs  = np.array([[c["high"]  for c in p] for p in paths])
        lows   = np.array([[c["low"]   for c in p] for p in paths])

        # Median candle; widen high/low to P75/P25 so the candle body shows uncertainty
        median_candles = [
            {
                "date":  paths[0][i]["date"],
                "open":  round(float(np.median(opens[:,  i])), 4),
                "high":  round(float(np.percentile(highs[:, i], 75)), 4),
                "low":   round(float(np.percentile(lows[:,  i], 25)), 4),
                "close": round(float(np.median(closes[:, i])), 4),
            }
            for i in range(len(paths[0]))
        ]

        # Last 60 real candles for the chart
        real_candles = [
            {
                "date":  str(dt.date() if hasattr(dt, "date") else dt)[:10],
                "open":  round(float(row["open"]),  4),
                "high":  round(float(row["high"]),  4),
                "low":   round(float(row["low"]),   4),
                "close": round(float(row["close"]), 4),
            }
            for dt, row in df.tail(60).iterrows()
        ]

        last_close = float(df["close"].iloc[-1])
        pred_end   = median_candles[-1]["close"]
        print(f"  {ticker}: {last_close:.2f} → {pred_end:.2f} ({pred_end/last_close-1:+.1%})")

        ticker_data[ticker] = {
            "last_real_close": last_close,
            "real_candles":    real_candles,
            "paths":           paths,
            "median":          median_candles,
        }

    # ── Portfolio simulation ──────────────────────────────────────────────────
    try:
        records = AuditLogger().load_all()
        weights = records[-1]["final_weights"] if records else {}
    except Exception:
        weights = {}

    valid_w = {t: w for t, w in weights.items() if t in ticker_data}
    if not valid_w:
        valid_w = {t: 1 / len(ticker_data) for t in ticker_data}
    total_w = sum(abs(v) for v in valid_w.values())
    valid_w = {t: v / total_w for t, v in valid_w.items()}

    try:
        from execution.alpaca import AlpacaAdapter
        import os
        paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        initial = AlpacaAdapter(paper=paper).get_equity()
        print(f"Starting equity from Alpaca: ${initial:,.2f}")
    except Exception as _e:
        initial = 100_000.0
        print(f"Alpaca unavailable ({_e}), using ${initial:,.0f}")

    portfolio_paths: list[list[dict]] = []

    for path_idx in range(n_paths):
        equity = initial
        curve: list[dict] = []
        for day_idx in range(horizon):
            day_ret = 0.0
            for ticker, weight in valid_w.items():
                td = ticker_data.get(ticker)
                if not td or path_idx >= len(td["paths"]):
                    continue
                path = td["paths"][path_idx]
                if day_idx >= len(path):
                    continue
                prev = td["last_real_close"] if day_idx == 0 else path[day_idx - 1]["close"]
                curr = path[day_idx]["close"]
                day_ret += weight * ((curr / prev - 1) if prev > 0 else 0)
            equity *= 1 + day_ret
            date_str = (
                ticker_data[next(iter(ticker_data))]["paths"][path_idx][day_idx]["date"]
                if ticker_data else ""
            )
            curve.append({"date": date_str, "equity": round(equity, 2)})
        portfolio_paths.append(curve)

    n_days = min(len(c) for c in portfolio_paths) if portfolio_paths else 0
    equities_by_day = [
        [portfolio_paths[p][d]["equity"] for p in range(len(portfolio_paths))]
        for d in range(n_days)
    ]
    date_by_day = [portfolio_paths[0][d]["date"] for d in range(n_days)] if portfolio_paths else []

    def _pct_curve(pct: float) -> list[dict]:
        return [{"date": date_by_day[d], "equity": round(float(np.percentile(equities_by_day[d], pct)), 2)}
                for d in range(n_days)]

    _kron = yaml.safe_load(open("config/settings.yaml"))["kronos"]
    result = {
        "simulation_date":  sim_date,
        "horizon_days":     horizon,
        "n_paths":          n_paths,
        "initial_equity":   round(initial, 2),
        "settings": {
            "n_paths":       n_paths,
            "horizon":       horizon,
            "model_size":    _kron.get("model_size"),
            "model_source":  _kron.get("model_source", "chronos"),
        },
        "tickers":          ticker_data,
        "portfolio_paths":  portfolio_paths,
        "portfolio_median": _pct_curve(50),
        "portfolio_p10":    _pct_curve(10),
        "portfolio_p90":    _pct_curve(90),
        "weights":          valid_w,
    }

    SIM_PATH.parent.mkdir(exist_ok=True)
    SIM_PATH.write_text(json.dumps(result, indent=2, default=str))

    if portfolio_paths:
        exp_ret  = portfolio_paths[0][-1]["equity"] / initial - 1  # median path
        prob_pos = sum(1 for p in portfolio_paths if p[-1]["equity"] > initial) / len(portfolio_paths)
        med_end  = float(np.median([p[-1]["equity"] for p in portfolio_paths]))
        print(f"\nMedian end equity : ${med_end:,.0f}  ({med_end/initial-1:+.1%})")
        print(f"Probability gain  : {prob_pos:.0%}  ({sum(1 for p in portfolio_paths if p[-1]['equity']>initial)}/{len(portfolio_paths)} paths)")

    print(f"Saved → {SIM_PATH}")
    return result
