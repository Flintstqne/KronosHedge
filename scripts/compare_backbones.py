"""
Parallel backbone comparison: Amazon Chronos-mini vs shiyu-coder Kronos-base.
Runs two independent backtests over the same period/universe and prints a side-by-side table.

Usage:
    uv run python scripts/compare_backbones.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlib_pipeline.backtest import Backtester

CONFIG = yaml.safe_load(open("config/settings.yaml"))
TICKERS = CONFIG["universe"]["tickers"]

COMMON = dict(
    tickers=TICKERS,
    start_date=date(2025, 6, 1),
    end_date=date(2026, 6, 1),
    initial_cash=100_000,
    run_agents=False,
    kronos_horizon=5,
    kronos_device="cpu",
    momentum_blend=CONFIG["alpha"]["momentum_blend"],
    top_n=CONFIG["alpha"]["top_n"],
    short_n=CONFIG["alpha"]["short_n"],
    pead_boost=CONFIG["alpha"]["pead_boost"],
    trailing_stop=CONFIG["risk"]["trailing_stop"],
    drawdown_stop=CONFIG["risk"]["drawdown_stop"],
    recovery_threshold=CONFIG["risk"]["recovery_threshold"],
    cash_reserve_pct=CONFIG["risk"]["cash_reserve_pct"],
    spy_reserve=CONFIG["risk"]["spy_reserve"],
    stop_cooldown_days=CONFIG["risk"]["stop_cooldown_days"],
    vix_threshold=CONFIG["risk"]["vix_threshold"],
)

RUNS = [
    {"label": "Chronos-mini  (Amazon)", "kronos_model_size": "mini",  "kronos_model_source": "chronos"},
    {"label": "Kronos-base   (shiyu-coder)", "kronos_model_size": "base", "kronos_model_source": "sy-kronos"},
]


def _run(cfg: dict) -> dict:
    label = cfg.pop("label")
    bt = Backtester(**{**COMMON, **cfg})
    bt.run()
    s = bt.summary()
    s["label"] = label
    return s


def main():
    print("Running backbone comparison — this will download models on first run (~500MB total).\n")

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run, dict(r)): r["label"] for r in RUNS}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                results[label] = fut.result()
                print(f"  ✓ {label} done")
            except Exception as e:
                print(f"  ✗ {label} failed: {e}")
                results[label] = {"label": label, "error": str(e)}

    # ── Print comparison table ────────────────────────────────────────────────
    METRICS = [
        ("Total return",        "total_return",         "{:+.2%}"),
        ("Ann. return",         "annualized_return",    "{:+.2%}"),
        ("Sharpe ratio",        "sharpe_ratio",         "{:.2f}"),
        ("Max drawdown",        "max_drawdown",         "{:.2%}"),
        ("Calmar ratio",        "calmar_ratio",         "{:.2f}"),
        ("Win rate",            "win_rate",             "{:.1%}"),
        ("SPY return (same period)", "benchmark_return", "{:+.2%}"),
    ]

    col_w = 28
    header = f"{'Metric':<26}" + "".join(r["label"][:col_w].rjust(col_w) for r in RUNS)
    print("\n" + "=" * (26 + col_w * len(RUNS)))
    print(header)
    print("-" * (26 + col_w * len(RUNS)))

    for display, key, fmt in METRICS:
        row = f"{display:<26}"
        for r in RUNS:
            label = r["label"]
            s = results.get(label, {})
            if "error" in s:
                row += "ERROR".rjust(col_w)
            elif key not in s:
                row += "—".rjust(col_w)
            else:
                row += fmt.format(s[key]).rjust(col_w)
        print(row)

    print("=" * (26 + col_w * len(RUNS)))
    print()


if __name__ == "__main__":
    main()
