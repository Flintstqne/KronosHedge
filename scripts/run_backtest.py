"""
Run a historical backtest and print summary stats.

Usage:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --start 2023-01-01 --end 2024-01-01
  python scripts/run_backtest.py --agents          # include LLM agents (slow + costs money)
  python scripts/run_backtest.py --write-logs      # write audit logs (feeds dashboard)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--cash", type=float, default=100_000.0)
    parser.add_argument("--agents", action="store_true", help="Run LLM agent pipeline")
    parser.add_argument("--write-logs", action="store_true", help="Write audit logs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from qlib_pipeline.backtest import Backtester
    from monitoring.logger import AuditLogger

    audit_logger = AuditLogger() if args.write_logs else None

    bt = Backtester(
        tickers=cfg["universe"]["tickers"],
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        lookback_days=cfg["kronos"]["lookback_days"],
        kronos_model_size=cfg["kronos"]["model_size"],
        kronos_horizon=cfg["kronos"]["horizon"],
        kronos_device=cfg["kronos"]["device"],
        qlib_weight=cfg["reconciliation"]["qlib_weight"],
        agent_weight=cfg["reconciliation"]["agent_weight"],
        llm_provider=cfg["agents"]["llm_provider"],
        llm_model=cfg["agents"]["llm_model"],
        initial_cash=args.cash,
        run_agents=args.agents,
        audit_logger=audit_logger,
    )

    print(f"\nRunning backtest {args.start} → {args.end}  "
          f"({'agents ON' if args.agents else 'Kronos+Qlib only'})\n")

    results = bt.run()
    stats = bt.summary()

    print("=" * 48)
    print(f"  Total return   : {stats.get('total_return', 0):+.2%}")
    print(f"  Sharpe ratio   : {stats.get('sharpe_ratio', 0):.2f}")
    print(f"  Max drawdown   : {stats.get('max_drawdown', 0):.2%}")
    print(f"  Win rate       : {stats.get('win_rate', 0):.0%}")
    print(f"  Final equity   : ${stats.get('final_equity', 0):,.2f}")
    print(f"  Trading days   : {stats.get('trading_days', 0)}")
    print("=" * 48)

    out = Path("logs/backtest_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nEquity curve saved to {out}")
    if args.write_logs:
        print("Audit logs written to ./logs/audit — open dashboard to visualize.")


if __name__ == "__main__":
    main()
