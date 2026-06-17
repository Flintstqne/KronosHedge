#!/usr/bin/env python3
"""
Persistent live runner for Kronos Hedge.

Runs one full cycle per trading day (Kronos inference -> agent signals ->
reconcile -> Alpaca orders -> audit log -> alerts). Retries on failure.
Uses the Alpaca market clock to skip non-trading days including US holidays.

Usage:
    # Run daily at 10:05 AM ET (30 min after open — data is settled by then)
    python scripts/run_live.py

    # Different time, e.g. near close
    python scripts/run_live.py --time 15:45

    # Run immediately right now, then follow the schedule
    python scripts/run_live.py --now

    # No real orders (safe for testing the pipeline)
    python scripts/run_live.py --dry-run --now

Recommended: run inside screen or tmux so it survives terminal disconnects.
    screen -S kronos
    source .venv/bin/activate
    python scripts/run_live.py
    Ctrl+A D   (detach)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ET = ZoneInfo("America/New_York")

# ── Logging setup ─────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run_live.log"),
    ],
)
log = logging.getLogger("run_live")


# ── Market helpers ────────────────────────────────────────────────────────────

def market_is_open_today() -> bool:
    """
    Ask Alpaca whether the market is open today.
    Falls back to weekday check if Alpaca is unavailable.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        clock = client.get_clock()
        # is_open reflects current moment; next_open/next_close tell us about today
        today = datetime.now(ET).date()
        next_open = clock.next_open.astimezone(ET)
        # If next_open is today, the market opens today (we may be before open)
        # If next_open is tomorrow+, today is a non-trading day
        return next_open.date() <= today
    except Exception as exc:
        log.warning(f"Market clock unavailable ({exc}), falling back to weekday check.")
        return datetime.now(ET).weekday() < 5


def seconds_until(hour: int, minute: int) -> float:
    """Seconds until next occurrence of HH:MM ET (today or tomorrow)."""
    now = datetime.now(ET)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


# ── Cycle runner ──────────────────────────────────────────────────────────────

def run_one_cycle(cfg: dict, dry_run: bool) -> bool:
    """Run a single Kronos Hedge cycle. Returns True on success."""
    from main import run_cycle
    try:
        run_cycle(cfg, dry_run=dry_run)
        return True
    except Exception as exc:
        log.error(f"Cycle raised an exception: {exc}", exc_info=True)
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos Hedge persistent runner")
    parser.add_argument(
        "--time", default="10:05",
        help="Daily run time in HH:MM ET (default: 10:05 — 35 min after open)",
    )
    parser.add_argument("--dry-run", action="store_true", help="No real orders")
    parser.add_argument("--now", action="store_true", help="Run immediately then schedule")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Max retry attempts per cycle on failure (default: 3)",
    )
    parser.add_argument(
        "--retry-wait", type=int, default=300,
        help="Seconds to wait between retry attempts (default: 300)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    import yaml
    load_dotenv(ROOT / ".env")
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    run_hour, run_minute = map(int, args.time.split(":"))

    log.info("=" * 60)
    log.info("Kronos Hedge live runner starting")
    log.info(f"  Schedule : {args.time} ET every trading day")
    log.info(f"  Dry-run  : {args.dry_run}")
    log.info(f"  Config   : {args.config}")
    log.info(f"  Tickers  : {cfg['universe']['tickers']}")
    log.info("=" * 60)

    # Optionally fire immediately before entering the schedule loop
    if args.now:
        if market_is_open_today():
            log.info("--now: running cycle immediately.")
            _attempt_cycle(cfg, args.dry_run, args.retries, args.retry_wait)
        else:
            log.info("--now: market is closed today, skipping immediate run.")

    # Main schedule loop
    while True:
        wait = seconds_until(run_hour, run_minute)
        next_run_dt = datetime.now(ET) + timedelta(seconds=wait)
        log.info(f"Next scheduled run: {next_run_dt.strftime('%A %Y-%m-%d %H:%M ET')} "
                 f"({wait / 3600:.1f}h from now)")

        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            log.info("Shutdown requested via Ctrl+C. Exiting.")
            sys.exit(0)

        if not market_is_open_today():
            log.info("Market is closed today (holiday or weekend). Skipping.")
            continue

        _attempt_cycle(cfg, args.dry_run, args.retries, args.retry_wait)


def _attempt_cycle(cfg: dict, dry_run: bool, max_retries: int, retry_wait: int) -> None:
    log.info(">>> CYCLE START")
    for attempt in range(1, max_retries + 1):
        if run_one_cycle(cfg, dry_run):
            log.info(">>> CYCLE COMPLETE")
            return
        if attempt < max_retries:
            log.warning(f"Attempt {attempt}/{max_retries} failed. "
                        f"Retrying in {retry_wait}s.")
            time.sleep(retry_wait)
    log.error(f">>> CYCLE FAILED after {max_retries} attempts.")


if __name__ == "__main__":
    main()
