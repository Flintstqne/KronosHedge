"""
Daily scheduler: runs the full cycle at US market open (09:31 ET) Mon–Fri.
Runs indefinitely until killed.

Usage:
  python scripts/scheduler.py                  # live paper trading
  python scripts/scheduler.py --dry-run        # log everything, no orders
  python scripts/scheduler.py --time 09:31     # override fire time (ET)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import schedule
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = structlog.get_logger()
ET = ZoneInfo("America/New_York")


def run_cycle(dry_run: bool = False) -> None:
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        log.info("skip_weekend", date=now.date().isoformat())
        return

    log.info("cycle_firing", time=now.isoformat())
    cmd = [sys.executable, "main.py"]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("cycle_failed", returncode=result.returncode)
    else:
        log.info("cycle_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--time", default="09:31", help="Fire time HH:MM Eastern")
    parser.add_argument("--run-now", action="store_true", help="Fire once immediately then schedule")
    args = parser.parse_args()

    log.info("scheduler_start", fire_time=args.time, dry_run=args.dry_run)

    schedule.every().day.at(args.time).do(run_cycle, dry_run=args.dry_run)

    if args.run_now:
        log.info("running_immediately")
        run_cycle(dry_run=args.dry_run)

    print(f"Scheduler running. Next fire: {args.time} ET (weekdays only). Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
