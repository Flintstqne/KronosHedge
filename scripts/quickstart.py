"""
Kronos Hedge quick-start.
Sets up the venv, seeds demo data, and opens the dashboard.

Usage: python3.11 scripts/quickstart.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PY   = VENV / "bin" / "python"
PIP  = VENV / "bin" / "pip"


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step(msg: str) -> None:
    print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")


def main() -> None:
    os.chdir(ROOT)

    step("1 / 5  Create virtual environment")
    if not VENV.exists():
        run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print("  .venv already exists — skipping.")

    step("2 / 5  Install core dependencies")
    run([str(PIP), "install", "-q",
         "streamlit", "plotly", "pandas", "numpy", "pydantic",
         "yfinance", "pyyaml", "python-dotenv", "structlog",
         "httpx", "diskcache"])

    step("3 / 5  Copy .env template")
    env_file = ROOT / ".env"
    if not env_file.exists():
        (ROOT / ".env.example").read_text()
        import shutil
        shutil.copy(ROOT / ".env.example", env_file)
        print("  .env created — fill in your API keys before running live.")
    else:
        print("  .env already exists — skipping.")

    step("4 / 5  Seed demo data (365 days, ~65% Kronos accuracy)")
    run([str(PY), "scripts/seed_demo_data.py", "--days", "365"])

    step("5 / 5  Launch dashboard")
    print("\n  Dashboard starting at http://localhost:8501")
    print("  Press Ctrl+C to stop.\n")
    try:
        run([str(VENV / "bin" / "streamlit"), "run", "monitoring/dashboard.py",
             "--server.headless", "true"])
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")


if __name__ == "__main__":
    main()
