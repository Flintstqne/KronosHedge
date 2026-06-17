"""Run forward simulation and save results to logs/forward_sim.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.forward_sim import run

if __name__ == "__main__":
    run()
