import sys
from pathlib import Path

# Add project root to sys.path so all packages are importable without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
