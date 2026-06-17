"""
Download US equity data for Qlib. Run once before first use.
Usage: python scripts/download_qlib_data.py
"""

import subprocess
import sys


def main():
    print("Downloading Qlib US equity data (~2GB). This may take several minutes.")
    subprocess.run([
        sys.executable, "-m", "qlib.run.get_data",
        "qlib_data",
        "--target_dir", "~/.qlib/qlib_data/us_data",
        "--region", "us",
    ], check=True)
    print("Done. Data stored at ~/.qlib/qlib_data/us_data")


if __name__ == "__main__":
    main()
