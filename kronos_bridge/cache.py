"""Disk-based prediction cache keyed by (ticker, last_date, horizon)."""

import json
import hashlib
from pathlib import Path
from typing import Optional

from .signal import ForecastSignal


class PredictionCache:
    def __init__(self, cache_dir: str = "./data/kronos_cache"):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, ticker: str, last_date: str, horizon: int) -> Path:
        raw = f"{ticker}|{last_date}|{horizon}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return self._dir / f"{ticker}_{h}.json"

    def get(self, ticker: str, last_date: str, horizon: int) -> Optional[ForecastSignal]:
        path = self._key(ticker, last_date, horizon)
        if path.exists():
            return ForecastSignal.model_validate_json(path.read_text())
        return None

    def set(self, signal: ForecastSignal, last_date: str) -> None:
        path = self._key(signal.ticker, last_date, signal.horizon)
        path.write_text(signal.model_dump_json(indent=2))

    def clear(self) -> None:
        for f in self._dir.glob("*.json"):
            f.unlink()
