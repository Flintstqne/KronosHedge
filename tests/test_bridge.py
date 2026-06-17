"""Tests for KronosSignalBridge using the naive baseline predictor."""

import pandas as pd
import pytest
from datetime import date, timedelta

from kronos_bridge.signal import KronosSignalBridge, ForecastSignal


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    import numpy as np
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": [1_000_000] * n,
    }, index=dates)


def test_generate_single():
    bridge = KronosSignalBridge(model_size="mini", horizon=5, device="cpu")
    df = _make_ohlcv()
    sig = bridge.generate("AAPL", df)
    assert isinstance(sig, ForecastSignal)
    assert sig.ticker == "AAPL"
    assert sig.horizon == 5
    assert len(sig.predicted_candles) == 5
    assert sig.directional_signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= sig.confidence <= 1.0


def test_generate_batch():
    bridge = KronosSignalBridge(model_size="mini", horizon=3, device="cpu")
    universe = {"AAPL": _make_ohlcv(), "MSFT": _make_ohlcv()}
    signals = bridge.generate_batch(universe)
    assert set(signals.keys()) == {"AAPL", "MSFT"}
    for sig in signals.values():
        assert len(sig.predicted_candles) == 3


def test_signal_serialization():
    bridge = KronosSignalBridge(horizon=5)
    sig = bridge.generate("TEST", _make_ohlcv())
    json_str = sig.model_dump_json()
    restored = ForecastSignal.model_validate_json(json_str)
    assert restored.ticker == sig.ticker
    assert len(restored.predicted_candles) == len(sig.predicted_candles)
