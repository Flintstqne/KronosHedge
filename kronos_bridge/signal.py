"""
KronosSignalBridge: wraps KronosPredictor, produces structured ForecastSignal per ticker.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_VENDOR_PATH = str(Path(__file__).resolve().parents[1] / "vendor" / "sy_kronos")

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class PredictedCandle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class ForecastSignal(BaseModel):
    ticker: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    horizon: int
    predicted_candles: list[PredictedCandle]
    predicted_return: float          # % return over horizon (close[t+N] vs close[t])
    predicted_volatility: float      # std of intra-candle (high-low)/close ranges
    predicted_max_drawdown: float    # worst trough-to-peak within horizon
    directional_signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float                # 0-1, based on inter-candle consistency
    last_close: float                # anchor close used for return calc


def _derive_metrics(
    last_close: float, predicted: pd.DataFrame
) -> tuple[float, float, float, str, float]:
    closes = predicted["close"].values
    predicted_return = (closes[-1] - last_close) / last_close

    ranges = (predicted["high"].values - predicted["low"].values) / predicted["close"].values
    predicted_volatility = float(np.std(ranges))

    # Max drawdown within predicted window
    peak = last_close
    max_dd = 0.0
    for c in closes:
        peak = max(peak, c)
        dd = (c - peak) / peak
        max_dd = min(max_dd, dd)
    predicted_max_drawdown = max_dd

    # Direction + confidence
    if predicted_return > 0.005:
        signal = "BUY"
    elif predicted_return < -0.005:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence: consistency of per-step returns all pointing same direction
    step_returns = np.diff(closes) / closes[:-1]
    agreement = np.sum(np.sign(step_returns) == np.sign(predicted_return))
    confidence = float(agreement / max(len(step_returns), 1))

    return predicted_return, predicted_volatility, predicted_max_drawdown, signal, confidence


class KronosSignalBridge:
    """
    Wraps Kronos KronosPredictor. Loads model lazily on first call.
    Falls back to a naive last-value baseline if Kronos not installed.
    """

    def __init__(
        self,
        model_size: str = "small",
        horizon: int = 5,
        device: str = "cpu",
        model_source: str = "chronos",  # "chronos" | "sy-kronos"
    ):
        self.model_size = model_size
        self.horizon = horizon
        self.device = device
        self.model_source = model_source
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        if self.model_source == "sy-kronos":
            vendor = Path(_VENDOR_PATH)
            if not vendor.exists():
                print(f"[kronos] vendor path missing: {_VENDOR_PATH}", flush=True)
            try:
                self._predictor = _SyKronosPredictor(
                    model_size=self.model_size, device=self.device
                )
                warnings.warn(
                    f"Loaded shiyu-coder Kronos-{self.model_size} as predictor.",
                    stacklevel=2,
                )
                return
            except Exception as e:
                import traceback
                print(f"[kronos] sy-kronos load failed:\n{traceback.format_exc()}", flush=True)
                warnings.warn(f"sy-kronos load failed ({e}), falling back to Chronos.", stacklevel=2)
        # Default: Amazon Chronos
        try:
            self._predictor = _ChronosPredictor(
                model_size=self.model_size, device=self.device
            )
            warnings.warn(
                f"Loaded Amazon Chronos (chronos-t5-{self.model_size}) as predictor.",
                stacklevel=2,
            )
        except Exception as chronos_err:
            warnings.warn(
                f"Chronos unavailable ({chronos_err}). Using naive baseline predictor.",
                stacklevel=2,
            )
            self._predictor = _NaivePredictor()

    def generate(self, ticker: str, df: pd.DataFrame) -> ForecastSignal:
        """Generate ForecastSignal for a single ticker."""
        self._load()
        df = _normalize_columns(df)
        last_close = float(df["close"].iloc[-1])

        raw: pd.DataFrame = self._predictor.predict(df, pred_len=self.horizon)
        raw = _normalize_columns(raw)

        candles = [
            PredictedCandle(
                timestamp=str(idx),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            )
            for idx, row in raw.iterrows()
        ]

        ret, vol, mdd, direction, conf = _derive_metrics(last_close, raw)

        return ForecastSignal(
            ticker=ticker,
            horizon=self.horizon,
            predicted_candles=candles,
            predicted_return=ret,
            predicted_volatility=vol,
            predicted_max_drawdown=mdd,
            directional_signal=direction,
            confidence=conf,
            last_close=last_close,
        )

    def generate_batch(
        self, universe: dict[str, pd.DataFrame]
    ) -> dict[str, ForecastSignal]:
        """Batch inference across full universe. Returns one ForecastSignal per ticker."""
        self._load()
        results: dict[str, ForecastSignal] = {}
        for ticker, df in universe.items():
            try:
                results[ticker] = self.generate(ticker, df)
            except Exception as exc:
                warnings.warn(f"Kronos inference failed for {ticker}: {exc}")
        return results


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "amount" not in df.columns and "close" in df.columns and "volume" in df.columns:
        df["amount"] = df["close"] * df["volume"]
    return df


class _ChronosPredictor:
    """
    Amazon Chronos zero-shot time series foundation model.
    Uses close-price history to forecast future closes; synthesises OHLCV from historical ranges.
    Model is downloaded once (~200 MB for 'small') and cached in ~/.cache/huggingface.
    """

    # Map our model_size names to Chronos HuggingFace model IDs
    _MODEL_MAP = {
        "tiny":   "amazon/chronos-t5-tiny",
        "mini":   "amazon/chronos-t5-mini",
        "small":  "amazon/chronos-t5-small",
        "medium": "amazon/chronos-t5-small",   # no medium; use small
        "large":  "amazon/chronos-t5-large",
    }

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        import torch
        from chronos import ChronosPipeline

        model_id = self._MODEL_MAP.get(model_size, "amazon/chronos-t5-small")
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self._pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
        )
        self._device = device

    def predict(self, df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        import torch

        closes = df["close"].values.astype(float)
        context = torch.tensor(closes, dtype=torch.float32).unsqueeze(0)

        # num_samples=20 gives us a distribution; use median as point forecast
        forecast = self._pipeline.predict(
            inputs=context,
            prediction_length=pred_len,
            num_samples=20,
        )
        # forecast shape: [1, num_samples, pred_len]
        predicted_closes = forecast[0].median(dim=0).values.numpy().astype(float)

        # Synthesise OHLCV from predicted closes + historical intraday range
        lookback = min(20, len(df))
        avg_range_pct = float(
            ((df["high"] - df["low"]) / df["close"]).tail(lookback).mean()
        )
        half_range = avg_range_pct / 2.0
        last_vol = float(df["volume"].iloc[-1])

        rows = []
        prev_close = float(closes[-1])
        for c in predicted_closes:
            rows.append({
                "open":   prev_close,
                "high":   c * (1.0 + half_range),
                "low":    c * (1.0 - half_range),
                "close":  c,
                "volume": last_vol,
                "amount": c * last_vol,
            })
            prev_close = c

        future_dates = pd.date_range(
            start=df.index[-1], periods=pred_len + 1, freq="B"
        )[1:]
        return pd.DataFrame(rows, index=future_dates)


class _SyKronosPredictor:
    """
    Wraps shiyu-coder/Kronos (NeoQuasar/Kronos-{base,small,mini}).
    Financial-native OHLCV model trained on 45+ global exchanges.
    Requires vendor/sy_kronos cloned from github.com/shiyu-coder/Kronos.
    """

    _TOKENIZER = {
        "mini":  "NeoQuasar/Kronos-Tokenizer-2k",
        "small": "NeoQuasar/Kronos-Tokenizer-base",
        "base":  "NeoQuasar/Kronos-Tokenizer-base",
    }
    _MAX_CTX = {"mini": 2048, "small": 512, "base": 512}

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        if _VENDOR_PATH not in sys.path:
            sys.path.insert(0, _VENDOR_PATH)
        from model import Kronos, KronosTokenizer, KronosPredictor as _KP  # type: ignore
        tok = KronosTokenizer.from_pretrained(self._TOKENIZER.get(model_size, "NeoQuasar/Kronos-Tokenizer-base"))
        mdl = Kronos.from_pretrained(f"NeoQuasar/Kronos-{model_size}")
        self._pred = _KP(mdl, tok, max_context=self._MAX_CTX.get(model_size, 512))

    def predict(self, df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        x_df = df[["open", "high", "low", "close"]].copy()
        x_ts = pd.Series(df.index.astype(str) if not hasattr(df.index, "strftime")
                         else df.index, name="timestamps")
        y_ts = pd.Series(
            pd.date_range(start=df.index[-1], periods=pred_len + 1, freq="B")[1:],
            name="timestamps",
        )
        out = self._pred.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1,
        )
        out = out.set_index(y_ts.values) if not isinstance(out.index, pd.DatetimeIndex) else out
        if "volume" not in out.columns:
            out["volume"] = float(df["volume"].iloc[-1])
        if "amount" not in out.columns:
            out["amount"] = out["close"] * out["volume"]
        return out


class _NaivePredictor:
    """Baseline: repeat last candle with tiny Gaussian noise. Used when Kronos unavailable."""

    def predict(self, df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        last = df.iloc[-1].copy()
        rows = []
        for i in range(1, pred_len + 1):
            noise = 1.0 + np.random.normal(0, 0.005)
            rows.append(
                {
                    "open": last["open"] * noise,
                    "high": last["high"] * noise,
                    "low": last["low"] * noise,
                    "close": last["close"] * noise,
                    "volume": last.get("volume", 0),
                    "amount": last.get("amount", 0),
                }
            )
            last = pd.Series(rows[-1])
        future_dates = pd.date_range(
            start=df.index[-1], periods=pred_len + 1, freq="B"
        )[1:]
        return pd.DataFrame(rows, index=future_dates)
