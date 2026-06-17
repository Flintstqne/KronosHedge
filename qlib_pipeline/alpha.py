"""
KronosAlphaFactor: wraps ForecastSignal predicted_return as a portfolio optimizer input.
Also computes a simple equal-weight portfolio from Kronos signals alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kronos_bridge import ForecastSignal


class KronosAlphaFactor:
    """
    Converts a batch of ForecastSignals into a weight vector for portfolio optimization.
    Uses mean-variance-inspired approach: maximize predicted return, penalize predicted vol.
    When price history is available, blends in cross-sectional momentum so the strategy
    retains real signal even while running on NaivePredictor.
    """

    def __init__(
        self,
        momentum_blend: float = 0.80,   # 80% momentum, 20% Kronos
        top_n: int = 10,                # concentrate in top N stocks; 0 = no filter
        short_n: int = 3,               # short bottom-N momentum stocks; 0 = disabled
        pead_boost: float = 0.20,       # score additive for recent EPS beaters
    ):
        self.momentum_blend = momentum_blend
        self.top_n = top_n
        self.short_n = short_n
        self.pead_boost = pead_boost

    def compute_weights(
        self,
        signals: dict[str, ForecastSignal],
        max_position: float = 0.10,
        prices_history: dict[str, pd.DataFrame] | None = None,
        momentum_history: dict[str, pd.DataFrame] | None = None,
        pead_signals: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        prices_history  — 60-day window used for realized-vol sizing and vol-regime scaling.
        momentum_history — 273-day window used for 12M-1M academic momentum signal.
                           Falls back to prices_history when not provided.
        """
        tickers = list(signals.keys())
        if not tickers:
            return {}

        returns = np.array([signals[t].predicted_return for t in tickers])
        confidences = np.array([signals[t].confidence for t in tickers])

        # Realized 20-day vol for position sizing (prices_history provides this)
        if prices_history:
            vols = np.array([
                prices_history[t]["close"].pct_change().tail(20).std() * np.sqrt(252) + 1e-8
                if t in prices_history else signals[t].predicted_volatility + 1e-8
                for t in tickers
            ])
            median_vol = float(np.median(vols))
            if median_vol > 0.30:
                max_position = max_position * 0.70
        else:
            vols = np.array([signals[t].predicted_volatility + 1e-8 for t in tickers])

        # Kronos score = (return / vol) * confidence — Sharpe-like weighting
        kronos_scores = (returns / vols) * confidences

        # Academic 12M-minus-1M cross-sectional momentum.
        # Uses momentum_history (273 days) if available, falls back to prices_history.
        # Skipping the last month avoids short-term mean-reversion in the signal.
        mom_src = momentum_history if momentum_history is not None else prices_history
        if mom_src and self.momentum_blend > 0:
            mom_vals = []
            for t in tickers:
                if t in mom_src:
                    c = mom_src[t]["close"]
                    n = len(c)
                    if n >= 252 and n > 21:
                        # Full 12M-1M: return from 252 days ago to 21 days ago
                        m = float(c.iloc[-21] / c.iloc[-252] - 1)
                    elif n > 21:
                        # Partial: use available history but still skip last month
                        m = float(c.iloc[-21] / c.iloc[0] - 1)
                    else:
                        m = 0.0
                else:
                    m = 0.0
                mom_vals.append(m)
            mom_arr = np.array(mom_vals)

            n = len(tickers)
            # Cross-sectional rank: top stock = 1.0, bottom = 0.0
            rank_scores = mom_arr.argsort().argsort().astype(float) / max(n - 1, 1)

            # Normalize Kronos scores to [0, 1] for blending
            ks_min, ks_max = kronos_scores.min(), kronos_scores.max()
            kronos_norm = (
                (kronos_scores - ks_min) / (ks_max - ks_min)
                if ks_max > ks_min else np.zeros(n)
            )

            scores = (1.0 - self.momentum_blend) * kronos_norm + self.momentum_blend * rank_scores
        else:
            scores = kronos_scores

        # PEAD boost: add EPS surprise magnitude × pead_boost to score
        if pead_signals and self.pead_boost > 0:
            for i, t in enumerate(tickers):
                if t in pead_signals:
                    scores[i] += pead_signals[t] * self.pead_boost

        # Long only
        scores = np.maximum(scores, 0)

        # Shorts: bottom short_n by score get negative weights (3% max each).
        short_weights: dict[str, float] = {}
        if self.short_n and len(tickers) > self.short_n:
            median_score = float(np.median(scores))
            bottom_idx = np.argsort(scores)[:self.short_n]
            short_max = max_position * 0.30   # 3% per short when max_position=0.10
            for i in bottom_idx:
                if scores[i] < median_score * 0.5:   # only genuinely weak names
                    short_weights[tickers[i]] = -short_max

        # Concentrate: zero out all but the top_n highest-scoring stocks
        if self.top_n and len(tickers) > self.top_n:
            threshold = np.partition(scores, -self.top_n)[-self.top_n]
            scores = np.where(scores >= threshold, scores, 0.0)

        if scores.sum() < 1e-10:
            return {t: 0.0 for t in tickers}

        raw_weights = scores / scores.sum()
        weights = np.minimum(raw_weights, max_position)
        total = weights.sum()
        if total > 0:
            weights = weights / total

        result = {t: float(w) for t, w in zip(tickers, weights)}
        result.update(short_weights)   # merge shorts (negative weights, no overlap with top-N)
        return result

