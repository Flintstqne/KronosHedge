"""
Performance metrics computed from audit log history.
All methods return plain Python dicts/lists for easy Streamlit consumption.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class PerformanceMetrics:
    def __init__(self, audit_records: list[dict]):
        self._records = audit_records

    # ── Equity curve ─────────────────────────────────────────────────────────

    def equity_curve(self) -> pd.DataFrame:
        rows = [
            {"timestamp": r["timestamp"], "equity": r["portfolio_equity"]}
            for r in self._records
            if "portfolio_equity" in r
        ]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "equity"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    # ── Return statistics ─────────────────────────────────────────────────────

    def daily_returns(self) -> pd.Series:
        ec = self.equity_curve()
        if len(ec) < 2:
            return pd.Series(dtype=float)
        return ec["equity"].pct_change().dropna()

    def cumulative_return(self) -> float:
        ec = self.equity_curve()
        if len(ec) < 2:
            return 0.0
        return (ec["equity"].iloc[-1] / ec["equity"].iloc[0]) - 1

    def sharpe_ratio(self, risk_free: float = 0.05) -> float:
        dr = self.daily_returns()
        if len(dr) < 2:
            return 0.0
        excess = dr - risk_free / 252
        std = excess.std()
        if std < 1e-10:
            return 0.0
        return float(excess.mean() / std * math.sqrt(252))

    def max_drawdown(self) -> float:
        ec = self.equity_curve()
        if len(ec) < 2:
            return 0.0
        equity = ec["equity"].values
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(dd.min())

    def win_rate(self) -> float:
        dr = self.daily_returns()
        if len(dr) == 0:
            return 0.0
        return float((dr > 0).sum() / len(dr))

    def summary_stats(self) -> dict[str, Any]:
        return {
            "cumulative_return": self.cumulative_return(),
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "win_rate": self.win_rate(),
            "total_runs": len(self._records),
        }

    # ── Kronos accuracy ───────────────────────────────────────────────────────

    def kronos_accuracy(self) -> pd.DataFrame:
        """
        Compare Kronos directional signal vs actual next-period return.
        Uses per-ticker last_close from consecutive runs for accurate comparison.
        Requires at least 2 runs.
        """
        rows = []
        for i in range(len(self._records) - 1):
            curr = self._records[i]
            nxt = self._records[i + 1]
            for ticker, sig in curr.get("kronos_signals", {}).items():
                predicted_ret = sig.get("predicted_return", 0)
                predicted_dir = sig.get("directional_signal", "HOLD")
                predicted_conf = sig.get("confidence", 0)
                curr_close = sig.get("last_close")

                # Use next run's last_close as the realised price
                nxt_sig = nxt.get("kronos_signals", {}).get(ticker)
                nxt_close = nxt_sig.get("last_close") if nxt_sig else None

                if curr_close and nxt_close and curr_close > 0:
                    actual_ret = (nxt_close - curr_close) / curr_close
                else:
                    actual_ret = None

                correct = None
                if actual_ret is not None:
                    correct = (
                        (predicted_ret > 0 and actual_ret > 0)
                        or (predicted_ret < 0 and actual_ret < 0)
                    )

                rows.append({
                    "timestamp": curr["timestamp"],
                    "ticker": ticker,
                    "predicted_return": predicted_ret,
                    "actual_return": actual_ret,
                    "predicted_direction": predicted_dir,
                    "confidence": predicted_conf,
                    "correct": correct,
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["timestamp", "ticker", "predicted_return", "actual_return",
                     "predicted_direction", "confidence", "correct"]
        )

    # ── Agent signal breakdown ────────────────────────────────────────────────

    def agent_signal_counts(self) -> pd.DataFrame:
        """How often each agent was bullish/bearish/neutral, across all runs."""
        agent_types = [
            "technical_signals", "fundamental_signals",
            "sentiment_signals", "valuation_signals",
        ]
        rows = []
        for record in self._records:
            for agent_type in agent_types:
                for ticker, sig in record.get(agent_type, {}).items():
                    if isinstance(sig, dict):
                        rows.append({
                            "agent": agent_type.replace("_signals", ""),
                            "ticker": ticker,
                            "signal": sig.get("signal", "neutral"),
                        })
            for ticker, investors in record.get("investor_signals", {}).items():
                for investor, sig in investors.items():
                    rows.append({
                        "agent": investor,
                        "ticker": ticker,
                        "signal": sig.get("signal", "neutral"),
                    })

        if not rows:
            return pd.DataFrame(columns=["agent", "ticker", "signal", "count"])

        df = pd.DataFrame(rows)
        return (
            df.groupby(["agent", "signal"])
            .size()
            .reset_index(name="count")
        )

    # ── Signal attribution ────────────────────────────────────────────────────

    def signal_attribution(self) -> pd.DataFrame:
        """
        Per-agent P&L attribution: how much of the portfolio's realized return
        was driven by each signal source.

        For each consecutive run pair, actual ticker return = (close[i+1]-close[i])/close[i].
        Each agent's contribution = final_weight[ticker] * actual_return * vote_direction,
        where vote_direction is +1 (bullish), -1 (bearish), 0 (neutral).
        """
        _vote = {"bullish": 1, "bearish": -1, "neutral": 0}
        agent_types = [
            "technical_signals", "fundamental_signals",
            "sentiment_signals", "valuation_signals", "news_sentiment_signals",
        ]

        rows = []
        for i in range(len(self._records) - 1):
            curr = self._records[i]
            nxt  = self._records[i + 1]
            weights = curr.get("final_weights", {})

            # compute actual returns from kronos last_close
            actual_ret: dict[str, float] = {}
            for ticker in weights:
                curr_sig = curr.get("kronos_signals", {}).get(ticker, {})
                nxt_sig  = nxt.get("kronos_signals",  {}).get(ticker, {})
                c0 = curr_sig.get("last_close") if isinstance(curr_sig, dict) else None
                c1 = nxt_sig.get("last_close")  if isinstance(nxt_sig,  dict) else None
                if c0 and c1 and c0 > 0:
                    actual_ret[ticker] = (c1 - c0) / c0

            for ticker, wt in weights.items():
                ret = actual_ret.get(ticker)
                if ret is None:
                    continue
                # standard signal agents
                for atype in agent_types:
                    sig = curr.get(atype, {}).get(ticker)
                    if isinstance(sig, dict):
                        direction = _vote.get(sig.get("signal", "neutral"), 0)
                        rows.append({
                            "timestamp": curr["timestamp"],
                            "agent": atype.replace("_signals", ""),
                            "ticker": ticker,
                            "weight": wt,
                            "actual_return": ret,
                            "attributed_return": wt * ret * direction,
                        })
                # investor bots
                for investor, sig in curr.get("investor_signals", {}).get(ticker, {}).items():
                    if isinstance(sig, dict):
                        direction = _vote.get(sig.get("signal", "neutral"), 0)
                        rows.append({
                            "timestamp": curr["timestamp"],
                            "agent": investor,
                            "ticker": ticker,
                            "weight": wt,
                            "actual_return": ret,
                            "attributed_return": wt * ret * direction,
                        })

        if not rows:
            return pd.DataFrame(columns=["agent", "ticker", "attributed_return"])

        df = pd.DataFrame(rows)
        return (
            df.groupby("agent")["attributed_return"]
            .sum()
            .reset_index()
            .sort_values("attributed_return", ascending=False)
        )

    # ── Order history ─────────────────────────────────────────────────────────

    def order_history(self) -> pd.DataFrame:
        rows = []
        for record in self._records:
            for order in record.get("orders", []):
                rows.append({
                    "timestamp": record["timestamp"],
                    **order,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["timestamp", "ticker", "side", "notional_usd", "status"]
        )
