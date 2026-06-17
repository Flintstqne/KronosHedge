"""
SignalReconciler: blends Qlib quantitative weights with agent portfolio decisions.
"""

from __future__ import annotations

from reconciliation.sector_filter import apply_sector_filter, SECTOR_MAP


class SignalReconciler:
    def __init__(
        self,
        qlib_weight: float = 0.60,
        agent_weight: float = 0.40,
        sector_filter: bool = True,
        max_sector_pct: float = 0.40,
        max_per_sector: int = 4,
    ):
        assert abs(qlib_weight + agent_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.qlib_weight    = qlib_weight
        self.agent_weight   = agent_weight
        self.sector_filter  = sector_filter
        self.max_sector_pct = max_sector_pct
        self.max_per_sector = max_per_sector

    def reconcile(
        self,
        qlib_weights: dict[str, float],
        agent_decisions: dict[str, dict],
    ) -> dict[str, float]:
        """
        qlib_weights: {ticker: portfolio_weight}  - from KronosAlphaFactor
        agent_decisions: {ticker: {action, quantity_pct, ...}}  - from portfolio_manager_agent
        Returns: final target weight per ticker (sums to <= 1.0, all >= 0).
        """
        tickers = set(qlib_weights) | set(agent_decisions)
        agent_weights = self._decisions_to_weights(agent_decisions)

        final: dict[str, float] = {}
        for ticker in tickers:
            q = qlib_weights.get(ticker, 0.0)
            a = agent_weights.get(ticker, 0.0)
            final[ticker] = self.qlib_weight * q + self.agent_weight * a

        normalized = self._normalize(final)

        if self.sector_filter:
            normalized = apply_sector_filter(
                normalized,
                sector_map=SECTOR_MAP,
                max_sector_pct=self.max_sector_pct,
                max_per_sector=self.max_per_sector,
            )

        return normalized

    def _decisions_to_weights(self, decisions: dict[str, dict]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for ticker, dec in decisions.items():
            action = dec.get("action", "HOLD")
            qty_pct = float(dec.get("quantity_pct", 0.0))
            if action == "BUY":
                weights[ticker] = qty_pct
            elif action == "SELL":
                weights[ticker] = -qty_pct  # short signal
            else:
                weights[ticker] = 0.0
        return weights

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        # Long-only: zero out shorts, normalize longs to sum ≤ 1
        long_only = {t: max(w, 0.0) for t, w in weights.items()}
        total = sum(long_only.values())
        if total < 1e-10:
            return long_only
        return {t: w / total for t, w in long_only.items()}
