"""
Portfolio executor: converts final target weights → buy/sell orders.
Computes required trades vs. current positions and submits via broker.
"""

from __future__ import annotations

import structlog

from .base import BrokerAdapter, Order

log = structlog.get_logger()


class PortfolioExecutor:
    def __init__(
        self,
        broker: BrokerAdapter,
        min_order_usd: float = 100.0,
        dry_run: bool = False,
    ):
        self.broker = broker
        self.min_order_usd = min_order_usd
        self.dry_run = dry_run

    def execute(self, target_weights: dict[str, float],
                prices: dict[str, float] | None = None) -> list[Order]:
        equity = self.broker.get_equity()
        current_positions = self.broker.get_positions()

        current_values: dict[str, float] = {
            t: p.market_value for t, p in current_positions.items()
        }

        orders: list[Order] = []

        # Sells first (free up cash)
        for ticker, current_val in current_values.items():
            target_val = equity * target_weights.get(ticker, 0.0)
            delta = target_val - current_val
            if delta < -self.min_order_usd:
                if target_val < self.min_order_usd:
                    # Full exit — close_position avoids fractional qty rounding errors
                    orders.append(self._close(ticker))
                else:
                    # Partial sell — cap at 99.95% of current value for price drift
                    sell_notional = min(abs(delta), current_val * 0.9995)
                    orders.append(self._place("sell", ticker, sell_notional,
                                              intended_price=(prices or {}).get(ticker, 0.0)))

        # Then buys
        for ticker, weight in target_weights.items():
            target_val = equity * weight
            current_val = current_values.get(ticker, 0.0)
            delta = target_val - current_val
            if delta > self.min_order_usd:
                orders.append(self._place("buy", ticker, delta,
                                          intended_price=(prices or {}).get(ticker, 0.0)))

        return orders

    def _close(self, ticker: str) -> Order:
        log.info("order", side="sell", ticker=ticker, notional="close_position",
                 dry_run=self.dry_run)
        if self.dry_run:
            return Order(ticker=ticker, side="sell", notional_usd=0.0,
                         order_id="DRY-RUN", status="simulated")
        return self.broker.close_position(ticker)

    def _place(self, side: str, ticker: str, notional: float,
               intended_price: float = 0.0) -> Order:
        log.info("order", side=side, ticker=ticker, notional=round(notional, 2),
                 dry_run=self.dry_run)
        if self.dry_run:
            return Order(ticker=ticker, side=side, notional_usd=notional,
                         order_id="DRY-RUN", status="simulated",
                         intended_price=intended_price)
        return self.broker.submit_order(ticker, side, notional,
                                        intended_price=intended_price)
