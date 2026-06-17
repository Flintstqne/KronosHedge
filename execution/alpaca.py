"""Alpaca Markets broker adapter. Paper and live trading."""

import os
import uuid

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .base import BrokerAdapter, Order, Position


class AlpacaAdapter(BrokerAdapter):
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        self._client = TradingClient(
            api_key=api_key or os.environ["ALPACA_API_KEY"],
            secret_key=secret_key or os.environ["ALPACA_SECRET_KEY"],
            paper=paper,
        )

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.equity)

    def get_positions(self) -> dict[str, Position]:
        raw = self._client.get_all_positions()
        out: dict[str, Position] = {}
        for p in raw:
            out[p.symbol] = Position(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pl=float(p.unrealized_pl),
            )
        return out

    def submit_order(self, ticker: str, side: str, notional_usd: float) -> Order:
        req = MarketOrderRequest(
            symbol=ticker,
            notional=round(notional_usd, 2),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        resp = self._client.submit_order(req)
        return Order(
            ticker=ticker,
            side=side,
            notional_usd=notional_usd,
            order_id=str(resp.id),
            status=str(resp.status),
        )

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()

    def get_portfolio_history(self, period: str = "1M") -> dict:
        """Returns equity curve dict for dashboard."""
        try:
            hist = self._client.get_portfolio_history(period=period, timeframe="1D")
            return {
                "timestamps": list(hist.timestamp),
                "equity": list(hist.equity),
                "profit_loss": list(hist.profit_loss),
                "profit_loss_pct": list(hist.profit_loss_pct),
            }
        except Exception:
            return {"timestamps": [], "equity": [], "profit_loss": [], "profit_loss_pct": []}
