from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    notional_usd: float
    order_id: str = ""
    status: str = "pending"
    intended_price: float = 0.0   # close price used for signal
    fill_price: float = 0.0       # actual execution price from broker


@dataclass
class Position:
    ticker: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float


class BrokerAdapter(ABC):
    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def submit_order(self, ticker: str, side: str, notional_usd: float) -> Order: ...

    @abstractmethod
    def cancel_all_orders(self) -> None: ...
