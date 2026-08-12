"""Core backtest types and data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    order_id: str | None = None
    timestamp: datetime | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"ord_{id(self)}"
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED) or self.remaining_quantity <= 0


@dataclass
class Fill:
    """Fill/execution representation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fill_id: str | None = None

    def __post_init__(self):
        if self.fill_id is None:
            self.fill_id = f"fill_{id(self)}"

    @property
    def value(self) -> float:
        return self.quantity * self.price

    @property
    def net_value(self) -> float:
        return self.value - self.commission


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    initial_capital: float = 0.0

    def __post_init__(self):
        if self.initial_capital == 0.0:
            self.initial_capital = self.cash

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        pos = self.get_position(symbol)
        if pos.quantity == 0:
            pos.quantity = quantity
            pos.avg_price = price
        elif (pos.quantity > 0 and quantity > 0) or (pos.quantity < 0 and quantity < 0):
            # Same direction - average up/down
            new_qty = pos.quantity + quantity
            pos.avg_price = (pos.quantity * pos.avg_price + quantity * price) / new_qty
            pos.quantity = new_qty
        else:
            # Opposite direction - reduce or flip
            if abs(quantity) >= abs(pos.quantity):
                # Flip or close
                realized = pos.quantity * (price - pos.avg_price)
                pos.realized_pnl += realized
                pos.quantity = quantity + pos.quantity
                pos.avg_price = price if pos.quantity != 0 else 0.0
            else:
                # Partial close
                realized = quantity * (price - pos.avg_price) if pos.quantity > 0 else quantity * (pos.avg_price - price)
                pos.realized_pnl += realized
                pos.quantity += quantity

    def mark_to_market(self, prices: dict[str, float]) -> None:
        for symbol, pos in self.positions.items():
            if symbol in prices and pos.quantity != 0:
                pos.market_value = pos.quantity * prices[symbol]
                pos.unrealized_pnl = pos.quantity * (prices[symbol] - pos.avg_price)


@dataclass
class Account:
    """Account state snapshot."""
    timestamp: datetime
    cash: float
    positions: dict[str, Position]
    total_value: float
    gross_exposure: float
    net_exposure: float
    daily_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    return_pct: float
    holding_period: float  # in days
