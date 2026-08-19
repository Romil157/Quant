"""Abstract broker adapter interface for live and paper execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quant.backtest.types import Order, Position


class BrokerAdapter(ABC):
    """Abstract base class for all broker execution backends."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker API / websocket."""
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to broker."""
        raise NotImplementedError

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker."""
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        raise NotImplementedError

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        """Fetch all currently open positions."""
        raise NotImplementedError

    @abstractmethod
    async def get_account(self) -> dict[str, Any]:
        """Fetch account balance, buying power, and equity."""
        raise NotImplementedError

    @abstractmethod
    async def stream_quotes(self, symbols: list[str]) -> Any:
        """Stream real-time quotes or bars."""
        raise NotImplementedError
