"""Unit tests for AlpacaBrokerAdapter."""
import asyncio
import pytest

from quant.backtest.types import Order, OrderSide, OrderType
from quant.brokers.alpaca import AlpacaAdapter


def test_alpaca_adapter_connect_and_order_flow():
    """Test connecting and submitting orders through AlpacaAdapter."""
    async def _test():
        adapter = AlpacaAdapter(api_key="test_key", secret_key="test_secret", paper=True)

        connected = await adapter.connect()
        assert connected is True

        order = Order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=10.0,
            order_type=OrderType.MARKET,
            limit_price=450.0,
        )

        submitted = await adapter.submit_order(order)
        assert submitted.status.value == "filled"

        positions = await adapter.get_positions()
        assert "SPY" in positions
        assert positions["SPY"].quantity == 10.0

        account = await adapter.get_account()
        assert account["status"] == "ACTIVE"

        await adapter.disconnect()

    asyncio.run(_test())
