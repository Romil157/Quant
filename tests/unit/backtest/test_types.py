"""Unit tests for backtest types."""
from datetime import datetime

from quant.backtest.types import (
    Account,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    Trade,
)


def test_order_creation():
    """Test order creation."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.quantity == 100
    assert order.order_type == OrderType.MARKET
    assert order.status == OrderStatus.PENDING
    assert order.order_id is not None


def test_order_remaining_quantity():
    """Test order remaining quantity."""
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
    assert order.remaining_quantity == 100

    order.filled_quantity = 30
    assert order.remaining_quantity == 70

    order.filled_quantity = 100
    assert order.remaining_quantity == 0
    assert order.is_complete


def test_fill_properties():
    """Test fill properties."""
    fill = Fill(
        order_id="ord_123",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=150.0,
        timestamp=datetime.now(),
        commission=1.0,
    )
    assert fill.value == 15000.0
    assert fill.net_value == 14999.0


def test_position():
    """Test position."""
    pos = Position(symbol="AAPL", quantity=100, avg_price=150.0)
    assert pos.is_long
    assert not pos.is_short
    assert not pos.is_flat

    pos_short = Position(symbol="AAPL", quantity=-100, avg_price=150.0)
    assert pos_short.is_short

    pos_flat = Position(symbol="AAPL", quantity=0)
    assert pos_flat.is_flat


def test_portfolio():
    """Test portfolio."""
    portfolio = Portfolio(cash=100000, initial_capital=100000)
    assert portfolio.total_value == 100000

    # Buy 100 shares at 150
    portfolio.cash -= 100 * 150  # Reduce cash for purchase
    portfolio.update_position("AAPL", 100, 150.0)
    portfolio.mark_to_market({"AAPL": 155.0})

    assert portfolio.positions["AAPL"].quantity == 100
    assert portfolio.positions["AAPL"].market_value == 15500.0
    # total_value = cash (85000) + market_value (15500) = 100500
    assert portfolio.total_value == 100500
    assert portfolio.gross_exposure == 15500.0
    assert portfolio.net_exposure == 15500.0


def test_portfolio_flip():
    """Test portfolio position flip."""
    portfolio = Portfolio(cash=100000, initial_capital=100000)

    # Buy 100
    portfolio.update_position("AAPL", 100, 150.0)
    assert portfolio.positions["AAPL"].quantity == 100

    # Sell 150 (flip to short 50)
    portfolio.update_position("AAPL", -150, 155.0)
    assert portfolio.positions["AAPL"].quantity == -50
    assert portfolio.positions["AAPL"].realized_pnl == 100 * (155 - 150)  # 500


def test_account():
    """Test account snapshot."""
    pos = Position(symbol="AAPL", quantity=100, avg_price=150.0)
    pos.market_value = 15500.0
    pos.unrealized_pnl = 500.0

    account = Account(
        timestamp=datetime.now(),
        cash=85000.0,
        positions={"AAPL": pos},
        total_value=100500.0,
        gross_exposure=15500.0,
        net_exposure=15500.0,
    )
    assert account.total_value == 100500.0


def test_trade():
    """Test trade record."""
    trade = Trade(
        symbol="AAPL",
        entry_time=datetime(2023, 1, 1),
        exit_time=datetime(2023, 1, 15),
        side=OrderSide.BUY,
        quantity=100,
        entry_price=150.0,
        exit_price=155.0,
        pnl=500.0,
        commission=2.0,
        return_pct=0.0333,
        holding_period=14.0,
    )
    assert trade.pnl == 500.0
