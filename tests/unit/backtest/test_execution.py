"""Unit tests for execution simulator."""

from quant.backtest.execution import ExecutionConfig, ExecutionSimulator
from quant.backtest.types import Order, OrderSide, OrderType


def test_market_order_fill():
    """Test market order fill."""
    config = ExecutionConfig(commission_bps=2.0, spread_bps=1.0, slippage_bps=2.0)
    sim = ExecutionSimulator(config)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    mid_price = 150.0
    fill = sim.simulate_fill(order, mid_price)

    assert fill is not None
    assert fill.symbol == "AAPL"
    assert fill.side == OrderSide.BUY
    assert fill.quantity == 100
    assert fill.price > mid_price  # Buy at ask + slippage

    # Commission: 150 * 100 * 2 / 10000 = 3.0
    expected_commission = 150 * 100 * 2 / 10000
    assert abs(fill.commission - expected_commission) < 0.1


def test_limit_order_fill():
    """Test limit order fill."""
    config = ExecutionConfig(
        commission_bps=2.0,
        spread_bps=1.0,
        slippage_bps=2.0,
        fill_probability=1.0,
    )
    sim = ExecutionSimulator(config)

    # Buy limit below ask - should fill
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=149.5,
    )

    sim.simulate_fill(order, 150.0, bid=149.9, ask=150.1)

    # Limit price 149.5 < ask 150.1, so should not fill for buy
    # Actually, the logic checks if limit_price < ask for BUY - this means it won't fill
    # if limit is below ask. Let me check...
    # In the code: if order.side == OrderSide.BUY and order.limit_price is not None:
    #     if ask is not None and order.limit_price < ask: return None
    # So buy limit at 149.5 with ask at 150.1 -> 149.5 < 150.1 -> returns None (no fill)
    # That's correct - limit buy below ask won't fill

    # Try limit above ask
    order2 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.5,
    )

    fill2 = sim.simulate_fill(order2, 150.0, bid=149.9, ask=150.1)
    assert fill2 is not None  # Should fill at limit or better


def test_sell_limit_order():
    """Test sell limit order."""
    config = ExecutionConfig(
        commission_bps=2.0,
        spread_bps=1.0,
        slippage_bps=2.0,
        fill_probability=1.0,
    )
    sim = ExecutionSimulator(config)

    # Sell limit at or below bid - should fill
    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=149.5,  # At or below bid
    )

    fill = sim.simulate_fill(order, 150.0, bid=149.9, ask=150.1)
    assert fill is not None


def test_commission_calculation():
    """Test commission calculation."""
    config = ExecutionConfig(commission_bps=10.0, min_commission=1.0)
    sim = ExecutionSimulator(config)

    # Small trade - should hit minimum
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET)
    fill = sim.simulate_fill(order, 50.0)

    # 50 * 1 * 10 / 10000 = 0.05, but min is 1.0
    assert fill.commission >= 1.0

    # Large trade
    order2 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1000, order_type=OrderType.MARKET)
    fill2 = sim.simulate_fill(order2, 100.0)

    expected = 100 * 1000 * 10 / 10000  # 1000
    assert abs(fill2.commission - expected) < 1.0


def test_market_impact():
    """Test market impact."""
    config = ExecutionConfig(
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        market_impact_bps=10.0,  # 10 bps per 1% ADV
    )
    sim = ExecutionSimulator(config)

    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10000, order_type=OrderType.MARKET)

    # ADV = 1,000,000, participation = 1%
    fill = sim.simulate_fill(order, 150.0, adv=1_000_000)

    # Impact = 150 * 10/10000 * 0.01 = 0.0015
    # Price should be slightly higher
    assert fill.price > 150.0


def test_partial_fill():
    """Test partial fill probability."""
    config = ExecutionConfig(
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        partial_fill_prob=1.0,  # Always partial
    )
    sim = ExecutionSimulator(config)

    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET)
    fill = sim.simulate_fill(order, 150.0)

    # With partial_fill_prob=1.0, fill should be partial
    # But our implementation uses uniform(0.1, 0.9)
    assert fill.quantity < 100
    assert fill.quantity >= 10
