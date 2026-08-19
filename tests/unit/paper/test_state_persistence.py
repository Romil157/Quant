"""Unit tests for paper trading state persistence using SQLite."""
from datetime import datetime
from pathlib import Path

import pytest

from quant.backtest.types import Fill, Order, OrderSide, OrderStatus, OrderType
from quant.paper.engine import PaperState


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path for tests."""
    db_path = tmp_path / "paper_state.db"
    return str(db_path)


def test_initial_state_creation(temp_db):
    """Test that PaperState initializes with default cash and empty positions."""
    state = PaperState(db_path=temp_db, initial_capital=150_000.0)

    assert state.get_cash() == 150_000.0
    assert state.get_all_positions() == {}


def test_cash_update_and_persistence(temp_db):
    """Test cash updates persist across instances."""
    state1 = PaperState(db_path=temp_db, initial_capital=100_000.0)
    state1.set_cash(85_450.50)
    assert state1.get_cash() == 85_450.50

    # Create new instance pointing to same SQLite DB
    state2 = PaperState(db_path=temp_db)
    assert state2.get_cash() == 85_450.50


def test_position_upsert_and_removal(temp_db):
    """Test inserting, updating, and removing positions."""
    state = PaperState(db_path=temp_db)

    # Insert position
    state.set_position("AAPL", quantity=100.0, avg_price=150.0, market_value=15000.0, unrealized_pnl=500.0, realized_pnl=0.0)
    pos = state.get_position("AAPL")
    assert pos is not None
    assert pos["symbol"] == "AAPL"
    assert pos["quantity"] == 100.0
    assert pos["avg_price"] == 150.0

    # Update position with additional shares
    state.set_position("AAPL", quantity=150.0, avg_price=155.0, market_value=23250.0, unrealized_pnl=750.0, realized_pnl=0.0)
    pos_updated = state.get_position("AAPL")
    assert pos_updated["quantity"] == 150.0
    assert pos_updated["avg_price"] == 155.0

    # Add second position
    state.set_position("MSFT", quantity=50.0, avg_price=300.0, market_value=15000.0, unrealized_pnl=-200.0, realized_pnl=0.0)
    all_pos = state.get_all_positions()
    assert len(all_pos) == 2
    assert "AAPL" in all_pos
    assert "MSFT" in all_pos

    # Delete position
    state.delete_position("AAPL")
    assert state.get_position("AAPL") is None
    assert len(state.get_all_positions()) == 1


def test_order_and_fill_recording(temp_db):
    """Test order status tracking and fill recording."""
    state = PaperState(db_path=temp_db)

    order = Order(
        order_id="ord_test_001",
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=50.0,
        order_type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        timestamp=datetime(2023, 1, 1, 10, 0),
    )
    state.save_order(order)

    fill = Fill(
        fill_id="fill_001",
        order_id=order.order_id,
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=50.0,
        price=450.0,
        commission=1.50,
        timestamp=datetime(2023, 1, 1, 10, 1),
    )
    state.save_fill(fill)

    # Save snapshot
    state.save_snapshot(
        timestamp=datetime(2023, 1, 1, 16, 0),
        cash=77_500.0,
        total_value=100_000.0,
        gross_exposure=0.225,
        net_exposure=0.225,
    )

    # Reopen DB to verify persistence
    state2 = PaperState(db_path=temp_db)
    snapshots = state2.get_snapshots(limit=10)
    assert len(snapshots) == 1
    assert snapshots[0]["cash"] == 77_500.0
