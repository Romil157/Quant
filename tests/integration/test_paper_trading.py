"""Integration tests for paper trading engine."""
import asyncio
from datetime import datetime

import pandas as pd
import pytest

from quant.backtest.engine import Strategy
from quant.paper.engine import MockDataFeed, PaperConfig, PaperEngine


class MockPaperStrategy(Strategy):
    """Simple strategy for paper trading tests."""

    def __init__(self):
        self.call_count = 0

    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> pd.Series:
        self.call_count += 1
        if self.call_count % 2 == 1:
            return pd.Series({"SPY": 0.5, "QQQ": 0.5})
        else:
            return pd.Series({"SPY": 0.3, "QQQ": 0.7})


@pytest.fixture
def temp_paper_db(tmp_path):
    return str(tmp_path / "paper_test.db")


def test_paper_engine_initialization_and_warmup(temp_paper_db):
    """Test paper engine initialization with mock data feed."""
    config = PaperConfig(
        enabled=True,
        symbols=["SPY", "QQQ"],
        initial_capital=100_000,
        state_db_path=temp_paper_db,
        data_provider="mock",
    )

    engine = PaperEngine(config)
    assert engine.portfolio.cash == 100_000
    assert engine.state.get_cash() == 100_000
    assert len(engine.portfolio.positions) == 0


def test_paper_engine_checkpoint_and_restart(temp_paper_db):
    """Test state persistence across engine restarts."""
    config = PaperConfig(
        enabled=True,
        symbols=["SPY", "QQQ"],
        initial_capital=100_000,
        state_db_path=temp_paper_db,
        data_provider="mock",
    )

    # First session: modify portfolio & checkpoint
    engine1 = PaperEngine(config)
    engine1.portfolio.cash = 60_000
    engine1.current_prices["SPY"] = 400.0
    engine1.portfolio.get_position("SPY").quantity = 100.0
    engine1.portfolio.get_position("SPY").avg_price = 400.0
    engine1.portfolio.get_position("SPY").market_value = 40_000.0
    engine1._checkpoint()

    # Second session: new engine instance with same DB
    engine2 = PaperEngine(config)
    assert engine2.portfolio.cash == 60_000
    assert "SPY" in engine2.portfolio.positions
    assert engine2.portfolio.positions["SPY"].quantity == 100.0
    assert engine2.portfolio.positions["SPY"].avg_price == 400.0


def test_mock_data_feed_bars():
    """Test mock data feed bar generation."""
    async def _test():
        feed = MockDataFeed(symbols=["SPY", "QQQ"])
        bar_spy = await feed.get_latest_bar("SPY")
        assert bar_spy is not None
        assert "open" in bar_spy
        assert "high" in bar_spy
        assert "low" in bar_spy
        assert "close" in bar_spy
        assert "volume" in bar_spy
        assert bar_spy["close"] > 0

    asyncio.run(_test())
