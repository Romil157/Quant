"""Unit tests for backtest engine."""
from datetime import datetime

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestEngine, Strategy
from quant.backtest.execution import ExecutionConfig


class SimpleTestStrategy(Strategy):
    """Simple test strategy that buys first symbol."""

    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol
        self.bars = 0

    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> pd.Series:
        self.bars += 1
        # Only trade on first bar
        if self.bars == 1:
            return pd.Series({self.symbol: 1.0})
        return pd.Series(dtype=float)


def create_test_data(symbol: str = "AAPL", days: int = 10) -> pd.DataFrame:
    """Create test OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, days)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
        'close': prices,
        'volume': np.random.lognormal(13, 0.5, days).astype(int),
    }, index=dates)

    return data


def test_backtest_engine_basic():
    """Test basic backtest engine run."""
    # Create test data
    data = {"AAPL": create_test_data("AAPL", 20)}

    # Config
    config = BacktestConfig(
        initial_capital=100000,
        execution=ExecutionConfig(
            commission_bps=2.0,
            spread_bps=1.0,
            slippage_bps=2.0,
        ),
    )

    # Engine
    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    # Run
    results = engine.run(data, strategy)

    assert 'equity_curve' in results
    assert 'returns' in results
    assert 'orders' in results
    assert 'fills' in results
    assert len(results['equity_curve']) == 20
    assert len(results['orders']) >= 1
    assert len(results['fills']) >= 1


def test_backtest_engine_multi_symbol():
    """Test backtest with multiple symbols."""
    data = {
        "AAPL": create_test_data("AAPL", 20),
        "MSFT": create_test_data("MSFT", 20),
    }

    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    results = engine.run(data, strategy)

    assert len(results['equity_curve']) == 20
    assert results['final_equity'] > 0


def test_backtest_portfolio_tracking():
    """Test portfolio tracking during backtest."""
    data = {"AAPL": create_test_data("AAPL", 10)}

    config = BacktestConfig(
        initial_capital=100000,
        execution=ExecutionConfig(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0),
    )

    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    engine.run(data, strategy)

    # Check portfolio was updated
    assert len(engine.portfolio.positions) >= 1
    assert engine.portfolio.total_value > 0
    assert len(engine.account_history) == 10


def test_backtest_risk_limits():
    """Test risk limit enforcement."""
    data = {"AAPL": create_test_data("AAPL", 20)}

    config = BacktestConfig(
        initial_capital=100000,
        max_drawdown=0.05,  # 5% max drawdown
    )

    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    results = engine.run(data, strategy)

    # Should have risk checks
    assert 'max_drawdown_hit' in results


def test_backtest_commission():
    """Test commission is applied."""
    data = {"AAPL": create_test_data("AAPL", 5)}

    config = BacktestConfig(
        initial_capital=100000,
        execution=ExecutionConfig(
            commission_bps=100.0,  # High commission for testing
            spread_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    results = engine.run(data, strategy)

    # Check fills have commission
    for fill in results['fills']:
        assert fill.commission > 0


def test_backtest_daily_returns():
    """Test daily returns calculation."""
    data = {"AAPL": create_test_data("AAPL", 10)}

    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    results = engine.run(data, strategy)

    returns = results['returns']
    assert len(returns) == 9  # N-1 returns for N bars
    assert returns.index.name is None  # Simple integer index


def test_backtest_results_structure():
    """Test results structure."""
    data = {"AAPL": create_test_data("AAPL", 5)}

    config = BacktestConfig(initial_capital=100000)
    engine = BacktestEngine(config)
    strategy = SimpleTestStrategy("AAPL")

    results = engine.run(data, strategy)

    required_keys = [
        'equity_curve', 'returns', 'orders', 'fills',
        'trades', 'account_history', 'final_equity',
        'total_return', 'max_drawdown_hit'
    ]

    for key in required_keys:
        assert key in results, f"Missing key: {key}"
