"""Unit tests for walk-forward validation."""
from datetime import datetime

import numpy as np
import pandas as pd

from quant.research.walkforward import (
    ParameterSweep,
    WalkForwardConfig,
    WalkForwardValidator,
)


def create_sample_data(symbols: list[str], days: int = 500) -> dict:
    """Create sample multi-symbol data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=days, freq="B")

    data = {}
    for i, sym in enumerate(symbols):
        # Different trends for each symbol
        drift = 0.0002 * (i + 1)
        vol = 0.01 * (1 + i * 0.2)
        returns = np.random.normal(drift, vol, days)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
            'close': prices,
            'volume': np.random.lognormal(13, 0.5, days).astype(int),
        }, index=dates)
        data[sym] = df

    return data


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, params: dict):
        self.params = params
        self.lookback = params.get('lookback', 20)

    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> pd.Series:
        # Simple momentum signal
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])

        signals = {}
        for sym in symbols:
            close_col = (sym, 'close')
            if close_col in data.columns:
                # Just return constant signal for testing
                signals[sym] = 1.0

        return pd.Series(signals)


def test_walkforward_config():
    """Test walk-forward config."""
    config = WalkForwardConfig(
        train_window=252,
        validation_window=63,
        test_window=63,
        step=63,
    )

    assert config.train_window == 252
    assert config.validation_window == 63
    assert config.step == 63


def test_parameter_sweep():
    """Test parameter sweep."""
    create_sample_data(['AAPL', 'MSFT'], 200)

    from quant.backtest.engine import BacktestConfig
    from quant.backtest.execution import ExecutionConfig

    bt_config = BacktestConfig(
        initial_capital=100000,
        execution=ExecutionConfig(commission_bps=0, spread_bps=0, slippage_bps=0),
    )

    sweep = ParameterSweep(
        param_grid={'lookback': [10, 20, 30]},
        backtest_config=bt_config,
    )

    # This would run full backtests - just test initialization
    assert sweep.param_grid == {'lookback': [10, 20, 30]}


def test_walkforward_fold_generation():
    """Test fold generation logic."""
    config = WalkForwardConfig(
        train_window=100,
        validation_window=20,
        test_window=20,
        step=20,
        expanding=False,
    )

    # Create index of 200 days
    index = pd.date_range("2020-01-01", periods=200, freq="B")

    validator = WalkForwardValidator(
        config=config,
        backtest_config=None,
        param_grid={},
    )

    folds = validator._generate_folds(index)

    # Should have multiple folds
    assert len(folds) > 0

    # Check fold structure
    for fold in folds:
        train_start, train_end, val_start, val_end, test_start, test_end = fold
        assert train_start < train_end
        assert train_end < val_start
        assert val_end < test_start
        assert test_start < test_end


def test_walkforward_expanding():
    """Test expanding window walk-forward."""
    config = WalkForwardConfig(
        train_window=100,
        validation_window=20,
        test_window=20,
        step=20,
        expanding=True,
    )

    index = pd.date_range("2020-01-01", periods=200, freq="B")

    validator = WalkForwardValidator(
        config=config,
        backtest_config=None,
        param_grid={},
    )

    folds = validator._generate_folds(index)

    # First fold train should start at 0
    assert folds[0][0] == index[0]

    # Each fold train window expands
    for i in range(1, len(folds)):
        assert folds[i][0] == index[0]  # Always start from beginning
        assert folds[i][1] > folds[i-1][1]  # Train end increases


def test_parameter_sweep_grid():
    """Test parameter sweep grid generation."""
    sweep = ParameterSweep(
        param_grid={'param1': [1, 2], 'param2': ['a', 'b']},
        backtest_config=None,
    )

    # Test grid iteration
    import itertools
    combos = list(itertools.product(*sweep.param_grid.values()))
    assert len(combos) == 4
    assert (1, 'a') in combos
    assert (2, 'b') in combos


def test_scoring_function():
    """Test custom scoring function."""
    def custom_score(metrics):
        return metrics.get('custom_metric', 0)

    sweep = ParameterSweep(
        param_grid={'x': [1, 2]},
        backtest_config=None,
        scoring=custom_score,
    )

    assert sweep.scoring({'custom_metric': 5}) == 5
