"""Integration test for full walk-forward research pipeline."""
from datetime import datetime

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, Strategy
from quant.backtest.execution import ExecutionConfig
from quant.research.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
)


class SimpleMovingAverageStrategy(Strategy):
    """Test strategy with tunable lookback parameter."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame, current_time: datetime) -> pd.Series:
        signals = {}
        for col in data.columns:
            if isinstance(col, tuple):
                sym, field = col
                if field == "close":
                    series = data[col].dropna()
                    if len(series) >= self.lookback:
                        sma = series.iloc[-self.lookback:].mean()
                        curr = series.iloc[-1]
                        signals[sym] = 1.0 if curr > sma else -1.0
                    else:
                        signals[sym] = 0.0
        return pd.Series(signals)


def make_test_market_data(symbols: list[str], days: int = 400) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-01", periods=days, freq="B")
    data = {}
    for i, sym in enumerate(symbols):
        np.random.seed(42 + i)
        drift = 0.0003
        vol = 0.012
        returns = np.random.normal(drift, vol, days)
        prices = 100 * np.exp(np.cumsum(returns))
        data[sym] = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.004,
                "low": prices * 0.996,
                "close": prices,
                "volume": np.random.randint(10000, 50000, days),
            },
            index=dates,
        )
    return data


def test_walkforward_pipeline_end_to_end():
    """Test full walk-forward validation pipeline with parameter sweep and OOS metrics."""
    symbols = ["AAPL", "MSFT"]
    data = make_test_market_data(symbols, days=350)

    config = WalkForwardConfig(
        train_window=120,
        validation_window=30,
        test_window=30,
        step=30,
        min_train_size=50,
    )

    bt_config = BacktestConfig(
        initial_capital=100_000,
        execution=ExecutionConfig(commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0),
    )

    validator = WalkForwardValidator(
        config=config,
        backtest_config=bt_config,
        param_grid={"lookback": [10, 20, 30]},
    )

    def strategy_factory(params: dict):
        return SimpleMovingAverageStrategy(lookback=params.get("lookback", 20))

    result = validator.validate(
        data=data,
        strategy_factory=strategy_factory,
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) > 0
    assert "mean_sharpe" in result.aggregate_metrics
    assert "lookback" in result.parameter_stability
