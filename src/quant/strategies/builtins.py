"""Built-in strategies for backtesting."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant.backtest.engine import Strategy


class MomentumStrategy(Strategy):
    """Time-series momentum strategy."""

    def __init__(
        self,
        lookback: int = 126,
        rebalance_freq: int = 21,
        top_n: int = 10,
    ):
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.last_rebalance: datetime | None = None
        self.bar_count = 0

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        self.bar_count += 1

        # Check rebalance frequency
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        for col in data.columns:
            if isinstance(col, tuple):
                symbol, field = col
                if field == 'close':
                    # We need historical data - use a rolling calculation
                    # For simplicity, return signal based on current momentum
                    pass

        # Simplified: just return random signals for demo
        # In practice, this would use the full historical data
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])

        if not symbols:
            return pd.Series(dtype=float)

        # Generate signals (placeholder)
        self.last_rebalance = current_time
        return pd.Series(dict.fromkeys(list(symbols)[:self.top_n], 1.0))


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using RSI and Bollinger Bands."""

    def __init__(
        self,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_window: int = 20,
        bb_std: float = 2.0,
    ):
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_window = bb_window
        self.bb_std = bb_std

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        # Placeholder implementation
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])

        if not symbols:
            return pd.Series(dtype=float)

        return pd.Series(dict.fromkeys(symbols, 0.0))


class BreakoutStrategy(Strategy):
    """Breakout strategy using Donchian channels."""

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
    ):
        self.entry_window = entry_window
        self.exit_window = exit_window

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])

        if not symbols:
            return pd.Series(dtype=float)

        return pd.Series(dict.fromkeys(symbols, 0.0))


class PairTradingStrategy(Strategy):
    """Statistical arbitrage pair trading."""

    def __init__(
        self,
        lookback: int = 63,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        return pd.Series(dtype=float)
