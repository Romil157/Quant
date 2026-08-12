"""Signal generation using feature engineering."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quant.backtest.engine import Strategy
from quant.features import (
    atr,
    bollinger_bands,
    breakout_levels,
    macd,
    momentum,
    rsi,
    sma,
    z_score,
)


class MomentumSignalStrategy(Strategy):
    """Time-series momentum with multiple lookback windows."""

    def __init__(
        self,
        lookback_windows: list[int] | None = None,
        rebalance_freq: int = 21,
        top_n: int = 10,
        min_momentum: float = 0.0,
    ):
        self.lookback_windows = lookback_windows or [21, 63, 126, 252]
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.min_momentum = min_momentum
        self.last_rebalance: datetime | None = None
        self.price_history: dict[str, pd.Series] = {}

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        # Extract close prices for each symbol
        symbols = self._extract_symbols(data)
        if not symbols:
            return pd.Series(dtype=float)

        # Update price history (always, regardless of rebalance)
        for symbol in symbols:
            close_col = (symbol, 'close')
            if close_col in data.columns:
                price = data[close_col].iloc[-1]
                if symbol not in self.price_history:
                    self.price_history[symbol] = pd.Series(dtype=float)
                self.price_history[symbol].loc[current_time] = price

        # Check rebalance frequency
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        # Calculate momentum signals
        signals = {}
        for symbol in symbols:
            if symbol not in self.price_history or len(self.price_history[symbol]) < max(self.lookback_windows):
                continue

            prices = self.price_history[symbol]
            mom_scores = []
            for window in self.lookback_windows:
                if len(prices) >= window:
                    mom = momentum(prices, window).iloc[-1]
                    mom_scores.append(mom)

            if mom_scores:
                avg_momentum = np.mean(mom_scores)
                if avg_momentum >= self.min_momentum:
                    signals[symbol] = avg_momentum

        # Select top N by momentum
        top_signals = {}
        if signals:
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            top_signals = dict(sorted_signals[:self.top_n])
            # Normalize to sum to 1
            total = sum(top_signals.values())
            if total > 0:
                top_signals = {k: v/total for k, v in top_signals.items()}

        self.last_rebalance = current_time
        return pd.Series(top_signals)

    def _extract_symbols(self, data: pd.DataFrame) -> list[str]:
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])
        return sorted(symbols)


class MeanReversionSignalStrategy(Strategy):
    """Mean reversion using RSI and Bollinger Bands."""

    def __init__(
        self,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rebalance_freq: int = 5,
        zscore_window: int = 20,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
    ):
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rebalance_freq = rebalance_freq
        self.zscore_window = zscore_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.last_rebalance: datetime | None = None
        self.price_history: dict[str, pd.Series] = {}
        self.positions: dict[str, float] = {}  # Track current positions

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        symbols = self._extract_symbols(data)
        if not symbols:
            return pd.Series(dtype=float)

        signals = {}
        for symbol in symbols:
            close_col = (symbol, 'close')
            if close_col not in data.columns:
                continue

            price = data[close_col].iloc[-1]
            if symbol not in self.price_history:
                self.price_history[symbol] = pd.Series(dtype=float)
            self.price_history[symbol].loc[current_time] = price

            prices = self.price_history[symbol]
            if len(prices) < max(self.rsi_window, self.bb_window, self.zscore_window):
                continue

            # Calculate indicators
            rsi_val = rsi(prices, self.rsi_window).iloc[-1]
            bb_upper, bb_middle, bb_lower = bollinger_bands(prices, self.bb_window, self.bb_std)
            bb_pos = (prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
            zscore = z_score(prices, self.zscore_window).iloc[-1]

            current_pos = self.positions.get(symbol, 0.0)

            # Mean reversion logic
            if current_pos == 0:
                # Entry signals
                if rsi_val < self.rsi_oversold and bb_pos < 0.2 and zscore < -self.zscore_entry:
                    signals[symbol] = 1.0  # Long
                elif rsi_val > self.rsi_overbought and bb_pos > 0.8 and zscore > self.zscore_entry:
                    signals[symbol] = -1.0  # Short
            elif current_pos > 0:
                # Exit long
                if rsi_val > 50 or bb_pos > 0.5 or zscore > -self.zscore_exit:
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = 1.0  # Hold
            elif current_pos < 0:
                # Exit short
                if rsi_val < 50 or bb_pos < 0.5 or zscore < self.zscore_exit:
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = -1.0  # Hold

        self.last_rebalance = current_time
        return pd.Series(signals)

    def _extract_symbols(self, data: pd.DataFrame) -> list[str]:
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])
        return sorted(symbols)

    def update_positions(self, positions: dict[str, float]) -> None:
        """Update tracked positions from portfolio."""
        self.positions = positions


class BreakoutSignalStrategy(Strategy):
    """Breakout strategy using Donchian channels and ATR trailing stops."""

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        atr_window: int = 14,
        atr_mult: float = 2.0,
        rebalance_freq: int = 1,
    ):
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.atr_window = atr_window
        self.atr_mult = atr_mult
        self.rebalance_freq = rebalance_freq
        self.last_rebalance: datetime | None = None
        self.price_history: dict[str, pd.DataFrame] = {}  # Store OHLC
        self.trailing_stops: dict[str, float] = {}
        self.positions: dict[str, float] = {}

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        symbols = self._extract_symbols(data)
        if not symbols:
            return pd.Series(dtype=float)

        signals = {}
        for symbol in symbols:
            ohlc = self._get_ohlc(data, symbol)
            if ohlc is None:
                continue

            if symbol not in self.price_history:
                self.price_history[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close'])

            # Append new bar
            new_bar = pd.DataFrame([ohlc], index=[current_time])
            self.price_history[symbol] = pd.concat([self.price_history[symbol], new_bar])

            hist = self.price_history[symbol]
            if len(hist) < max(self.entry_window, self.exit_window, self.atr_window):
                continue

            # Calculate Donchian channels
            upper_entry, lower_entry = breakout_levels(hist['high'], hist['low'], self.entry_window)
            upper_exit, lower_exit = breakout_levels(hist['high'], hist['low'], self.exit_window)

            # Calculate ATR
            atr_vals = atr(hist['high'], hist['low'], hist['close'], self.atr_window)
            current_atr = atr_vals.iloc[-1]

            current_price = hist['close'].iloc[-1]
            current_pos = self.positions.get(symbol, 0.0)

            if current_pos == 0:
                # Entry: breakout above upper or below lower
                if current_price >= upper_entry.iloc[-1]:
                    signals[symbol] = 1.0
                    self.trailing_stops[symbol] = current_price - self.atr_mult * current_atr
                elif current_price <= lower_entry.iloc[-1]:
                    signals[symbol] = -1.0
                    self.trailing_stops[symbol] = current_price + self.atr_mult * current_atr
            elif current_pos > 0:
                # Long: trail stop
                new_stop = current_price - self.atr_mult * current_atr
                self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), new_stop)

                # Exit: price hits trailing stop or exit channel
                if current_price <= self.trailing_stops[symbol] or current_price <= lower_exit.iloc[-1]:
                    signals[symbol] = 0.0
                    self.trailing_stops.pop(symbol, None)
                else:
                    signals[symbol] = 1.0
            elif current_pos < 0:
                # Short: trail stop
                new_stop = current_price + self.atr_mult * current_atr
                self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, float('inf')), new_stop)

                # Exit
                if current_price >= self.trailing_stops[symbol] or current_price >= upper_exit.iloc[-1]:
                    signals[symbol] = 0.0
                    self.trailing_stops.pop(symbol, None)
                else:
                    signals[symbol] = -1.0

        self.last_rebalance = current_time
        return pd.Series(signals)

    def _extract_symbols(self, data: pd.DataFrame) -> list[str]:
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])
        return sorted(symbols)

    def _get_ohlc(self, data: pd.DataFrame, symbol: str) -> dict | None:
        ohlc = {}
        for field in ['open', 'high', 'low', 'close']:
            col = (symbol, field)
            if col in data.columns:
                ohlc[field] = data[col].iloc[-1]
            else:
                return None
        return ohlc

    def update_positions(self, positions: dict[str, float]) -> None:
        self.positions = positions


class MACDMomentumStrategy(Strategy):
    """MACD-based momentum with trend following."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        trend_window: int = 200,
        rebalance_freq: int = 5,
    ):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.trend_window = trend_window
        self.rebalance_freq = rebalance_freq
        self.last_rebalance: datetime | None = None
        self.price_history: dict[str, pd.Series] = {}

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        symbols = self._extract_symbols(data)
        if not symbols:
            return pd.Series(dtype=float)

        signals = {}
        for symbol in symbols:
            close_col = (symbol, 'close')
            if close_col not in data.columns:
                continue

            price = data[close_col].iloc[-1]
            if symbol not in self.price_history:
                self.price_history[symbol] = pd.Series(dtype=float)
            self.price_history[symbol].loc[current_time] = price

            prices = self.price_history[symbol]
            if len(prices) < max(self.slow, self.trend_window):
                continue

            # MACD
            macd_line, signal_line, histogram = macd(prices, self.fast, self.slow, self.signal)
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]
            hist_val = histogram.iloc[-1]

            # Trend filter
            trend = sma(prices, self.trend_window).iloc[-1]
            in_uptrend = prices.iloc[-1] > trend
            in_downtrend = prices.iloc[-1] < trend

            # MACD crossover signals with trend filter
            if macd_val > signal_val and hist_val > 0 and in_uptrend:
                signals[symbol] = 1.0
            elif macd_val < signal_val and hist_val < 0 and in_downtrend:
                signals[symbol] = -1.0
            else:
                signals[symbol] = 0.0

        self.last_rebalance = current_time
        return pd.Series(signals)

    def _extract_symbols(self, data: pd.DataFrame) -> list[str]:
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])
        return sorted(symbols)


class DualMomentumStrategy(Strategy):
    """Dual momentum: absolute + relative momentum (Gary Antonacci style)."""

    def __init__(
        self,
        lookback: int = 126,
        skip_month: int = 1,
        n_assets: int = 10,
        rebalance_freq: int = 21,
        risk_free_symbol: str = "CASH",
    ):
        self.lookback = lookback
        self.skip_month = skip_month  # Skip most recent month
        self.n_assets = n_assets
        self.rebalance_freq = rebalance_freq
        self.risk_free_symbol = risk_free_symbol
        self.last_rebalance: datetime | None = None
        self.price_history: dict[str, pd.Series] = {}

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_time: datetime,
    ) -> pd.Series:
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.rebalance_freq:
                return pd.Series(dtype=float)

        symbols = self._extract_symbols(data)
        if not symbols:
            return pd.Series(dtype=float)

        # Exclude risk-free asset from ranking
        risky_symbols = [s for s in symbols if s != self.risk_free_symbol]

        # Update price history
        for symbol in risky_symbols:
            close_col = (symbol, 'close')
            if close_col in data.columns:
                price = data[close_col].iloc[-1]
                if symbol not in self.price_history:
                    self.price_history[symbol] = pd.Series(dtype=float)
                self.price_history[symbol].loc[current_time] = price

        # Calculate momentum for each asset
        momentum_scores = {}
        for symbol in risky_symbols:
            if symbol not in self.price_history:
                continue
            prices = self.price_history[symbol]
            if len(prices) < self.lookback + self.skip_month * 21:
                continue

            # Momentum skipping most recent month
            skip_bars = self.skip_month * 21
            total_lookback = self.lookback + skip_bars
            if len(prices) >= total_lookback:
                mom = (prices.iloc[-skip_bars-1] / prices.iloc[-total_lookback]) - 1
                momentum_scores[symbol] = mom

        if not momentum_scores:
            return pd.Series(dtype=float)

        # Rank by momentum
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        # Absolute momentum: only assets with positive momentum
        positive_mom = [(s, m) for s, m in ranked if m > 0]

        if not positive_mom:
            # All negative - go to cash
            signals = {self.risk_free_symbol: 1.0}
        else:
            # Select top N
            selected = positive_mom[:self.n_assets]
            # Equal weight among selected
            weight = 1.0 / len(selected)
            signals = {s: weight for s, _ in selected}

        self.last_rebalance = current_time
        return pd.Series(signals)

    def _extract_symbols(self, data: pd.DataFrame) -> list[str]:
        symbols = set()
        for col in data.columns:
            if isinstance(col, tuple):
                symbols.add(col[0])
        return sorted(symbols)
