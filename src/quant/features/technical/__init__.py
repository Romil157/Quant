"""Technical indicator features for quant research."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns: (price_t - price_{t-1}) / price_{t-1}."""
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns: log(price_t / price_{t-1})."""
    return np.log(prices / prices.shift(1))


def rolling_returns(prices: pd.Series, window: int) -> pd.Series:
    """Calculate rolling returns over a window: (price_t / price_{t-window}) - 1."""
    return prices / prices.shift(window) - 1


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=window, min_periods=window).mean()


def ema(prices: pd.Series, window: int, adjust: bool = False) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=window, adjust=adjust, min_periods=window).mean()


def moving_average_distance(prices: pd.Series, window: int, ma_type: str = "sma") -> pd.Series:
    """Distance from moving average: (price - MA) / MA."""
    if ma_type == "sma":
        ma = sma(prices, window)
    elif ma_type == "ema":
        ma = ema(prices, window)
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")
    return (prices - ma) / ma


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """Momentum: price_t / price_{t-window} - 1."""
    return prices / prices.shift(window) - 1


def breakout_levels(
    high: pd.Series,
    low: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Calculate breakout levels: highest high and lowest low over window."""
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    return upper, lower


def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator: MACD line, Signal line, Histogram."""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values


def bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle (SMA), upper, lower."""
    middle = sma(prices, window)
    std = prices.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def z_score(prices: pd.Series, window: int = 20) -> pd.Series:
    """Z-score normalization over rolling window."""
    mean = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    return (prices - mean) / std
