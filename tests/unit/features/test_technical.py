"""Unit tests for technical indicators."""
import numpy as np
import pandas as pd

from quant.features.technical import (
    atr,
    bollinger_bands,
    breakout_levels,
    ema,
    log_returns,
    macd,
    momentum,
    moving_average_distance,
    rolling_returns,
    rsi,
    simple_returns,
    sma,
)


def test_simple_returns():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    returns = simple_returns(prices)
    expected = pd.Series([np.nan, 0.02, -0.0098039, 0.0198019])
    pd.testing.assert_series_equal(returns, expected, check_names=False, rtol=1e-4)


def test_log_returns():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0])
    returns = log_returns(prices)
    # log(102/100) = 0.0198026, log(101/102) = -0.009852, log(103/101) = 0.019608
    assert pd.isna(returns.iloc[0])
    assert abs(returns.iloc[1] - 0.0198026) < 1e-4
    assert abs(returns.iloc[2] - (-0.009852)) < 1e-4


def test_rolling_returns():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
    returns = rolling_returns(prices, window=2)
    # (101/100)-1 = 0.01, (103/102)-1 = 0.0098, (105/101)-1 = 0.0396
    assert pd.isna(returns.iloc[0])
    assert pd.isna(returns.iloc[1])
    assert abs(returns.iloc[2] - 0.01) < 1e-4


def test_sma():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
    ma = sma(prices, window=3)
    assert pd.isna(ma.iloc[0])
    assert pd.isna(ma.iloc[1])
    assert abs(ma.iloc[2] - 101.0) < 1e-4  # (100+102+101)/3
    assert abs(ma.iloc[3] - 102.0) < 1e-4  # (102+101+103)/3


def test_ema():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
    ma = ema(prices, window=3)
    assert pd.isna(ma.iloc[0])
    assert pd.isna(ma.iloc[1])
    # EMA values should be reasonable
    assert all(ma.iloc[2:] > 100)
    assert all(ma.iloc[2:] < 106)


def test_moving_average_distance():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
    dist = moving_average_distance(prices, window=3, ma_type="sma")
    # Distance from SMA
    assert pd.isna(dist.iloc[0])
    assert pd.isna(dist.iloc[1])


def test_momentum():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0])
    mom = momentum(prices, window=2)
    # (101/100)-1 = 0.01, (103/102)-1 = 0.0098
    assert pd.isna(mom.iloc[0])
    assert pd.isna(mom.iloc[1])


def test_breakout_levels():
    high = pd.Series([102.0, 103.0, 102.0, 105.0, 106.0])
    low = pd.Series([99.0, 100.0, 98.0, 100.0, 102.0])
    upper, lower = breakout_levels(high, low, window=3)
    assert pd.isna(upper.iloc[0])
    assert pd.isna(upper.iloc[1])
    assert upper.iloc[2] == 103.0  # max of [102, 103, 102]
    assert lower.iloc[2] == 98.0   # min of [99, 100, 98]


def test_macd():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0])
    macd_line, signal_line, histogram = macd(prices, fast=3, slow=6, signal=2)
    # All should have same length
    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    assert len(histogram) == len(prices)
    # First few should be NaN
    assert pd.isna(macd_line.iloc[0])


def test_rsi():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
                        110.0, 108.0, 109.0, 111.0, 112.0])
    rsi_vals = rsi(prices, window=5)
    # RSI should be between 0 and 100
    valid = rsi_vals.dropna()
    assert all(valid >= 0)
    assert all(valid <= 100)


def test_bollinger_bands():
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
                        110.0, 108.0, 109.0, 111.0, 112.0, 113.0, 111.0, 112.0, 114.0, 115.0])
    upper, middle, lower = bollinger_bands(prices, window=5, num_std=2.0)
    # Middle should be SMA
    assert len(upper) == len(prices)
    assert len(middle) == len(prices)
    assert len(lower) == len(prices)
    # Upper > Middle > Lower when valid
    valid_mask = ~upper.isna()
    assert all(upper[valid_mask] > middle[valid_mask])
    assert all(middle[valid_mask] > lower[valid_mask])


def test_atr():
    high = pd.Series([102.0, 103.0, 102.0, 105.0, 106.0, 107.0, 108.0])
    low = pd.Series([99.0, 100.0, 98.0, 100.0, 102.0, 103.0, 104.0])
    close = pd.Series([101.0, 102.0, 100.0, 103.0, 104.0, 105.0, 106.0])
    atr_vals = atr(high, low, close, window=3)
    # Should have values after window period
    assert pd.isna(atr_vals.iloc[0])
    assert pd.isna(atr_vals.iloc[1])
    assert atr_vals.iloc[2] > 0
    assert all(atr_vals.dropna() > 0)
