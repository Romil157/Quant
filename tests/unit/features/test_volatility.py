"""Unit tests for volatility features."""
import numpy as np
import pandas as pd

from quant.features.volatility import (
    atr_volatility,
    ewma_volatility,
    garman_klass_volatility,
    parkinson_volatility,
    realized_volatility,
    rogers_satchell_volatility,
    rolling_volatility,
    volatility_cone,
)


def test_realized_volatility():
    # Create returns with known volatility
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    rv = realized_volatility(returns, window=20)
    assert pd.isna(rv.iloc[0])
    assert pd.isna(rv.iloc[18])
    valid = rv.dropna()
    # Should be close to 0.01 * sqrt(20) = 0.0447
    assert all(abs(v - 0.0447) < 0.02 for v in valid)


def test_rolling_volatility():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    rv = rolling_volatility(returns, window=20)
    assert pd.isna(rv.iloc[0])
    valid = rv.dropna()
    assert len(valid) > 0


def test_ewma_volatility():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    rv = ewma_volatility(returns, span=20)
    assert pd.isna(rv.iloc[0])
    valid = rv.dropna()
    assert len(valid) > 0
    assert all(v > 0 for v in valid)


def test_garman_klass_volatility():
    # Create OHLC data
    np.random.seed(42)
    n = 50
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = 100

    gk = garman_klass_volatility(
        pd.Series(open_), pd.Series(high), pd.Series(low), pd.Series(close), window=10
    )
    assert pd.isna(gk.iloc[0])
    valid = gk.dropna()
    assert len(valid) > 0
    assert all(v > 0 for v in valid)


def test_parkinson_volatility():
    np.random.seed(42)
    n = 50
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))

    park = parkinson_volatility(pd.Series(high), pd.Series(low), window=10)
    assert pd.isna(park.iloc[0])
    valid = park.dropna()
    assert len(valid) > 0
    assert all(v > 0 for v in valid)


def test_rogers_satchell_volatility():
    np.random.seed(42)
    n = 50
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = 100

    rs = rogers_satchell_volatility(
        pd.Series(open_), pd.Series(high), pd.Series(low), pd.Series(close), window=10
    )
    assert pd.isna(rs.iloc[0])
    valid = rs.dropna()
    assert len(valid) > 0
    assert all(v > 0 for v in valid)


def test_atr_volatility():
    np.random.seed(42)
    n = 50
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))

    atr_vol = atr_volatility(pd.Series(high), pd.Series(low), pd.Series(close), window=10)
    assert pd.isna(atr_vol.iloc[0])
    valid = atr_vol.dropna()
    assert len(valid) > 0
    assert all(v > 0 for v in valid)


def test_volatility_cone():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    cone = volatility_cone(returns, windows=[5, 10, 20])
    assert isinstance(cone, pd.DataFrame)
    assert list(cone.columns) == ["vol_5", "vol_10", "vol_20"]
    assert len(cone) == len(returns)
