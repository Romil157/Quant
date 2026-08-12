"""Unit tests for data validation module."""
import pandas as pd

from quant.data.validation import (
    ValidationResult,
    validate_data_gaps,
    validate_extreme_outliers,
    validate_no_duplicate_timestamps,
    validate_no_missing_timestamps,
    validate_no_negative_prices,
    validate_no_negative_volume,
    validate_no_zero_prices,
    validate_ohlc_relationships,
    validate_timezone_consistency,
    validate_trading_session_consistency,
)


def test_validate_ohlc_relationships_valid():
    """Test OHLC validation with valid data."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
    })
    result = validate_ohlc_relationships(df)
    assert result.is_valid is True, f"Expected valid, got errors: {result.errors}"
    assert result.metrics["high_violations"] == 0
    assert result.metrics["low_violations"] == 0


def test_validate_ohlc_relationships_high_violation():
    """Test OHLC validation with High violation."""
    df = pd.DataFrame({
        "open": [100.0],
        "high": [99.0],  # violation: high < close
        "low": [98.0],
        "close": [101.0],
    })
    result = validate_ohlc_relationships(df)
    assert result.is_valid is False
    assert result.metrics["high_violations"] == 1


def test_validate_ohlc_relationships_low_violation():
    """Test OHLC validation with Low violation."""
    df = pd.DataFrame({
        "open": [100.0],
        "high": [102.0],
        "low": [103.0],  # violation: low > open
        "close": [101.0],
    })
    result = validate_ohlc_relationships(df)
    assert result.is_valid is False
    assert result.metrics["low_violations"] == 1


def test_validate_no_negative_prices_valid():
    """Test no negative prices validation."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    })
    result = validate_no_negative_prices(df)
    assert result.is_valid is True


def test_validate_no_negative_prices_invalid():
    """Test negative prices validation."""
    df = pd.DataFrame({
        "open": [100.0, -1.0],  # negative
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    })
    result = validate_no_negative_prices(df)
    assert result.is_valid is False
    assert result.metrics["negative_price_rows"] == 1


def test_validate_no_negative_volume_valid():
    """Test no negative volume validation."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    })
    result = validate_no_negative_volume(df)
    assert result.is_valid is True


def test_validate_no_negative_volume_invalid():
    """Test negative volume validation."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, -50],  # negative
    })
    result = validate_no_negative_volume(df)
    assert result.is_valid is False


def test_validate_no_zero_prices_valid():
    """Test no zero prices validation."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    })
    result = validate_no_zero_prices(df)
    assert result.is_valid is True


def test_validate_no_zero_prices_invalid():
    """Test zero prices validation."""
    df = pd.DataFrame({
        "open": [100.0, 0.0],  # zero
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    })
    result = validate_no_zero_prices(df)
    assert result.is_valid is False
    assert result.metrics["zero_price_rows"] == 1


def test_validate_no_duplicate_timestamps_valid():
    """Test no duplicate timestamps."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    }, index=pd.date_range("2020-01-02", periods=2, freq="B"))
    result = validate_no_duplicate_timestamps(df)
    assert result.is_valid is True


def test_validate_no_duplicate_timestamps_invalid():
    """Test duplicate timestamps."""
    df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 99.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1000, 1200, 1300],
    }, index=pd.DatetimeIndex([
        "2020-01-02", "2020-01-02", "2020-01-03"
    ], name="timestamp"))
    result = validate_no_duplicate_timestamps(df)
    assert result.is_valid is False


def test_validate_no_missing_timestamps_valid():
    """Test no missing timestamps."""
    df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 99.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1000, 1200, 1300],
    }, index=pd.date_range("2020-01-02", periods=3, freq="B"))
    result = validate_no_missing_timestamps(df)
    assert result.is_valid is True


def test_validate_timezone_consistency_naive():
    """Test timezone consistency with naive index."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    }, index=pd.date_range("2020-01-02", periods=2, freq="D"))
    result = validate_timezone_consistency(df)
    assert result.metrics["timezone_issues"] == 0


def test_validate_extreme_outliers_no_outliers():
    """Test outlier detection with no outliers."""
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0, 101.5, 102.5],
    })
    result = validate_extreme_outliers(df, z_threshold=3.0)
    assert result.is_valid is True
    assert result.metrics["outlier_count"] == 0


def test_validate_extreme_outliers_with_outliers():
    """Test outlier detection with outliers."""
    # Use data where outlier is obvious - with small samples, std is inflated so use lower threshold
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0, 101.5, 10.0],  # 10 is low outlier
    })
    result = validate_extreme_outliers(df, z_threshold=1.0)
    assert result.is_valid is False
    assert result.metrics["outlier_count"] >= 1


def test_validate_data_gaps_no_gaps():
    """Test data gap detection with no gaps (daily data)."""
    df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 99.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1000, 1200, 1300],
    }, index=pd.date_range("2020-01-02", periods=3, freq="B"))
    result = validate_data_gaps(df, expected_freq="B")
    assert result.is_valid is True


def test_validate_trading_session_consistent():
    """Test trading session consistency with in-session hours."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    }, index=pd.DatetimeIndex([
        "2020-01-02 10:00:00",
        "2020-01-02 14:00:00",
    ], name="timestamp", tz="America/New_York"))
    result = validate_trading_session_consistency(df, market_open=9, market_close=16, tz="America/New_York")
    assert result.is_valid is True


def test_validate_trading_session_out_of_hours():
    """Test trading session consistency with out-of-session hours."""
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1200],
    }, index=pd.DatetimeIndex([
        "2020-01-02 22:00:00",
        "2020-01-03 03:00:00",
    ], name="timestamp", tz="America/New_York"))
    result = validate_trading_session_consistency(df, market_open=9, market_close=16, tz="America/New_York")
    assert result.is_valid is False


def test_validation_result():
    """Test ValidationResult basic functionality."""
    r = ValidationResult()
    assert bool(r) is True
    r.add_error("test error")
    assert bool(r) is False
    assert "test error" in r.errors
    r.add_warning("test warning")
    assert "test warning" in r.warnings
    r.add_metric("test_key", 42)
    assert r.metrics["test_key"] == 42
