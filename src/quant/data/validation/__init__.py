"""Data validation module for market data quality checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.providers.base import ProviderError


class ValidationError(ProviderError):
    """Raised when data fails validation checks."""
    pass


class LookaheadError(ValidationError):
    """Raised when signal uses future information."""
    pass


class MissingBarsError(ValidationError):
    """Raised when bars are missing from the data."""
    pass


class OutlierError(ValidationError):
    """Raised when price/volume outliers are detected."""
    pass


class ValidationResult:
    """Result of a data validation run."""
    def __init__(self) -> None:
        self.is_valid: bool = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.metrics: dict[str, int | float] = {}

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_metric(self, key: str, value: int | float) -> None:
        self.metrics[key] = value

    def __bool__(self) -> bool:
        return self.is_valid


def validate_ohlc_relationships(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLC relationships: High >= max(Open, Close, Low) and Low <= min(Open, Close, High)."""
    result = ValidationResult()
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        result.add_error(f"Missing required OHLC columns. Got: {set(df.columns)}")
        return result

    # Check High >= max(Open, Close, Low)
    high_ok = (df["high"] >= df[["open", "close", "low"]].max(axis=1))
    low_ok = (df["low"] <= df[["open", "close", "high"]].min(axis=1))

    high_fail_count = int((~high_ok).sum())
    low_fail_count = int((~low_ok).sum())

    if high_fail_count > 0:
        result.add_error(
            f"{high_fail_count} rows have High < max(Open, Close, Low)"
        )
    if low_fail_count > 0:
        result.add_error(
            f"{low_fail_count} rows have Low > min(Open, Close, High)"
        )

    # If any failures, mark invalid
    if high_fail_count > 0 or low_fail_count > 0:
        result.is_valid = False

    result.add_metric("high_violations", high_fail_count)
    result.add_metric("low_violations", low_fail_count)
    return result


def validate_no_negative_prices(df: pd.DataFrame) -> ValidationResult:
    """Validate no negative prices."""
    result = ValidationResult()
    price_cols = ["open", "high", "low", "close"]
    present = [c for c in price_cols if c in df.columns]
    if not present:
        result.add_error("No price columns found")
        return result

    negatives = df[present].lt(0).any(axis=1)
    neg_count = int(negatives.sum())

    if neg_count > 0:
        result.is_valid = False
        result.add_error(f"{neg_count} rows with negative price(s)")
        # Show which rows and columns
        for col in present:
            col_neg = df[col].lt(0).sum()
            if col_neg > 0:
                result.add_warning(f"{col_neg} negative {col} values")

    result.add_metric("negative_price_rows", neg_count)
    return result


def validate_no_negative_volume(df: pd.DataFrame) -> ValidationResult:
    """Validate no negative volume."""
    result = ValidationResult()
    if "volume" not in df.columns:
        result.add_error("Volume column not found")
        return result

    negatives = df["volume"].lt(0).sum()
    if negatives > 0:
        result.is_valid = False
        result.add_error(f"{negatives} rows with negative volume")
        result.add_metric("negative_volume_rows", negatives)
    else:
        result.add_metric("negative_volume_rows", 0)
    return result


def validate_no_zero_prices(df: pd.DataFrame) -> ValidationResult:
    """Validate no zero prices (open, high, low, close)."""
    result = ValidationResult()
    price_cols = ["open", "high", "low", "close"]
    present = [c for c in price_cols if c in df.columns]
    if not present:
        result.add_error("No price columns found")
        return result

    zeros = df[present].eq(0).any(axis=1).sum()
    if zeros > 0:
        result.is_valid = False
        result.add_error(f"{zeros} rows with zero price(s)")
        result.add_metric("zero_price_rows", zeros)
    else:
        result.add_metric("zero_price_rows", 0)
    return result


def _get_ts_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract DatetimeIndex from DataFrame, whether in column or index."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "timestamp" in df.columns:
        return df.set_index("timestamp").index
    raise ValueError("No DatetimeIndex found and no 'timestamp' column present")


def validate_no_duplicate_timestamps(df: pd.DataFrame) -> ValidationResult:
    """Validate no duplicate timestamps."""
    result = ValidationResult()
    try:
        ts_index = _get_ts_index(df)
    except ValueError as e:
        result.add_error(str(e))
        result.is_valid = False
        return result

    dupes = ts_index.duplicated(keep=False).sum()
    if dupes > 0:
        result.is_valid = False
        result.add_error(f"{int(dupes)} duplicate timestamp(s) found")
        result.add_metric("duplicate_timestamps", int(dupes))
    else:
        result.add_metric("duplicate_timestamps", 0)
    return result


def validate_no_missing_timestamps(df: pd.DataFrame, freq: str = "B") -> ValidationResult:
    """Validate no missing timestamps in the expected business day frequency."""
    result = ValidationResult()
    try:
        ts_index = _get_ts_index(df)
    except ValueError as e:
        result.add_error(str(e))
        result.is_valid = False
        return result

    if len(ts_index) < 2:
        result.add_metric("missing_timestamps", 0)
        return result

    # Generate expected business days
    expected = pd.date_range(start=ts_index[0], end=ts_index[-1], freq=freq)
    actual = set(ts_index)
    missing = expected.difference(actual)
    missing_count = len(missing)

    if missing_count > 0:
        result.is_valid = False
        result.add_error(f"{missing_count} missing timestamp(s) in range")
        result.add_metric("missing_timestamps", missing_count)
        # Show first few missing
        if len(missing) <= 5:
            result.add_warning(f"Missing dates: {sorted(missing)[:5]}")
    result.add_metric("missing_timestamps", missing_count)
    return result


def validate_timezone_consistency(df: pd.DataFrame) -> ValidationResult:
    """Validate timezone consistency across the DataFrame."""
    result = ValidationResult()
    try:
        ts_index = _get_ts_index(df)
    except ValueError as e:
        result.add_error(str(e))
        result.add_metric("timezone_issues", 0)
        return result

    if len(ts_index) == 0:
        result.add_metric("timezone_issues", 0)
        return result

    tzs = ts_index.tz
    if tzs is None:
        result.add_warning("Index is timezone-naive")
        result.add_metric("timezone_issues", 0)
        return result

    unique_tzs = ts_index.tz.unique()
    if len(unique_tzs) > 1:
        result.is_valid = False
        result.add_error(f"Inconsistent timezones detected: {unique_tzs}")
        result.add_metric("timezone_issues", len(unique_tzs))
    else:
        result.add_metric("timezone_issues", 0)
    return result


def validate_extreme_outliers(
    df: pd.DataFrame,
    price_col: str = "close",
    z_threshold: float = 5.0,
) -> ValidationResult:
    """Validate extreme price outliers using Z-score."""
    result = ValidationResult()
    if price_col not in df.columns:
        result.add_error(f"Price column '{price_col}' not found")
        return result

    prices = df[price_col].dropna()
    if len(prices) < 3:
        result.add_metric("outlier_count", 0)
        return result

    mean = prices.mean()
    std = prices.std()
    if std == 0:
        result.add_metric("outlier_count", 0)
        return result

    z_scores = np.abs((prices - mean) / std)
    outliers = z_scores > z_threshold
    outlier_count = int(outliers.sum())

    if outlier_count > 0:
        result.is_valid = False
        result.add_error(f"{outlier_count} extreme outlier(s) in {price_col} (z > {z_threshold})")
        result.add_metric("outlier_count", outlier_count)
    else:
        result.add_metric("outlier_count", 0)
    return result


def validate_data_gaps(
    df: pd.DataFrame,
    expected_freq: str = "1d",
) -> ValidationResult:
    """Validate data gaps - check for missing bars between first and last timestamp."""
    result = ValidationResult()
    try:
        ts_index = _get_ts_index(df)
    except ValueError as e:
        result.add_error(str(e))
        result.is_valid = False
        return result

    if len(ts_index) < 2:
        result.add_metric("gap_count", 0)
        return result

    # Generate full expected range using the expected frequency
    expected = pd.date_range(start=ts_index[0], end=ts_index[-1], freq=expected_freq)
    actual_set = set(ts_index)
    expected_set = set(expected)

    gaps = expected_set - actual_set
    gap_count = len(gaps)

    if gap_count > 0:
        result.is_valid = False
        result.add_error(f"{gap_count} data gap(s) detected")
    result.add_metric("gap_count", gap_count)
    return result


def validate_trading_session_consistency(
    df: pd.DataFrame,
    market_open: int = 9,
    market_close: int = 16,
    tz: str = "America/New_York",
) -> ValidationResult:
    """Validate that timestamps fall within trading session hours."""
    result = ValidationResult()
    try:
        ts_index = _get_ts_index(df)
    except ValueError as e:
        result.add_error(str(e))
        result.is_valid = False
        return result

    if len(ts_index) == 0:
        result.add_metric("session_violations", 0)
        return result

    try:
        from zoneinfo import ZoneInfo
        tz_info = ZoneInfo(tz)
    except Exception:
        tz_info = None

    if tz_info is not None:
        # Convert index to target timezone
        try:
            # Handle both tz-aware and tz-naive indices
            if ts_index.tz is not None:
                ts_index_tz = ts_index.tz_convert(tz_info)
            else:
                ts_index_tz = ts_index.tz_localize(None).tz_localize(tz_info)
        except Exception:
            ts_index_tz = ts_index
    else:
        ts_index_tz = ts_index

    # Check hour of day in target timezone
    hours = ts_index_tz.hour
    violations_mask = (hours < market_open) | (hours >= market_close)
    violations = int(violations_mask.sum())

    if violations > 0:
        result.is_valid = False
        result.add_error(f"{violations} bar(s) outside trading hours ({market_open}:00-{market_close}:00 {tz})")
        result.add_metric("session_violations", violations)
    else:
        result.add_metric("session_violations", 0)

    # Show a few examples if violations exist
    if violations > 0 and violations <= 5:
        bad_hours = hours[violations_mask].tolist()
        result.add_warning(f"Bars with out-of-session hours: {bad_hours}")

    return result
