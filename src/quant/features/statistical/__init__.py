"""Statistical features for quant research."""
from __future__ import annotations

import numpy as np
import pandas as pd


def z_score(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (value - rolling_mean) / rolling_std."""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    return (series - rolling_mean) / rolling_std


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean."""
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=window).std()


def rolling_skew(series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness."""
    return series.rolling(window=window, min_periods=window).skew()


def rolling_kurt(series: pd.Series, window: int) -> pd.Series:
    """Rolling kurtosis."""
    return series.rolling(window=window, min_periods=window).kurt()


def rolling_corr(
    series1: pd.Series,
    series2: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling correlation between two series."""
    return series1.rolling(window=window, min_periods=window).corr(series2)


def rolling_cov(
    series1: pd.Series,
    series2: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling covariance between two series."""
    return series1.rolling(window=window, min_periods=window).cov(series2)


def rolling_beta(
    series_y: pd.Series,
    series_x: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling beta (OLS slope) of y on x."""
    cov = rolling_cov(series_y, series_x, window)
    var_x = rolling_std(series_x, window) ** 2
    return cov / var_x


def rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling quantile."""
    return series.rolling(window=window, min_periods=window).quantile(q)


def percent_rank(series: pd.Series, window: int) -> pd.Series:
    """Percent rank of current value within rolling window."""
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True,
    )


def ewma(series: pd.Series, span: int) -> pd.Series:
    """Exponentially Weighted Moving Average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def ewm_std(series: pd.Series, span: int) -> pd.Series:
    """Exponentially Weighted Moving Standard Deviation."""
    return series.ewm(span=span, adjust=False, min_periods=span).std()


def hurst_exponent(series: pd.Series | np.ndarray) -> float:
    """Calculate Hurst exponent for mean reversion detection."""
    # Convert to pandas Series if numpy array
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    if len(series) < 10:
        return np.nan

    # Convert to returns if not already
    returns = series.pct_change().dropna()
    if len(returns) < 10:
        return np.nan

    # Calculate R/S statistic
    lags = range(2, min(20, len(returns) // 2))
    tau = []
    for lag in lags:
        # Difference series
        diff = returns.diff(lag).dropna()
        if len(diff) < 10:
            continue
        rescaled_range = (diff.max() - diff.min()) / diff.std() if diff.std() > 0 else 0
        tau.append(rescaled_range)

    if len(tau) < 2:
        return np.nan

    # Fit log(R/S) = log(c) + H * log(lag)
    log_lags = np.log(list(lags)[:len(tau)])
    log_tau = np.log(tau)

    if len(log_lags) != len(log_tau):
        return np.nan

    H = np.polyfit(log_lags, log_tau, 1)[0]
    return float(np.clip(H, 0, 1))


def half_life(series: pd.Series | np.ndarray) -> float:
    """Calculate half-life of mean reversion using Ornstein-Uhlenbeck."""
    # Convert to pandas Series if numpy array
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    if len(series) < 10:
        return np.nan

    # Use log prices for OU process
    y = np.log(series).diff().dropna()
    if len(y) < 10:
        return np.nan

    # OU process: dy = theta * (mu - y) * dt + sigma * dW
    # Discrete: y_t = y_{t-1} + theta * (mu - y_{t-1}) + epsilon
    # Regression: y_t - y_{t-1} = theta * mu - theta * y_{t-1} + epsilon

    y_lag = y.shift(1).dropna()
    y_diff = y.diff().dropna()

    # Align
    common_idx = y_lag.index.intersection(y_diff.index)
    y_lag = y_lag.loc[common_idx]
    y_diff = y_diff.loc[common_idx]

    if len(y_lag) < 10:
        return np.nan

    # OLS regression
    X = np.column_stack([np.ones(len(y_lag)), -y_lag.values])
    beta = np.linalg.lstsq(X, y_diff.values, rcond=None)[0]

    theta = beta[1]  # Coefficient on -y_{t-1}

    if theta <= 0:
        return np.inf

    return float(np.log(2) / theta)


def cointegration(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 100,
) -> pd.Series:
    """Rolling cointegration test (Engle-Granger)."""
    import warnings
    warnings.filterwarnings('ignore')

    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return pd.Series(np.nan, index=series1.index)

    results = []
    for i in range(window, len(series1)):
        s1 = series1.iloc[i-window:i]
        s2 = series2.iloc[i-window:i]

        try:
            _, pvalue, _ = coint(s1, s2)
            results.append(pvalue)
        except Exception:
            results.append(np.nan)

    return pd.Series(results, index=series1.index[window:])


# Alias for backward compatibility
rolling_correlation = rolling_corr
