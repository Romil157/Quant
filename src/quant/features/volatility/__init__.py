"""Volatility features for quant research."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.technical import atr


def realized_volatility(
    returns: pd.Series,
    window: int,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Realized volatility from returns."""
    rv = returns.rolling(window=window, min_periods=window).std() * np.sqrt(window)
    if annualize:
        rv = rv * np.sqrt(periods_per_year)
    return rv


def rolling_volatility(
    returns: pd.Series,
    window: int,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling volatility (same as realized_volatility)."""
    return realized_volatility(returns, window, annualize, periods_per_year)


def ewma_volatility(
    returns: pd.Series,
    span: int,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """EWMA volatility."""
    ewm_var = returns.ewm(span=span, adjust=False, min_periods=span).var()
    vol = np.sqrt(ewm_var)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Garman-Klass volatility estimator using OHLC data."""
    # GK estimator: 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
    hl = np.log(high / low)
    co = np.log(close / open_)
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    gk_vol = np.sqrt(gk.rolling(window=window, min_periods=window).mean())
    if annualize:
        gk_vol = gk_vol * np.sqrt(periods_per_year)
    return gk_vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Parkinson volatility estimator using high-low range."""
    hl = np.log(high / low)
    park = np.sqrt((1 / (4 * np.log(2))) * hl**2)
    park_vol = park.rolling(window=window, min_periods=window).mean()
    if annualize:
        park_vol = park_vol * np.sqrt(periods_per_year)
    return park_vol


def rogers_satchell_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rogers-Satchell volatility estimator."""
    ho = np.log(high / open_)
    lo = np.log(low / open_)
    co = np.log(close / open_)
    rs = ho * (ho - co) + lo * (lo - co)
    rs_vol = np.sqrt(rs.rolling(window=window, min_periods=window).mean())
    if annualize:
        rs_vol = rs_vol * np.sqrt(periods_per_year)
    return rs_vol


def atr_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """ATR-based volatility."""
    atr_vals = atr(high, low, close, window)
    if annualize:
        atr_vals = atr_vals * np.sqrt(periods_per_year / window)
    return atr_vals


def volatility_cone(
    returns: pd.Series,
    windows: list[int],
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Volatility cone: rolling volatility at multiple windows."""
    result = {}
    for w in windows:
        result[f"vol_{w}"] = realized_volatility(
            returns, w, annualize, periods_per_year
        )
    return pd.DataFrame(result)
