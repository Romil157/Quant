"""Microstructure features for quant research."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bid_ask_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Bid-ask spread: ask - bid."""
    return ask - bid


def relative_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Relative spread: (ask - bid) / mid."""
    mid = (bid + ask) / 2
    return (ask - bid) / mid


def volume_weighted_average_price(
    prices: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """Volume Weighted Average Price (VWAP)."""
    pv = prices * volumes
    return pv.rolling(window=window, min_periods=window).sum() / volumes.rolling(
        window=window, min_periods=window
    ).sum()


def time_weighted_average_price(
    prices: pd.Series,
    times: pd.Series,
    window: int,
) -> pd.Series:
    """Time Weighted Average Price (TWAP)."""
    # For regularly spaced data, TWAP = SMA
    return prices.rolling(window=window, min_periods=window).mean()


def volume_profile(
    prices: pd.Series,
    volumes: pd.Series,
    bins: int = 20,
) -> pd.DataFrame:
    """Volume profile across price levels."""
    # Create price bins
    min_price = prices.min()
    max_price = prices.max()
    bin_edges = np.linspace(min_price, max_price, bins + 1)

    # Assign each price to a bin
    bin_indices = np.digitize(prices, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    # Sum volumes per bin
    vol_per_bin = pd.Series(0.0, index=range(bins))
    for i, b in enumerate(bin_indices):
        vol_per_bin.iloc[b] += volumes.iloc[i]

    # Create bin labels (midpoints)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    return pd.DataFrame({
        "price_level": bin_mids,
        "volume": vol_per_bin.values,
    })


def kyle_lambda(
    returns: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """Kyle's Lambda: price impact coefficient."""
    # Regress returns on signed volume
    signed_vol = volumes * np.sign(returns)
    cov = returns.rolling(window=window, min_periods=window).cov(signed_vol)
    var_vol = signed_vol.rolling(window=window, min_periods=window).var()
    return cov / var_vol


def amihud_illiquidity(
    returns: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """Amihud illiquidity ratio: |return| / volume."""
    illiq = returns.abs() / volumes.replace(0, np.nan)
    return illiq.rolling(window=window, min_periods=window).mean()


def roll_measure(
    prices: pd.Series,
    window: int,
) -> pd.Series:
    """Roll's measure of bid-ask spread from serial covariance."""
    # Roll measure: 2 * sqrt(-Cov(r_t, r_{t-1}))
    returns = prices.pct_change()
    cov = returns.rolling(window=window, min_periods=window).cov(returns.shift(1))
    return 2 * np.sqrt(-cov.clip(upper=0))


def order_flow_imbalance(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    window: int,
) -> pd.Series:
    """Order flow imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)."""
    total = buy_volume + sell_volume
    return (buy_volume - sell_volume) / total.replace(0, np.nan)


def volume_change(volumes: pd.Series, window: int = 1) -> pd.Series:
    """Volume change over window."""
    return volumes / volumes.shift(window) - 1


def volume_z_score(volumes: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of volume."""
    rolling_mean = volumes.rolling(window=window, min_periods=window).mean()
    rolling_std = volumes.rolling(window=window, min_periods=window).std()
    return (volumes - rolling_mean) / rolling_std


def volume_moving_average(volumes: pd.Series, window: int) -> pd.Series:
    """Volume moving average."""
    return volumes.rolling(window=window, min_periods=window).mean()
