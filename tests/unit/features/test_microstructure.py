"""Unit tests for microstructure features."""
import pandas as pd

from quant.features.microstructure import (
    amihud_illiquidity,
    bid_ask_spread,
    kyle_lambda,
    order_flow_imbalance,
    relative_spread,
    roll_measure,
    time_weighted_average_price,
    volume_change,
    volume_moving_average,
    volume_profile,
    volume_weighted_average_price,
    volume_z_score,
)


def test_bid_ask_spread():
    bid = pd.Series([100.0, 100.1, 100.2, 100.3])
    ask = pd.Series([100.1, 100.2, 100.3, 100.4])
    spread = bid_ask_spread(bid, ask)
    assert all(abs(spread - 0.1) < 1e-10)


def test_relative_spread():
    bid = pd.Series([100.0, 100.1, 100.2])
    ask = pd.Series([100.1, 100.2, 100.3])
    rel = relative_spread(bid, ask)
    assert all(rel > 0)
    assert all(rel < 0.01)


def test_volume_weighted_average_price():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    volumes = pd.Series([1000, 2000, 1500, 1000])
    vwap = volume_weighted_average_price(prices, volumes, window=3)
    # At index 2: (100*1000 + 101*2000 + 102*1500) / (1000+2000+1500) = 455000/4500 = 101.111
    assert pd.isna(vwap.iloc[0])
    assert pd.isna(vwap.iloc[1])
    assert abs(vwap.iloc[2] - 101.111) < 0.01


def test_time_weighted_average_price():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    times = pd.Series([1, 2, 3, 4])
    twap = time_weighted_average_price(prices, times, window=3)
    assert pd.isna(twap.iloc[0])
    assert pd.isna(twap.iloc[1])
    assert twap.iloc[2] == 101.0


def test_volume_profile():
    prices = pd.Series([100, 101, 102, 100, 101, 102, 103])
    volumes = pd.Series([1000, 2000, 1500, 1000, 2000, 1500, 1000])
    profile = volume_profile(prices, volumes, bins=5)
    assert isinstance(profile, pd.DataFrame)
    assert "price_level" in profile.columns
    assert "volume" in profile.columns
    assert len(profile) == 5


def test_kyle_lambda():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.015, -0.015, 0.025, -0.025])
    volumes = pd.Series([1000, 1000, 2000, 2000, 1500, 1500, 2500, 2500])
    kl = kyle_lambda(returns, volumes, window=4)
    assert len(kl) == len(returns)


def test_amihud_illiquidity():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    volumes = pd.Series([1000, 2000, 1500, 1000])
    illiq = amihud_illiquidity(returns, volumes, window=2)
    assert len(illiq) == len(returns)
    assert pd.isna(illiq.iloc[0])


def test_roll_measure():
    prices = pd.Series([100.0, 100.5, 100.0, 100.5, 100.0, 100.5])
    roll = roll_measure(prices, window=3)
    assert len(roll) == len(prices)


def test_order_flow_imbalance():
    buy_vol = pd.Series([1000, 2000, 1500, 1000])
    sell_vol = pd.Series([800, 1500, 2000, 1200])
    ofi = order_flow_imbalance(buy_vol, sell_vol, window=2)
    assert len(ofi) == len(buy_vol)
    assert abs(ofi.iloc[0] - 0.111) < 0.01


def test_volume_change():
    volumes = pd.Series([1000, 1100, 1050, 1200])
    change = volume_change(volumes, window=1)
    assert pd.isna(change.iloc[0])
    assert abs(change.iloc[1] - 0.1) < 0.01
    assert abs(change.iloc[2] - (-0.04545)) < 0.01


def test_volume_z_score():
    volumes = pd.Series([1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400])
    zs = volume_z_score(volumes, window=4)
    assert pd.isna(zs.iloc[0])
    assert pd.isna(zs.iloc[2])
    valid = zs.dropna()
    assert len(valid) > 0


def test_volume_moving_average():
    volumes = pd.Series([1000, 1100, 1050, 1200, 1150])
    vma = volume_moving_average(volumes, window=3)
    assert pd.isna(vma.iloc[0])
    assert pd.isna(vma.iloc[1])
    assert vma.iloc[2] == (1000 + 1100 + 1050) / 3
