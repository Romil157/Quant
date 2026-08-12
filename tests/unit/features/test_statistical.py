"""Unit tests for statistical features."""
import numpy as np
import pandas as pd

from quant.features.statistical import (
    ewm_std,
    ewma,
    percent_rank,
    rolling_beta,
    rolling_corr,
    rolling_cov,
    rolling_kurt,
    rolling_mean,
    rolling_quantile,
    rolling_skew,
    rolling_std,
    z_score,
)


def test_z_score():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    zs = z_score(series, window=5)
    assert pd.isna(zs.iloc[0])
    assert pd.isna(zs.iloc[3])
    # At index 4 (5th element), mean=3, std=1.58, value=5 -> z=(5-3)/1.58=1.26
    assert abs(zs.iloc[4] - 1.2649) < 0.01


def test_rolling_mean():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = rolling_mean(series, window=3)
    assert pd.isna(mean.iloc[0])
    assert pd.isna(mean.iloc[1])
    assert mean.iloc[2] == 2.0
    assert mean.iloc[3] == 3.0
    assert mean.iloc[4] == 4.0


def test_rolling_std():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    std = rolling_std(series, window=3)
    assert pd.isna(std.iloc[0])
    assert pd.isna(std.iloc[1])
    assert std.iloc[2] > 0


def test_rolling_skew():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    skew = rolling_skew(series, window=5)
    assert pd.isna(skew.iloc[0])
    # Normal distribution should have skew near 0
    valid = skew.dropna()
    assert len(valid) > 0


def test_rolling_kurt():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    kurt = rolling_kurt(series, window=5)
    assert pd.isna(kurt.iloc[0])
    valid = kurt.dropna()
    assert len(valid) > 0


def test_rolling_corr():
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    corr = rolling_corr(s1, s2, window=5)
    assert pd.isna(corr.iloc[0])
    valid = corr.dropna()
    assert all(abs(valid - 1.0) < 1e-6)


def test_rolling_cov():
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    cov = rolling_cov(s1, s2, window=3)
    assert pd.isna(cov.iloc[0])
    assert pd.isna(cov.iloc[1])
    valid = cov.dropna()
    assert all(v > 0 for v in valid)


def test_rolling_beta():
    s_y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    s_x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    beta = rolling_beta(s_y, s_x, window=3)
    assert pd.isna(beta.iloc[0])
    assert pd.isna(beta.iloc[1])
    # y = 2*x, so beta should be 2
    valid = beta.dropna()
    assert all(abs(v - 2.0) < 0.01 for v in valid)


def test_rolling_quantile():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    quant = rolling_quantile(series, window=5, q=0.5)
    assert pd.isna(quant.iloc[0])
    assert pd.isna(quant.iloc[3])
    # Median of [1,2,3,4,5] = 3
    assert quant.iloc[4] == 3.0


def test_percent_rank():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    pr = percent_rank(series, window=5)
    assert pd.isna(pr.iloc[0])
    assert pd.isna(pr.iloc[3])
    # At index 4, value=5, in [1,2,3,4,5], rank of 5 = 4/4 = 1.0
    assert pr.iloc[4] == 1.0


def test_ewma():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ewma_vals = ewma(series, span=3)
    assert pd.isna(ewma_vals.iloc[0])
    assert pd.isna(ewma_vals.iloc[1])
    # Values should be increasing
    valid = ewma_vals.dropna()
    assert all(np.diff(valid) > 0)


def test_ewm_std():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ewm_std_vals = ewm_std(series, span=3)
    assert pd.isna(ewm_std_vals.iloc[0])
    assert pd.isna(ewm_std_vals.iloc[1])
    valid = ewm_std_vals.dropna()
    assert all(v > 0 for v in valid)
