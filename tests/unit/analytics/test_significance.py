"""Unit tests for statistical significance testing (PSR, DSR, Block Bootstrap)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.analytics.significance import (
    BootstrapCI,
    DSRResult,
    PSRResult,
    calculate_block_bootstrap_ci,
    calculate_dsr,
    calculate_psr,
    calculate_psr_from_stats,
    compute_deflated_sharpe,
    expected_max_sharpe,
)


def test_psr_normal_high_sharpe() -> None:
    """Test PSR on a normal return series with high Sharpe ratio (SR_hat >> SR*)."""
    rng = np.random.default_rng(42)
    # Generate 1000 daily returns with positive mean (annualized Sharpe ~ 2.0)
    mu_daily = 0.12 / 252.0
    sigma_daily = 0.05 / np.sqrt(252.0)
    returns = rng.normal(loc=mu_daily, scale=sigma_daily, size=1000)

    result = calculate_psr(returns, benchmark_sr=0.0)

    assert isinstance(result, PSRResult)
    assert result.sharpe_ratio > 1.8
    assert result.psr > 0.999
    assert result.is_significant is True
    assert result.n_observations == 1000


def test_psr_equal_benchmark() -> None:
    """When SR_hat == SR*, PSR should be exactly 0.5 (z = 0)."""
    result = calculate_psr_from_stats(
        sr_hat=1.5,
        n_observations=500,
        sr_benchmark=1.5,
        skewness=0.0,
        excess_kurtosis=0.0,
    )
    assert pytest.approx(result.psr, abs=1e-5) == 0.5
    assert pytest.approx(result.z_score, abs=1e-5) == 0.0
    assert result.is_significant is False


def test_psr_fat_tails_and_negative_skew_penalty() -> None:
    """Fat tails and negative skew should reduce PSR relative to normal series with equal Sharpe."""
    n_obs = 1000
    sr_hat = 1.2
    sr_bench = 0.0

    # Normal distribution: skew=0, kurt_excess=0
    res_normal = calculate_psr_from_stats(
        sr_hat=sr_hat,
        n_observations=n_obs,
        sr_benchmark=sr_bench,
        skewness=0.0,
        excess_kurtosis=0.0,
    )

    # Fat-tailed & negatively skewed: skew=-1.5, kurt_excess=6.0 (e.g. crash risk)
    res_fat_tail = calculate_psr_from_stats(
        sr_hat=sr_hat,
        n_observations=n_obs,
        sr_benchmark=sr_bench,
        skewness=-1.5,
        excess_kurtosis=6.0,
    )

    # Both have identical point-estimate Sharpe, but fat tails increase estimation variance
    assert res_normal.sharpe_ratio == res_fat_tail.sharpe_ratio
    assert res_normal.z_score > res_fat_tail.z_score
    assert res_normal.psr > res_fat_tail.psr


def test_psr_edge_cases() -> None:
    """Test edge cases: empty series, single observation, zero variance."""
    res_empty = calculate_psr(pd.Series([], dtype=float))
    assert res_empty.psr == 0.5
    assert res_empty.is_significant is False

    res_single = calculate_psr(pd.Series([0.01]))
    assert res_single.psr == 0.5
    assert res_single.is_significant is False

    res_zero_std = calculate_psr(pd.Series([0.01, 0.01, 0.01, 0.01]))
    assert res_zero_std.psr == 0.5
    assert res_zero_std.is_significant is False


def test_expected_max_sharpe_properties() -> None:
    """Test expected maximum Sharpe ratio formula properties."""
    # N=1 should yield 0.0 expected max under null
    assert expected_max_sharpe(n_trials=1, sr_variance=1.0) == 0.0
    assert expected_max_sharpe(n_trials=10, sr_variance=0.0) == 0.0

    # Monotonically increasing with N
    em_10 = expected_max_sharpe(n_trials=10, sr_variance=0.25)
    em_100 = expected_max_sharpe(n_trials=100, sr_variance=0.25)
    assert 0.0 < em_10 < em_100

    # Scales with sqrt of variance
    em_var1 = expected_max_sharpe(n_trials=20, sr_variance=1.0)
    em_var4 = expected_max_sharpe(n_trials=20, sr_variance=4.0)
    assert pytest.approx(em_var4, rel=1e-5) == em_var1 * 2.0


def test_dsr_null_hypothesis_simulation() -> None:
    """Demonstrate DSR eliminates false discoveries under null hypothesis of 20 zero-skill strategies.

    With 20 independent strategies drawn from pure noise (N=0, Var=1), 1-2 strategies
    will show high point-estimate Sharpe ratios by sheer luck (data mining bias).
    Standard uncorrected PSR (testing against SR*=0) would falsely mark them as significant,
    whereas DSR corrects for the 20 trials and identifies that no strategy demonstrates skill.
    """
    rng = np.random.default_rng(123)
    n_strategies = 20
    n_days = 500

    # Generate 20 zero-skill strategies (true mean = 0)
    strategies_returns: dict[str, np.ndarray] = {}
    for i in range(n_strategies):
        # Pure noise daily returns
        rets = rng.normal(loc=0.0, scale=0.01, size=n_days)
        strategies_returns[f"Strategy_{i+1}"] = rets

    dsr_results = compute_deflated_sharpe(strategies_returns, confidence_threshold=0.95)

    # Check uncorrected PSR vs Deflated Sharpe Ratio
    point_sharpes = [res.sharpe_ratio for res in dsr_results.values()]
    max_sharpe = max(point_sharpes)
    best_strat = [k for k, v in dsr_results.items() if v.sharpe_ratio == max_sharpe][0]

    best_res = dsr_results[best_strat]
    # In this sample, by chance the top strategy has an annualized Sharpe around ~1.0
    assert best_res.sharpe_ratio > 0.5
    # The expected max Sharpe under null for N=20 trials will be elevated
    assert best_res.expected_max_sharpe > 0.0
    # Crucially: after deflating for N=20 trials, DSR significance is False!
    assert best_res.is_significant is False

    # Across all 20 null strategies, none (or at most alpha level <= 5%) should pass DSR
    num_significant = sum(1 for res in dsr_results.values() if res.is_significant)
    assert num_significant == 0, f"Expected 0 false discoveries under DSR, got {num_significant}"


def test_block_bootstrap_ci_properties() -> None:
    """Verify block bootstrap confidence interval properties."""
    rng = np.random.default_rng(99)
    n_obs = 1000

    # Base return series
    rets_low_vol = rng.normal(loc=0.0005, scale=0.005, size=n_obs)
    rets_high_vol = rng.normal(loc=0.0005, scale=0.020, size=n_obs)

    ci_low_vol = calculate_block_bootstrap_ci(rets_low_vol, block_length=20, n_bootstraps=500, random_seed=42)
    ci_high_vol = calculate_block_bootstrap_ci(rets_high_vol, block_length=20, n_bootstraps=500, random_seed=42)

    # Check structure
    for key in ["cagr", "sharpe", "max_dd"]:
        assert key in ci_low_vol
        item = ci_low_vol[key]
        assert isinstance(item, BootstrapCI)
        assert item.lower_ci <= item.upper_ci

    # Property 1: Higher return volatility produces wider confidence interval for CAGR and Max Drawdown
    width_cagr_low = ci_low_vol["cagr"].upper_ci - ci_low_vol["cagr"].lower_ci
    width_cagr_high = ci_high_vol["cagr"].upper_ci - ci_high_vol["cagr"].lower_ci
    assert width_cagr_high > width_cagr_low

    width_dd_low = ci_low_vol["max_dd"].upper_ci - ci_low_vol["max_dd"].lower_ci
    width_dd_high = ci_high_vol["max_dd"].upper_ci - ci_high_vol["max_dd"].lower_ci
    assert width_dd_high > width_dd_low

    # Property 2: More observations narrow the confidence interval width
    rets_short = rng.normal(loc=0.0005, scale=0.01, size=200)
    rets_long = rng.normal(loc=0.0005, scale=0.01, size=2000)

    ci_short = calculate_block_bootstrap_ci(rets_short, block_length=20, n_bootstraps=500, random_seed=42)
    ci_long = calculate_block_bootstrap_ci(rets_long, block_length=20, n_bootstraps=500, random_seed=42)

    width_cagr_short = ci_short["cagr"].upper_ci - ci_short["cagr"].lower_ci
    width_cagr_long = ci_long["cagr"].upper_ci - ci_long["cagr"].lower_ci
    assert width_cagr_short > width_cagr_long


def test_block_bootstrap_ci_edge_cases() -> None:
    """Test block bootstrap with empty or single-element inputs."""
    res_empty = calculate_block_bootstrap_ci(pd.Series([], dtype=float))
    assert res_empty["sharpe"].bootstrap_samples == 0

    res_single = calculate_block_bootstrap_ci(pd.Series([0.05]))
    assert res_single["sharpe"].bootstrap_samples == 0


def test_calculate_dsr_single_strategy() -> None:
    """Test calculate_dsr direct function interface."""
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.001, scale=0.01, size=500)
    all_sharpes = [0.2, 0.5, 0.8, 1.2, 1.5]

    res = calculate_dsr(rets, all_sharpes=all_sharpes, n_trials=5, confidence_threshold=0.95)

    assert isinstance(res, DSRResult)
    assert res.n_trials == 5
    assert res.expected_max_sharpe > 0.0
    assert 0.0 <= res.dsr <= 1.0
    assert isinstance(res.is_significant, bool)
