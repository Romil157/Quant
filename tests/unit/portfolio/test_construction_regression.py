"""Regression tests for portfolio construction methods.

Validates that all portfolio construction methods produce distinct, mathematically
expected weights on synthetic datasets with known covariance and return structures,
and do not silently degrade to equal-weight.
"""
import numpy as np
import pandas as pd
import pytest

from quant.portfolio.construction import (
    ConstructionMethod,
    PortfolioConstraints,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    mean_variance,
    minimum_variance,
    risk_parity,
    volatility_targeting,
)


@pytest.fixture
def synthetic_3asset_universe():
    """Create a 3-asset universe with known distinct risk/return characteristics."""
    symbols = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]
    # Variances: 0.01 (10% vol), 0.04 (20% vol), 0.16 (40% vol)
    # Moderate correlation: 0.2
    cov = pd.DataFrame(
        [
            [0.01, 0.004, 0.008],
            [0.004, 0.04, 0.016],
            [0.008, 0.016, 0.16],
        ],
        index=symbols,
        columns=symbols,
    )
    vols = pd.Series([0.10, 0.20, 0.40], index=symbols)
    exp_returns = pd.Series([0.06, 0.12, 0.18], index=symbols)
    signals = pd.Series([1.0, 1.0, 1.0], index=symbols)
    return symbols, cov, vols, exp_returns, signals


def test_default_constraints_is_unconstrained_single_asset():
    """Verify that default PortfolioConstraints allows up to 1.0 max_position."""
    constraints = PortfolioConstraints()
    assert constraints.max_position == 1.0
    assert constraints.long_only is True
    assert constraints.max_gross_exposure == 1.0


def test_infeasible_constraints_raise_value_error():
    """Verify that infeasible long-only constraints (N * max_position < 1.0) raise ValueError."""
    # 3 assets with 20% cap = max possible sum is 60%, impossible to sum to 100%
    constraints = PortfolioConstraints(max_position=0.20, long_only=True)
    signals = pd.Series([1.0, 1.0, 1.0], index=["A", "B", "C"])
    cov = pd.DataFrame(np.eye(3) * 0.04, index=["A", "B", "C"], columns=["A", "B", "C"])

    with pytest.raises(ValueError, match="Infeasible portfolio constraints"):
        equal_weight(signals, constraints)

    with pytest.raises(ValueError, match="Infeasible portfolio constraints"):
        minimum_variance(cov, constraints)

    with pytest.raises(ValueError, match="Infeasible portfolio constraints"):
        risk_parity(cov, constraints)


def test_equal_weight_regression(synthetic_3asset_universe):
    """Verify equal weight produces exact 1/N weights."""
    symbols, _, _, _, signals = synthetic_3asset_universe
    weights = equal_weight(signals)

    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6
    for s in symbols:
        assert abs(weights[s] - 1 / 3) < 1e-6


def test_inverse_volatility_differs_from_equal_weight(synthetic_3asset_universe):
    """Verify inverse volatility weights are inversely proportional to vol and differ from 1/N."""
    symbols, _, vols, _, signals = synthetic_3asset_universe
    weights = inverse_volatility(signals, vols)

    assert abs(weights.sum() - 1.0) < 1e-6
    # Lowest volatility asset must have the highest weight
    assert weights["LOW_VOL"] > weights["MED_VOL"] > weights["HIGH_VOL"]
    # Must differ significantly from equal weight (1/3)
    eq_w = np.ones(3) / 3
    assert np.linalg.norm(np.asarray(weights, dtype=float) - eq_w) > 0.15


def test_risk_parity_differs_from_equal_weight(synthetic_3asset_universe):
    """Verify risk parity solves equal risk contributions and differs from equal-weight."""
    symbols, cov, _, _, _ = synthetic_3asset_universe
    weights = risk_parity(cov)

    assert abs(weights.sum() - 1.0) < 1e-6
    # Lower variance asset must receive higher weight in risk parity
    assert weights["LOW_VOL"] > weights["MED_VOL"] > weights["HIGH_VOL"]
    # Must differ significantly from equal weight (1/3)
    eq_w = np.ones(3) / 3
    w_arr = np.asarray(weights, dtype=float)
    assert np.linalg.norm(w_arr - eq_w) > 0.15

    # Check risk contributions are approximately equal
    cov_arr = np.asarray(cov, dtype=float)
    port_vol = np.sqrt(w_arr @ cov_arr @ w_arr)
    mrc = (cov_arr @ w_arr) / port_vol
    rc = w_arr * mrc
    # Relative risk contributions should be close to 1/3 each
    rc_pct = rc / rc.sum()
    assert np.allclose(rc_pct, 1 / 3, atol=0.02)


def test_minimum_variance_differs_from_equal_weight(synthetic_3asset_universe):
    """Verify minimum variance concentrates in the lowest-variance asset."""
    symbols, cov, _, _, _ = synthetic_3asset_universe
    weights = minimum_variance(cov)

    assert abs(weights.sum() - 1.0) < 1e-6
    # Minimum variance should strongly favor LOW_VOL
    assert weights["LOW_VOL"] > weights["MED_VOL"]
    assert weights["LOW_VOL"] > weights["HIGH_VOL"]
    # Compare against equal weight variance
    cov_arr = np.asarray(cov, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    eq_w = np.ones(3) / 3
    min_var = w_arr @ cov_arr @ w_arr
    eq_var = eq_w @ cov_arr @ eq_w
    assert min_var < eq_var  # Must achieve strictly lower variance than equal weight
    assert np.linalg.norm(w_arr - eq_w) > 0.20


def test_mean_variance_differs_from_equal_weight(synthetic_3asset_universe):
    """Verify mean-variance balances return vs risk and differs from equal weight."""
    symbols, cov, _, exp_returns, _ = synthetic_3asset_universe
    weights = mean_variance(exp_returns, cov, risk_aversion=2.0)

    assert abs(weights.sum() - 1.0) < 1e-6
    eq_w = np.ones(3) / 3
    assert np.linalg.norm(np.asarray(weights, dtype=float) - eq_w) > 0.10


def test_maximum_sharpe_differs_from_equal_weight(synthetic_3asset_universe):
    """Verify maximum Sharpe optimizer finds highest Sharpe portfolio."""
    symbols, cov, _, exp_returns, _ = synthetic_3asset_universe
    weights = maximum_sharpe(exp_returns, cov, risk_free_rate=0.02)

    assert abs(weights.sum() - 1.0) < 1e-6
    eq_w = np.ones(3) / 3
    w_arr = np.asarray(weights, dtype=float)
    assert np.linalg.norm(w_arr - eq_w) > 0.10

    # Sharpe ratio of optimal portfolio must exceed Sharpe of equal-weight portfolio
    cov_arr = np.asarray(cov, dtype=float)
    mu_arr = np.asarray(exp_returns, dtype=float)

    opt_ret = w_arr @ mu_arr
    opt_vol = np.sqrt(w_arr @ cov_arr @ w_arr)
    opt_sr = (opt_ret - 0.02) / opt_vol

    eq_ret = eq_w @ mu_arr
    eq_vol = np.sqrt(eq_w @ cov_arr @ eq_w)
    eq_sr = (eq_ret - 0.02) / eq_vol

    assert opt_sr > eq_sr


def test_all_seven_methods_produce_pairwise_distinct_allocations(synthetic_3asset_universe):
    """Verify that all 7 methods produce distinct portfolio allocations on heterogeneous data."""
    symbols, cov, vols, exp_returns, signals = synthetic_3asset_universe

    allocations = {
        ConstructionMethod.EQUAL_WEIGHT: equal_weight(signals),
        ConstructionMethod.INVERSE_VOLATILITY: inverse_volatility(signals, vols),
        ConstructionMethod.VOLATILITY_TARGETING: volatility_targeting(signals, vols, target_vol=0.15),
        ConstructionMethod.RISK_PARITY: risk_parity(cov),
        ConstructionMethod.MINIMUM_VARIANCE: minimum_variance(cov),
        ConstructionMethod.MEAN_VARIANCE: mean_variance(exp_returns, cov),
        ConstructionMethod.MAXIMUM_SHARPE: maximum_sharpe(exp_returns, cov),
    }

    # Verify each non-equal-weight method differs from EQUAL_WEIGHT
    eq = allocations[ConstructionMethod.EQUAL_WEIGHT]
    eq_arr = np.asarray(eq, dtype=float)
    for method, w in allocations.items():
        if method == ConstructionMethod.EQUAL_WEIGHT:
            continue
        dist = float(np.linalg.norm(np.asarray(w, dtype=float) - eq_arr))
        assert dist > 0.05, f"Method {method.value} unexpectedly degraded to equal weight (dist={dist:.4f})"

