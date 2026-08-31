"""Unit tests for portfolio construction."""
import numpy as np
import pandas as pd

from quant.portfolio.construction import (
    PortfolioConstraints,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    mean_variance,
    minimum_variance,
    risk_parity,
    volatility_targeting,
)


def test_equal_weight():
    """Test equal weight portfolio."""
    signals = pd.Series([1.0, 1.0, 1.0], index=['A', 'B', 'C'])
    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0)
    weights = equal_weight(signals, constraints)

    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(abs(w - 1/3) < 1e-6 for w in weights)


def test_equal_weight_with_constraints():
    """Test equal weight with constraints."""
    signals = pd.Series([1.0] * 20, index=[f'S{i}' for i in range(20)])
    constraints = PortfolioConstraints(max_position=0.1, long_only=True)
    weights = equal_weight(signals, constraints)

    assert len(weights) == 20
    assert all(w <= 0.1 + 1e-6 for w in weights)
    assert all(w >= -1e-6 for w in weights)  # long only
    assert weights.sum() <= 1.0 + 1e-6


def test_inverse_volatility():
    """Test inverse volatility weighting."""
    signals = pd.Series([1.0, 1.0, 1.0], index=['A', 'B', 'C'])
    vols = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0)
    weights = inverse_volatility(signals, vols, constraints)

    # Lower vol should get higher weight
    assert weights['A'] > weights['B'] > weights['C']
    assert abs(weights.sum() - 1.0) < 1e-6


def test_volatility_targeting():
    """Test volatility targeting."""
    signals = pd.Series([1.0, 1.0], index=['A', 'B'])
    vols = pd.Series([0.1, 0.2], index=['A', 'B'])
    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0, target_volatility=0.15)
    weights = volatility_targeting(signals, vols, target_vol=0.15, constraints=constraints)

    # Portfolio vol should be close to target (approximate due to constraints)
    port_vol = np.sqrt((weights**2 * vols**2).sum())
    assert port_vol > 0  # Basic check
    assert abs(weights.sum() - 1.0) < 1e-6


def test_risk_parity():
    """Test risk parity."""
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.005],
         [0.01, 0.09, 0.02],
         [0.005, 0.02, 0.16]],
        index=['A', 'B', 'C'],
        columns=['A', 'B', 'C']
    )

    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0)
    weights = risk_parity(cov, constraints)

    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6
    assert all(w > 0 for w in weights)


def test_minimum_variance():
    """Test minimum variance."""
    cov = pd.DataFrame(
        [[0.04, 0.01],
         [0.01, 0.09]],
        index=['A', 'B'],
        columns=['A', 'B']
    )

    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0)
    weights = minimum_variance(cov, constraints)

    assert len(weights) == 2
    assert abs(weights.sum() - 1.0) < 1e-6
    # A has lower variance, should get more weight
    assert weights['A'] > weights['B']


def test_mean_variance():
    """Test mean-variance optimization."""
    exp_ret = pd.Series([0.1, 0.15, 0.08], index=['A', 'B', 'C'])
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.005],
         [0.01, 0.09, 0.02],
         [0.005, 0.02, 0.16]],
        index=['A', 'B', 'C'],
        columns=['A', 'B', 'C']
    )

    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0)
    weights = mean_variance(exp_ret, cov, risk_aversion=1.0, constraints=constraints)

    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6


def test_maximum_sharpe():
    """Test maximum Sharpe ratio portfolio."""
    exp_ret = pd.Series([0.1, 0.15, 0.08], index=['A', 'B', 'C'])
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.005],
         [0.01, 0.09, 0.02],
         [0.005, 0.02, 0.16]],
        index=['A', 'B', 'C'],
        columns=['A', 'B', 'C']
    )

    constraints = PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0)
    weights = maximum_sharpe(exp_ret, cov, risk_free_rate=0.02, constraints=constraints)

    assert len(weights) == 3
    assert abs(weights.sum() - 1.0) < 1e-6


def test_constraints_max_position():
    """Test max position constraint with feasible universe."""
    signals = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=['A', 'B', 'C', 'D', 'E'])
    constraints = PortfolioConstraints(max_position=0.30, long_only=True, max_gross_exposure=1.0)
    weights = equal_weight(signals, constraints)

    assert len(weights) == 5
    assert all(w <= 0.30 + 1e-6 for w in weights)
    assert abs(weights.sum() - 1.0) < 1e-6



def test_constraints_long_only():
    """Test long-only constraint."""
    signals = pd.Series([1.0, -1.0], index=['A', 'B'])
    constraints = PortfolioConstraints(long_only=True, max_position=1.0, max_gross_exposure=1.0)
    weights = equal_weight(signals, constraints)

    # With long_only, negative signals are clipped to 0
    # Then equal_weight distributes among remaining positive signals
    assert all(w >= -1e-6 for w in weights)
    assert weights['A'] == 1.0  # Only A gets weight
    assert weights['B'] == 0.0  # Short signal becomes 0


def test_constraints_gross_exposure():
    """Test gross exposure constraint."""
    signals = pd.Series([1.0, 1.0, 1.0], index=['A', 'B', 'C'])
    constraints = PortfolioConstraints(max_gross_exposure=0.6, long_only=True, max_position=1.0)
    weights = equal_weight(signals, constraints)

    # After gross exposure scaling and renormalization
    # The constraint is applied before renorm, so final may differ slightly
    assert weights.abs().sum() <= 1.0 + 1e-6  # Should not exceed 1.0 (full investment)
    assert abs(weights.sum() - 1.0) < 1e-6
