"""Portfolio construction algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ConstructionMethod(Enum):
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MEAN_VARIANCE = "mean_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"


@dataclass
class PortfolioConstraints:
    """Portfolio construction constraints."""
    max_position: float = 0.10
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_turnover: float = 1.0
    long_only: bool = True
    target_volatility: float | None = None
    sector_limits: dict[str, float] | None = None
    factor_neutral: list[str] | None = None


def equal_weight(
    signals: pd.Series,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Equal weight portfolio - respects signal direction (sign)."""
    if constraints is None:
        constraints = PortfolioConstraints()

    # Filter out zero signals
    active = signals[signals != 0]
    n = len(active)
    if n == 0:
        return pd.Series(dtype=float)

    # Equal weight magnitude, preserve signal direction
    weights = pd.Series(np.sign(active) * (1.0 / n), index=active.index)
    return _apply_constraints(weights, constraints)


def inverse_volatility(
    signals: pd.Series,
    volatilities: pd.Series,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Inverse volatility weighting - respects signal direction."""
    if constraints is None:
        constraints = PortfolioConstraints()

    active = signals[signals != 0]
    vols = volatilities.reindex(active.index)
    inv_vol = 1.0 / vols.replace(0, np.nan)
    weights = inv_vol / inv_vol.sum()
    weights = weights * np.sign(active)  # Preserve direction
    return _apply_constraints(weights, constraints)


def volatility_targeting(
    signals: pd.Series,
    volatilities: pd.Series,
    target_vol: float = 0.15,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Volatility targeting portfolio."""
    if constraints is None:
        constraints = PortfolioConstraints()

    # Start with equal weight
    n = len(signals)
    weights = pd.Series(1.0 / n, index=signals.index)

    # Scale to target volatility
    port_vol = np.sqrt((weights**2 * volatilities**2).sum())
    if port_vol > 0:
        scale = target_vol / port_vol
        weights = weights * scale

    return _apply_constraints(weights, constraints)


def risk_parity(
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Risk parity portfolio - equal risk contribution."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(cov_matrix)

    def risk_contribution(weights):
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        mrc = (cov_matrix @ weights) / port_vol
        rc = weights * mrc
        return rc

    def objective(weights):
        rc = risk_contribution(weights)
        target_rc = np.sum(rc) / n
        return np.sum((rc - target_rc)**2)

    # Initial guess
    x0 = np.ones(n) / n

    # Constraints
    bounds = [(0, constraints.max_position) if constraints.long_only else (-constraints.max_position, constraints.max_position) for _ in range(n)]

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))},
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

    if result.success:
        weights = pd.Series(result.x, index=cov_matrix.index)
    else:
        weights = pd.Series(1.0 / n, index=cov_matrix.index)

    return _apply_constraints(weights, constraints)


def minimum_variance(
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Minimum variance portfolio."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(cov_matrix)

    def objective(weights):
        return weights @ cov_matrix @ weights

    x0 = np.ones(n) / n
    bounds = [(0, constraints.max_position) if constraints.long_only else (-constraints.max_position, constraints.max_position) for _ in range(n)]
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))},
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

    if result.success:
        weights = pd.Series(result.x, index=cov_matrix.index)
    else:
        weights = pd.Series(1.0 / n, index=cov_matrix.index)

    return _apply_constraints(weights, constraints)


def mean_variance(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Mean-variance optimization."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(expected_returns)

    def objective(weights):
        return risk_aversion * weights @ cov_matrix @ weights - weights @ expected_returns

    x0 = np.ones(n) / n
    bounds = [(0, constraints.max_position) if constraints.long_only else (-constraints.max_position, constraints.max_position) for _ in range(n)]
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))},
    ]

    result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

    if result.success:
        weights = pd.Series(result.x, index=expected_returns.index)
    else:
        weights = pd.Series(1.0 / n, index=expected_returns.index)

    return _apply_constraints(weights, constraints)


def maximum_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Maximum Sharpe ratio portfolio."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(expected_returns)

    def neg_sharpe(weights):
        port_ret = weights @ expected_returns
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        if port_vol == 0:
            return 1e6
        return -(port_ret - risk_free_rate) / port_vol

    x0 = np.ones(n) / n
    bounds = [(0, constraints.max_position) if constraints.long_only else (-constraints.max_position, constraints.max_position) for _ in range(n)]
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))},
    ]

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method='SLSQP')

    if result.success:
        weights = pd.Series(result.x, index=expected_returns.index)
    else:
        weights = pd.Series(1.0 / n, index=expected_returns.index)

    return _apply_constraints(weights, constraints)


def _apply_constraints(weights: pd.Series, constraints: PortfolioConstraints) -> pd.Series:
    """Apply portfolio constraints."""
    w = weights.copy()

    # Max position
    w = w.clip(-constraints.max_position, constraints.max_position)

    # Long only
    if constraints.long_only:
        w = w.clip(lower=0)

    # Gross exposure
    gross = w.abs().sum()
    if gross > constraints.max_gross_exposure:
        w = w * constraints.max_gross_exposure / gross

    # Net exposure
    net = w.sum()
    if net > constraints.max_net_exposure:
        scale = constraints.max_net_exposure / net if net > 0 else 0
        w = w * scale

    # Renormalize to sum to 1 (if not zero)
    total = w.sum()
    if total > 0:
        w = w / total

    return w
