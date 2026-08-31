"""Portfolio construction algorithms."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


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
    max_position: float = 1.0
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_turnover: float = 1.0
    long_only: bool = True
    target_volatility: float | None = None
    sector_limits: dict[str, float] | None = None
    factor_neutral: list[str] | None = None


def _validate_constraints(n: int, constraints: PortfolioConstraints) -> None:
    """Validate that constraints are mathematically feasible for n assets."""
    if n <= 0:
        return
    if constraints.long_only and (n * constraints.max_position < 1.0 - 1e-6):
        raise ValueError(
            f"Infeasible portfolio constraints: long-only max_position={constraints.max_position} "
            f"across {n} assets cannot sum to 1.0 (maximum possible sum={n * constraints.max_position:.4f})"
        )


def _regularize_covariance(cov: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Ensure covariance matrix is symmetric and strictly positive definite."""
    cov_sym: np.ndarray = np.asarray((cov + cov.T) / 2.0, dtype=float)
    try:
        min_eig = float(np.min(np.linalg.eigvalsh(cov_sym)))
        if min_eig < jitter:
            cov_sym = cov_sym + (jitter - min_eig + 1e-8) * np.eye(cov_sym.shape[0])
    except Exception:
        cov_sym = cov_sym + jitter * np.eye(cov_sym.shape[0])
    return np.asarray(cov_sym, dtype=float)


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

    _validate_constraints(n, constraints)

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
    n = len(active)
    if n == 0:
        return pd.Series(dtype=float)

    _validate_constraints(n, constraints)

    vols = volatilities.reindex(active.index)
    inv_vol = 1.0 / vols.replace(0, np.nan)
    inv_vol = inv_vol.fillna(inv_vol.median() if not np.isnan(inv_vol.median()) else 1.0)
    inv_sum = inv_vol.sum()
    if inv_sum <= 0 or np.isnan(inv_sum):
        weights = pd.Series(1.0 / n, index=active.index)
    else:
        weights = inv_vol / inv_sum
    weights = weights * np.sign(active)  # Preserve direction
    return _apply_constraints(weights, constraints)


def volatility_targeting(
    signals: pd.Series,
    volatilities: pd.Series,
    target_vol: float = 0.15,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Volatility targeting portfolio - weights assets inversely to volatility scaled to target vol."""
    if constraints is None:
        constraints = PortfolioConstraints()

    active = signals[signals != 0]
    n = len(active)
    if n == 0:
        return pd.Series(dtype=float)

    _validate_constraints(n, constraints)

    vols = volatilities.reindex(active.index)
    inv_vol = 1.0 / vols.replace(0, np.nan)
    inv_vol = inv_vol.fillna(inv_vol.median() if not np.isnan(inv_vol.median()) else 1.0)
    inv_sum = inv_vol.sum()
    if inv_sum > 0:
        weights = (inv_vol / inv_sum) * np.sign(active)
    else:
        weights = pd.Series(np.sign(active) * (1.0 / n), index=active.index)

    # Scale by target volatility relative to portfolio volatility
    port_vol = float(np.sqrt(float((weights**2 * vols**2).sum())))
    if port_vol > 0:
        scale = target_vol / port_vol
        weights = weights * min(scale, 1.5)

    return _apply_constraints(weights, constraints)


def risk_parity(
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
    strict: bool = False,
) -> pd.Series:
    """Risk parity portfolio - equal risk contribution."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov_matrix.index)

    _validate_constraints(n, constraints)
    cov = _regularize_covariance(np.asarray(cov_matrix, dtype=float))

    def risk_contribution(weights: np.ndarray):
        port_vol = float(np.sqrt(max(1e-12, float(weights @ cov @ weights))))
        mrc = (cov @ weights) / port_vol
        return weights * mrc

    def objective(weights: np.ndarray) -> float:
        rc = risk_contribution(weights)
        target_rc = np.sum(rc) / n
        return float(np.sum((rc - target_rc) ** 2))

    # Inverse volatility starting point
    variances = np.diag(cov)
    inv_vols = 1.0 / np.sqrt(np.maximum(variances, 1e-8))
    x0 = inv_vols / np.sum(inv_vols)

    upper_bound = min(constraints.max_position, 1.0)
    bounds = [(0.0, upper_bound) if constraints.long_only else (-upper_bound, upper_bound) for _ in range(n)]

    cons: list[dict[str, object]] = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    if not constraints.long_only and constraints.max_gross_exposure < np.inf:
        cons.append({'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))})

    result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 500})

    if result.success and result.x is not None and not np.isnan(result.x).any() and np.all(np.isfinite(result.x)):
        weights = pd.Series(list(result.x), index=cov_matrix.index, dtype=float)
    else:
        msg = f"Risk parity solver warning ({result.message}). Using regularized inverse-volatility allocation."
        if strict:
            logger.error("risk_parity_failed: %s (status=%s)", result.message, result.status)
            raise RuntimeError(msg)
        logger.warning("risk_parity_degraded: %s (status=%s)", result.message, result.status)
        weights = pd.Series(list(x0), index=cov_matrix.index, dtype=float)

    return _apply_constraints(weights, constraints)


def minimum_variance(
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
    strict: bool = False,
) -> pd.Series:
    """Minimum variance portfolio."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov_matrix.index)

    _validate_constraints(n, constraints)
    cov = _regularize_covariance(np.asarray(cov_matrix, dtype=float))

    # Compute analytical unconstrained minimum variance weights as initial guess
    x0 = np.ones(n) / n
    try:
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n)
        analytical_x = (inv_cov @ ones) / (ones @ inv_cov @ ones)
        if not np.isnan(analytical_x).any() and np.all(np.isfinite(analytical_x)):
            if constraints.long_only:
                analytical_x = np.clip(analytical_x, 0.0, constraints.max_position)
                if analytical_x.sum() > 0:
                    analytical_x = analytical_x / analytical_x.sum()
            x0 = analytical_x
    except Exception:
        pass

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def jacobian(weights: np.ndarray):
        return 2.0 * (cov @ weights)

    upper_bound = min(constraints.max_position, 1.0)
    bounds = [(0.0, upper_bound) if constraints.long_only else (-upper_bound, upper_bound) for _ in range(n)]
    cons: list[dict[str, object]] = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    if not constraints.long_only and constraints.max_gross_exposure < np.inf:
        cons.append({'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))})

    result = minimize(objective, x0, jac=jacobian, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 500})

    if result.success and result.x is not None and not np.isnan(result.x).any() and np.all(np.isfinite(result.x)):
        weights = pd.Series(list(result.x), index=cov_matrix.index, dtype=float)
    else:
        msg = f"Minimum variance solver warning ({result.message}). Using analytical estimation."
        if strict:
            logger.error("minimum_variance_failed: %s (status=%s)", result.message, result.status)
            raise RuntimeError(msg)
        logger.warning("minimum_variance_degraded: %s (status=%s)", result.message, result.status)
        weights = pd.Series(list(x0), index=cov_matrix.index, dtype=float)

    return _apply_constraints(weights, constraints)


def mean_variance(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    constraints: PortfolioConstraints | None = None,
    strict: bool = False,
) -> pd.Series:
    """Mean-variance optimization."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(expected_returns)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=expected_returns.index)

    _validate_constraints(n, constraints)
    cov = _regularize_covariance(np.asarray(cov_matrix, dtype=float))
    mu = np.asarray(expected_returns, dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(risk_aversion * (weights @ cov @ weights) - (weights @ mu))

    def jacobian(weights: np.ndarray):
        return 2.0 * risk_aversion * (cov @ weights) - mu

    # Unconstrained Markowitz initial guess
    x0 = np.ones(n) / n
    try:
        inv_cov = np.linalg.pinv(cov)
        analytical_x = inv_cov @ (mu / (2.0 * risk_aversion))
        if constraints.long_only:
            analytical_x = np.clip(analytical_x, 0.0, constraints.max_position)
        if analytical_x.sum() > 0:
            x0 = analytical_x / analytical_x.sum()
    except Exception:
        pass

    upper_bound = min(constraints.max_position, 1.0)
    bounds = [(0.0, upper_bound) if constraints.long_only else (-upper_bound, upper_bound) for _ in range(n)]
    cons: list[dict[str, object]] = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    if not constraints.long_only and constraints.max_gross_exposure < np.inf:
        cons.append({'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))})

    result = minimize(objective, x0, jac=jacobian, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 500})

    if result.success and result.x is not None and not np.isnan(result.x).any() and np.all(np.isfinite(result.x)):
        weights = pd.Series(list(result.x), index=expected_returns.index, dtype=float)
    else:
        msg = f"Mean-variance solver warning ({result.message}). Using analytical estimation."
        if strict:
            logger.error("mean_variance_failed: %s (status=%s)", result.message, result.status)
            raise RuntimeError(msg)
        logger.warning("mean_variance_degraded: %s (status=%s)", result.message, result.status)
        weights = pd.Series(list(x0), index=expected_returns.index, dtype=float)

    return _apply_constraints(weights, constraints)


def maximum_sharpe(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    constraints: PortfolioConstraints | None = None,
    strict: bool = False,
) -> pd.Series:
    """Maximum Sharpe ratio portfolio."""
    if constraints is None:
        constraints = PortfolioConstraints()

    n = len(expected_returns)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=expected_returns.index)

    _validate_constraints(n, constraints)
    cov = _regularize_covariance(np.asarray(cov_matrix, dtype=float))
    mu = np.asarray(expected_returns, dtype=float)

    def neg_sharpe(weights: np.ndarray):
        port_ret = float(weights @ mu)
        port_var = float(weights @ cov @ weights)
        if port_var <= 0:
            return 1e6
        port_vol = float(np.sqrt(port_var))
        return -(port_ret - risk_free_rate) / port_vol

    # Initial guess from excess returns over volatility
    variances = np.diag(cov)
    volatilities = np.sqrt(np.maximum(variances, 1e-8))
    excess_ret = np.maximum(mu - risk_free_rate, 1e-6)
    sr_guess = excess_ret / volatilities
    x0 = sr_guess / np.sum(sr_guess) if np.sum(sr_guess) > 0 else np.ones(n) / n

    upper_bound = min(constraints.max_position, 1.0)
    bounds = [(0.0, upper_bound) if constraints.long_only else (-upper_bound, upper_bound) for _ in range(n)]
    cons: list[dict[str, object]] = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    if not constraints.long_only and constraints.max_gross_exposure < np.inf:
        cons.append({'type': 'ineq', 'fun': lambda w: constraints.max_gross_exposure - np.sum(np.abs(w))})

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 500})

    if result.success and result.x is not None and not np.isnan(result.x).any() and np.all(np.isfinite(result.x)):
        weights = pd.Series(list(result.x), index=expected_returns.index, dtype=float)
    else:
        msg = f"Maximum Sharpe solver warning ({result.message}). Using risk-adjusted estimation."
        if strict:
            logger.error("maximum_sharpe_failed: %s (status=%s)", result.message, result.status)
            raise RuntimeError(msg)
        logger.warning("maximum_sharpe_degraded: %s (status=%s)", result.message, result.status)
        weights = pd.Series(list(x0), index=expected_returns.index, dtype=float)

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

