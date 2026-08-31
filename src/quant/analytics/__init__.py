"""Analytics utilities for performance and risk calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from prices."""
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def calculate_drawdown(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    peak = equity.expanding().max()
    return (equity - peak) / peak


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate maximum drawdown."""
    dd = calculate_drawdown(equity)
    return float(abs(dd.min()))


def calculate_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Calculate Sharpe ratio (annualized)."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns.mean() * 252 - risk_free
    return float(excess / (returns.std() * np.sqrt(252)))


def calculate_sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Calculate Sortino ratio (annualized)."""
    if len(returns) < 2:
        return 0.0
    excess = returns.mean() * 252 - risk_free
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    return float(excess / downside_std) if downside_std > 0 else 0.0


def calculate_calmar(returns: pd.Series, max_dd: float) -> float:
    """Calculate Calmar ratio."""
    if max_dd == 0:
        return 0.0
    ann_return = returns.mean() * 252
    return float(ann_return / max_dd)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    var = calculate_var(returns, confidence)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def calculate_beta(
    returns: pd.Series,
    benchmark: pd.Series,
) -> float:
    """Calculate beta vs benchmark."""
    if len(returns) < 2 or len(benchmark) < 2:
        return 1.0

    aligned = pd.concat([returns, benchmark], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return 1.0

    r_port = aligned.iloc[:, 0]
    r_bench = aligned.iloc[:, 1]

    if r_bench.var() == 0:
        return 1.0

    return float(r_port.cov(r_bench) / r_bench.var())


def calculate_correlation(
    returns: pd.Series,
    benchmark: pd.Series,
) -> float:
    """Calculate correlation vs benchmark."""
    if len(returns) < 2 or len(benchmark) < 2:
        return 0.0

    aligned = pd.concat([returns, benchmark], axis=1, join='inner').dropna()
    if len(aligned) < 2:
        return 0.0

    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


__all__ = [
    "calculate_returns",
    "calculate_log_returns",
    "calculate_drawdown",
    "calculate_max_drawdown",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_calmar",
    "calculate_var",
    "calculate_cvar",
    "calculate_beta",
    "calculate_correlation",
    "calculate_psr",
    "calculate_psr_from_stats",
    "calculate_dsr",
    "compute_deflated_sharpe",
    "expected_max_sharpe",
    "calculate_block_bootstrap_ci",
    "PSRResult",
    "DSRResult",
    "BootstrapCI",
]
