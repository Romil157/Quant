"""Statistical significance testing for quantitative trading strategies.

Implements:
- Probabilistic Sharpe Ratio (PSR) (Bailey & López de Prado, 2014)
- Deflated Sharpe Ratio (DSR) (Bailey & López de Prado, 2014)
- Circular Block Bootstrap confidence intervals for CAGR, Sharpe, and Max Drawdown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI: float = 0.577215664901532860606


@dataclass(frozen=True)
class PSRResult:
    """Result of Probabilistic Sharpe Ratio (PSR) calculation."""

    psr: float
    is_significant: bool
    sharpe_ratio: float
    benchmark_sharpe: float
    z_score: float
    skewness: float
    excess_kurtosis: float
    n_observations: int
    confidence_threshold: float = 0.95


@dataclass(frozen=True)
class DSRResult:
    """Result of Deflated Sharpe Ratio (DSR) calculation."""

    dsr: float
    is_significant: bool
    sharpe_ratio: float
    expected_max_sharpe: float
    n_trials: int
    sr_variance: float
    psr_result: PSRResult
    confidence_threshold: float = 0.95


@dataclass(frozen=True)
class BootstrapCI:
    """Confidence interval from block bootstrap resampling."""

    point_estimate: float
    lower_ci: float
    upper_ci: float
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000


def calculate_psr_from_stats(
    sr_hat: float,
    n_observations: int,
    sr_benchmark: float = 0.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    periods_per_year: int = 252,
    confidence_threshold: float = 0.95,
) -> PSRResult:
    """Calculate Probabilistic Sharpe Ratio (PSR) from summary statistics.

    Parameters
    ----------
    sr_hat : float
        Observed annualized Sharpe ratio.
    n_observations : int
        Number of return observations (e.g. trading days).
    sr_benchmark : float, default 0.0
        Annualized benchmark Sharpe ratio to test against.
    skewness : float, default 0.0
        Sample skewness of returns (scale-invariant).
    excess_kurtosis : float, default 0.0
        Sample Fisher excess kurtosis of returns (0 for standard normal).
    periods_per_year : int, default 252
        Periods per year used for annualizing Sharpe ratios.
    confidence_threshold : float, default 0.95
        Confidence threshold for significance (e.g. 0.95).

    Returns
    -------
    PSRResult
        Dataclass containing PSR probability, significance boolean, and diagnostics.
    """
    if n_observations < 2:
        return PSRResult(
            psr=0.5,
            is_significant=False,
            sharpe_ratio=float(sr_hat),
            benchmark_sharpe=float(sr_benchmark),
            z_score=0.0,
            skewness=float(skewness),
            excess_kurtosis=float(excess_kurtosis),
            n_observations=n_observations,
            confidence_threshold=confidence_threshold,
        )

    # Convert annualized Sharpe to per-period Sharpe for consistent variance formula
    scale = math.sqrt(periods_per_year)
    sr_hat_period = sr_hat / scale
    sr_bench_period = sr_benchmark / scale

    # Pearson kurtosis gamma4 = excess_kurtosis + 3
    # (gamma4 - 1) / 4 = (excess_kurtosis + 2) / 4 = excess_kurtosis / 4 + 0.5
    denom_var = 1.0 - skewness * sr_hat_period + ((excess_kurtosis + 2.0) / 4.0) * (sr_hat_period**2)

    if denom_var <= 0:
        denom_var = 1e-12

    denom_std = math.sqrt(denom_var)

    numerator = (sr_hat_period - sr_bench_period) * math.sqrt(n_observations - 1)
    z_score = numerator / denom_std

    # Standard normal CDF
    psr_val = float(stats.norm.cdf(z_score))

    return PSRResult(
        psr=psr_val,
        is_significant=bool(psr_val >= confidence_threshold),
        sharpe_ratio=float(sr_hat),
        benchmark_sharpe=float(sr_benchmark),
        z_score=float(z_score),
        skewness=float(skewness),
        excess_kurtosis=float(excess_kurtosis),
        n_observations=int(n_observations),
        confidence_threshold=confidence_threshold,
    )


def calculate_psr(
    returns: pd.Series | np.ndarray,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
    confidence_threshold: float = 0.95,
) -> PSRResult:
    """Calculate Probabilistic Sharpe Ratio (PSR) directly from return series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Return series (e.g. daily returns).
    benchmark_sr : float, default 0.0
        Annualized benchmark Sharpe ratio.
    periods_per_year : int, default 252
        Periods per year.
    confidence_threshold : float, default 0.95
        Significance threshold.

    Returns
    -------
    PSRResult
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]

    n = len(clean_rets)
    if n < 2 or np.std(clean_rets, ddof=1) == 0:
        return PSRResult(
            psr=0.5,
            is_significant=False,
            sharpe_ratio=0.0,
            benchmark_sharpe=benchmark_sr,
            z_score=0.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_observations=n,
            confidence_threshold=confidence_threshold,
        )

    mean_r = float(np.mean(clean_rets))
    std_r = float(np.std(clean_rets, ddof=1))
    sr_hat = float(mean_r / std_r * math.sqrt(periods_per_year)) if std_r > 0 else 0.0

    skew_val = float(stats.skew(clean_rets, bias=False)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(clean_rets, fisher=True, bias=False)) if n > 3 else 0.0

    return calculate_psr_from_stats(
        sr_hat=sr_hat,
        n_observations=n,
        sr_benchmark=benchmark_sr,
        skewness=skew_val,
        excess_kurtosis=kurt_val,
        periods_per_year=periods_per_year,
        confidence_threshold=confidence_threshold,
    )


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Compute expected maximum Sharpe ratio under the null hypothesis of zero skill.

    Uses the Bailey & López de Prado (2014) approximation via Euler-Mascheroni constant:
    E[max SR_0] ≈ sqrt(V[SR_hat]) * ((1-γ)*Φ^-1(1 - 1/N) + γ*Φ^-1(1 - 1/(N*e)))

    Parameters
    ----------
    n_trials : int
        Number of independent strategies / trials tested (N).
    sr_variance : float
        Variance of the Sharpe ratio estimates across trials.

    Returns
    -------
    float
        Expected maximum Sharpe ratio.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0

    sr_std = math.sqrt(sr_variance)

    # Φ^-1(1 - 1/N)
    q1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    # Φ^-1(1 - 1/(N*e))
    q2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))

    em_max = sr_std * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2)
    return max(0.0, float(em_max))


def calculate_dsr(
    returns: pd.Series | np.ndarray,
    all_sharpes: list[float] | np.ndarray,
    n_trials: int | None = None,
    periods_per_year: int = 252,
    confidence_threshold: float = 0.95,
) -> DSRResult:
    """Calculate Deflated Sharpe Ratio (DSR) for a single strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Return series of the strategy being evaluated.
    all_sharpes : list[float] or np.ndarray
        Annualized Sharpe ratios of all N strategies tested.
    n_trials : int, optional
        Number of trials N. Defaults to len(all_sharpes).
    periods_per_year : int, default 252
        Periods per year.
    confidence_threshold : float, default 0.95
        Confidence threshold.

    Returns
    -------
    DSRResult
    """
    sharpes_arr = np.asarray(all_sharpes, dtype=float)
    n_t = int(n_trials if n_trials is not None else len(sharpes_arr))
    sr_var = float(np.var(sharpes_arr, ddof=1)) if n_t > 1 and len(sharpes_arr) > 1 else 0.0

    exp_max_sr = expected_max_sharpe(n_trials=n_t, sr_variance=sr_var)

    psr_res = calculate_psr(
        returns=returns,
        benchmark_sr=exp_max_sr,
        periods_per_year=periods_per_year,
        confidence_threshold=confidence_threshold,
    )

    return DSRResult(
        dsr=psr_res.psr,
        is_significant=psr_res.is_significant,
        sharpe_ratio=psr_res.sharpe_ratio,
        expected_max_sharpe=exp_max_sr,
        n_trials=n_t,
        sr_variance=sr_var,
        psr_result=psr_res,
        confidence_threshold=confidence_threshold,
    )


def compute_deflated_sharpe(
    strategy_returns: dict[str, pd.Series | np.ndarray],
    n_trials: int | None = None,
    periods_per_year: int = 252,
    confidence_threshold: float = 0.95,
) -> dict[str, DSRResult]:
    """Compute Deflated Sharpe Ratio (DSR) for all strategies across a benchmark run.

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series | np.ndarray]
        Mapping of strategy name to return series.
    n_trials : int, optional
        Number of trials tested. Defaults to len(strategy_returns).
    periods_per_year : int, default 252
        Periods per year.
    confidence_threshold : float, default 0.95
        Significance threshold.

    Returns
    -------
    dict[str, DSRResult]
        Mapping of strategy name to DSR result.
    """
    n_total = n_trials if n_trials is not None else len(strategy_returns)

    # First pass: calculate observed annualized Sharpe for each strategy
    all_sharpes: list[float] = []
    strat_sharpes: dict[str, float] = {}

    for name, rets in strategy_returns.items():
        arr = np.asarray(rets, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 2 and np.std(arr, ddof=1) > 0:
            sr = float(np.mean(arr) / np.std(arr, ddof=1) * math.sqrt(periods_per_year))
        else:
            sr = 0.0
        strat_sharpes[name] = sr
        all_sharpes.append(sr)

    # Second pass: calculate DSR for each strategy using cross-sectional Sharpe variance
    results: dict[str, DSRResult] = {}
    for name, rets in strategy_returns.items():
        results[name] = calculate_dsr(
            returns=rets,
            all_sharpes=all_sharpes,
            n_trials=n_total,
            periods_per_year=periods_per_year,
            confidence_threshold=confidence_threshold,
        )

    return results


def calculate_block_bootstrap_ci(
    returns: pd.Series | np.ndarray,
    block_length: int = 20,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
    random_seed: int = 42,
) -> dict[str, BootstrapCI]:
    """Calculate Circular Block Bootstrap confidence intervals for CAGR, Sharpe, and Max Drawdown.

    Preserves autocorrelation structures by resampling contiguous blocks in a circular array.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Return series.
    block_length : int, default 20
        Length of contiguous blocks (e.g. 20 trading days ~ 1 month).
    n_bootstraps : int, default 1000
        Number of bootstrap resamples B.
    confidence_level : float, default 0.95
        Target confidence level (e.g. 0.95 for 95% CI).
    periods_per_year : int, default 252
        Trading periods per year.
    random_seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, BootstrapCI]
        Confidence intervals for 'cagr', 'sharpe', and 'max_dd'.
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]
    n = len(clean_rets)

    if n < 2:
        empty_ci = BootstrapCI(point_estimate=0.0, lower_ci=0.0, upper_ci=0.0, confidence_level=confidence_level, bootstrap_samples=0)
        return {"cagr": empty_ci, "sharpe": empty_ci, "max_dd": empty_ci}

    # Point estimates
    # 1. Total return & CAGR
    tot_ret = float(np.prod(1.0 + clean_rets) - 1.0)
    cagr_base = max(1.0 + tot_ret, 1e-4)
    point_cagr = float(cagr_base ** (periods_per_year / n) - 1.0)

    # 2. Sharpe
    std_r = float(np.std(clean_rets, ddof=1))
    point_sharpe = float(np.mean(clean_rets) / std_r * math.sqrt(periods_per_year)) if std_r > 0 else 0.0

    # 3. Max Drawdown
    cum_wealth = np.maximum(np.cumprod(np.maximum(1.0 + clean_rets, 0.0)), 1e-12)
    running_max = np.maximum.accumulate(cum_wealth)
    drawdowns = np.clip((cum_wealth - running_max) / running_max, -1.0, 0.0)
    point_max_dd = float(abs(np.min(drawdowns)))

    # Set up circular array
    block_l = max(1, min(block_length, n))
    extended = np.concatenate([clean_rets, clean_rets[:block_l]])

    rng = np.random.default_rng(random_seed)
    n_blocks = math.ceil(n / block_l)

    boot_cagr = np.empty(n_bootstraps, dtype=float)
    boot_sharpe = np.empty(n_bootstraps, dtype=float)
    boot_max_dd = np.empty(n_bootstraps, dtype=float)

    for b in range(n_bootstraps):
        start_indices = rng.integers(0, n, size=n_blocks)
        # Construct synthetic sample of length n
        sample_blocks = [extended[idx : idx + block_l] for idx in start_indices]
        sample = np.concatenate(sample_blocks)[:n]

        # Metric 1: CAGR
        b_tot = float(np.prod(np.maximum(1.0 + sample, 0.0)) - 1.0)
        b_base = max(1.0 + b_tot, 1e-4)
        boot_cagr[b] = b_base ** (periods_per_year / n) - 1.0

        # Metric 2: Sharpe
        b_std = float(np.std(sample, ddof=1))
        boot_sharpe[b] = (float(np.mean(sample)) / b_std * math.sqrt(periods_per_year)) if b_std > 0 else 0.0

        # Metric 3: Max Drawdown
        b_cum = np.maximum(np.cumprod(np.maximum(1.0 + sample, 0.0)), 1e-12)
        b_peak = np.maximum.accumulate(b_cum)
        b_dd = np.clip((b_cum - b_peak) / b_peak, -1.0, 0.0)
        boot_max_dd[b] = float(abs(np.min(b_dd)))

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    return {
        "cagr": BootstrapCI(
            point_estimate=point_cagr,
            lower_ci=float(np.percentile(boot_cagr, lower_pct)),
            upper_ci=float(np.percentile(boot_cagr, upper_pct)),
            confidence_level=confidence_level,
            bootstrap_samples=n_bootstraps,
        ),
        "sharpe": BootstrapCI(
            point_estimate=point_sharpe,
            lower_ci=float(np.percentile(boot_sharpe, lower_pct)),
            upper_ci=float(np.percentile(boot_sharpe, upper_pct)),
            confidence_level=confidence_level,
            bootstrap_samples=n_bootstraps,
        ),
        "max_dd": BootstrapCI(
            point_estimate=point_max_dd,
            lower_ci=float(np.percentile(boot_max_dd, lower_pct)),
            upper_ci=float(np.percentile(boot_max_dd, upper_pct)),
            confidence_level=confidence_level,
            bootstrap_samples=n_bootstraps,
        ),
    }
