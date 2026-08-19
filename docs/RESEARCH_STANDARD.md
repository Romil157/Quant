# Research Standard

This document defines the minimum bar a research result must clear before being considered "trustworthy" for decision-making in the Quant platform.

## Core Requirements

Every research output (backtest, strategy comparison, ML experiment, paper trading report) must satisfy **all** of the following:

### 1. Out-of-Sample Evaluation
- **Walk-forward validation** is mandatory — no single-period backtests as final evidence
- Minimum configuration: train=252, validation=63, test=63, step=63 (or equivalent for different frequencies)
- Results must report performance on **held-out test windows**, not training/validation windows

### 2. Transaction Costs Included
- All results must use the platform's `ExecutionSimulator` with realistic costs:
  - Commission (default: 2 bps)
  - Bid-ask spread (default: 1 bps)
  - Slippage (default: 2 bps)
  - Market impact (configurable, default: 0)
- Zero-cost results are explicitly labeled as "frictionless" and not used for decisions

### 3. Parameter Sensitivity Checked
- Every strategy with tunable parameters must report **parameter sensitivity**
- Minimum: sweep a reasonable grid around the chosen parameters
- Report: mean/std of key metrics across the grid, not just the "best" point
- Flag parameters where small changes cause large metric swings (high CV)

### 4. Multiple-Testing Correction Applied
- When comparing multiple strategies OR sweeping parameters, apply **Bonferroni correction**
- Report: nominal alpha, number of comparisons, Bonferroni-corrected alpha
- Only results surviving correction are considered statistically significant

### 5. Transparent Reporting
Every research output must include:
- Data source, date range, symbols, provider
- Full strategy configuration (all parameters)
- Execution config (commission, spread, slippage, impact)
- Portfolio constraints (max position, gross/net exposure, long-only)
- Risk limits (max drawdown, action on breach)
- Number of walk-forward folds / test periods
- Aggregate metrics with std across folds
- Parameter stability (mean/std/min/max/CV per parameter)

## Non-Negotiable Gates

A result **cannot** be used for allocation decisions if:
- It lacks walk-forward validation
- It reports only in-sample / training performance
- It omits transaction costs
- It tests multiple strategies/parameters without multiple-testing correction
- It lacks parameter sensitivity analysis
- It cannot be reproduced from the reported configuration

## Enforcement

- CI pipeline runs walk-forward validation on every strategy in `STRATEGY_REGISTRY`
- PRs adding new strategies must include walk-forward results
- Benchmark report (`scripts/run_benchmark.py`) is the canonical comparison artifact
- Research standard violations block merge

## Version

v1.0 — 2026-08-18