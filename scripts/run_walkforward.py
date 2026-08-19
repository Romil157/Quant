#!/usr/bin/env python
"""Run walk-forward validation for any registered strategy.

This is the unified entry point for walk-forward validation across all strategies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from quant.backtest.engine import BacktestConfig
from quant.backtest.execution import ExecutionConfig
from quant.data import download_data
from quant.portfolio.construction import PortfolioConstraints
from quant.research import (
    WalkForwardConfig,
    WalkForwardValidator,
)
from quant.strategies import STRATEGY_REGISTRY, create_strategy


def _strategy_factory(name: str):
    """Return a callable ``params -> Strategy`` for the registry strategy."""
    def factory(params: dict[str, Any]):
        try:
            return create_strategy(name, **params)
        except TypeError:
            return create_strategy(name)

    return factory


def _build_execution_config(cfg: dict[str, Any]) -> ExecutionConfig:
    exec_cfg = cfg.get("execution", {})
    return ExecutionConfig(
        commission_bps=float(exec_cfg.get("commission_bps", 2.0)),
        spread_bps=float(exec_cfg.get("spread_bps", 1.0)),
        slippage_bps=float(exec_cfg.get("slippage_bps", 2.0)),
        market_impact_bps=float(exec_cfg.get("market_impact_bps", 0.0)),
    )


def _build_constraints(cfg: dict[str, Any]) -> PortfolioConstraints:
    pf = cfg.get("portfolio", {})
    return PortfolioConstraints(
        max_position=float(pf.get("max_position", 1.0)),
        max_gross_exposure=float(pf.get("max_gross_exposure", 1.0)),
        max_net_exposure=float(pf.get("max_net_exposure", 1.0)),
        long_only=bool(pf.get("long_only", True)),
        target_volatility=pf.get("target_volatility"),
    )


def _parse_strategy_kwargs(args: argparse.Namespace, strategy_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI strategy overrides on top of YAML strategy params."""
    params: dict[str, Any] = dict(strategy_cfg.get("params", {}))

    if args.lookback is not None:
        params["lookback"] = args.lookback
    if args.top_n is not None:
        params["top_n"] = args.top_n
    if args.rebalance_freq is not None:
        params["rebalance_freq"] = args.rebalance_freq

    return params


def _filter_params(strategy_cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the strategy's __init__ actually accepts."""
    import inspect

    sig = inspect.signature(strategy_cls.__init__)
    accepted = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    return {k: v for k, v in params.items() if k in accepted}


def _apply_bonferroni(alpha: float, n_tests: int) -> float:
    """Apply Bonferroni correction for multiple testing."""
    return alpha / n_tests if n_tests > 0 else alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument("--config", default="configs/research.yaml", help="Config file path")
    parser.add_argument("--strategy", help="Strategy registry name (e.g. buy_and_hold, momentum)")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, help="Override initial capital")
    parser.add_argument("--symbols", nargs="+", help="Override symbol universe")
    parser.add_argument("--provider", default=None, help="Data provider override (mock/parquet/yfinance)")
    parser.add_argument("--lookback", type=int, help="Strategy lookback override")
    parser.add_argument("--top-n", type=int, help="Strategy top_n override")
    parser.add_argument("--rebalance-freq", type=int, help="Strategy rebalance frequency override")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--report", action="store_true", help="Generate an HTML research report")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for multiple testing correction (default: 0.05)",
    )
    parser.add_argument(
        "--save-artifacts",
        default=None,
        help="Directory to write equity_curve.parquet + returns.parquet + fills.parquet",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bt_cfg = cfg.get("backtest", {})
    strategy_cfg = cfg.get("strategy", {})
    research_cfg = cfg.get("research", {})

    strategy_name = args.strategy or strategy_cfg.get("name", "buy_and_hold")
    if strategy_name not in STRATEGY_REGISTRY:
        valid = ", ".join(sorted(STRATEGY_REGISTRY))
        print(f"Error: unknown strategy {strategy_name!r}. Valid: {valid}", file=sys.stderr)
        sys.exit(2)

    start = args.start or bt_cfg.get("start") or research_cfg.get("start_date", "2018-01-01")
    end = args.end or bt_cfg.get("end") or research_cfg.get("end_date", "2024-12-31")
    if not start or not end:
        print("Error: --start and --end (or backtest.start/end in config) are required", file=sys.stderr)
        sys.exit(2)

    capital = args.capital if args.capital is not None else float(bt_cfg.get("initial_capital", 100_000))

    symbols = args.symbols or cfg.get("symbols") or research_cfg.get("universe", ["SPY", "QQQ", "IWM"])
    provider = args.provider or cfg.get("provider", "mock")

    strategy_params = _parse_strategy_kwargs(args, strategy_cfg)
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    filtered_params = _filter_params(strategy_cls, strategy_params)

    try:
        _ = create_strategy(strategy_name, **filtered_params)
    except TypeError as e:
        print(f"Error: strategy {strategy_name!r} rejected parameters {filtered_params}: {e}", file=sys.stderr)
        sys.exit(2)

    wf_cfg = research_cfg.get("walk_forward", {})
    walk_forward_config = WalkForwardConfig(
        train_window=int(wf_cfg.get("train_window", 252)),
        validation_window=int(wf_cfg.get("validation_window", 63)),
        test_window=int(wf_cfg.get("test_window", 63)),
        step=int(wf_cfg.get("step", 63)),
        expanding=bool(wf_cfg.get("expanding", False)),
    )

    # Extended parameter grid for walk-forward
    param_grid: dict[str, list] = {
        "lookback": [63, 126, 252],
        "top_n": [3, 5, 10],
        "rebalance_freq": [21, 63],
    }

    print(f"Running walk-forward validation: strategy={strategy_name} provider={provider} symbols={symbols}")
    print(f"  date range: {start} -> {end}  capital={capital:,.0f}")
    print(f"  walk-forward: train={walk_forward_config.train_window} "
          f"val={walk_forward_config.validation_window} test={walk_forward_config.test_window} "
          f"step={walk_forward_config.step}")
    print(f"  parameter grid: {param_grid}")
    print(f"  Bonferroni alpha: {args.alpha} -> {args.alpha / len(param_grid.get('lookback', [1])) / len(param_grid.get('top_n', [1])) / len(param_grid.get('rebalance_freq', [1])):.6f}")

    try:
        data = download_data(symbols=symbols, start_date=start, end_date=end, provider=provider)
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Error: no data returned for the requested symbols", file=sys.stderr)
        sys.exit(1)

    bt_config = BacktestConfig(
        initial_capital=capital,
        start_date=start,
        end_date=end,
        timeframe=bt_cfg.get("timeframe", "1d"),
        execution=_build_execution_config(cfg),
        portfolio_constraints=_build_constraints(cfg),
        max_drawdown=float(cfg.get("risk", {}).get("max_drawdown", 0.20)),
        max_drawdown_action=cfg.get("risk", {}).get("max_drawdown_action", "reduce_exposure"),
    )

    validator = WalkForwardValidator(
        config=walk_forward_config,
        backtest_config=bt_config,
        param_grid=param_grid,
    )

    factory = _strategy_factory(strategy_name)

    try:
        result = validator.validate(data, factory)
    except Exception as e:
        print(f"Walk-forward validation failed: {e}", file=sys.stderr)
        raise

    print()
    print("=" * 70)
    print(f"Walk-Forward Results: {strategy_name}")
    print("=" * 70)
    print(f"  Folds run: {len(result.folds)}")
    for fold in result.folds:
        test = fold.test_metrics or {}
        sharpe = test.get("sharpe_ratio", float("nan"))
        ret = test.get("total_return", float("nan"))
        print(f"  fold {fold.fold_id}: test return={ret * 100:+.2f}%  sharpe={sharpe:.2f}  "
              f"params={fold.best_params}")

    print()
    print(f"  Aggregate metrics: {result.aggregate_metrics}")
    print(f"  Parameter stability: {result.parameter_stability}")
    print("=" * 70)

    # Apply Bonferroni correction for multiple testing disclosure
    n_params = len(param_grid.get("lookback", [1])) * len(param_grid.get("top_n", [1])) * len(param_grid.get("rebalance_freq", [1]))
    bonferroni_alpha = _apply_bonferroni(0.05, n_params)
    print("\nMultiple Testing Correction (Bonferroni):")
    print(f"  Parameter combinations tested: {n_params}")
    print("  Nominal alpha: 0.05")
    print(f"  Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")
    print(f"  Interpretation: Results with p < {bonferroni_alpha:.6f} survive correction")

    if args.report:
        try:
            from quant.research import ReportConfig, ResearchReport

            report_cfg = ReportConfig(
                output_dir=Path(cfg.get("output", {}).get("results_dir", "reports/backtest")),
                format="html",
            )
            report = ResearchReport(report_cfg)
            output_path = report.generate_backtest_report(
                results={"folds": [f.__dict__ for f in result.folds], "aggregate": result.aggregate_metrics},
                experiment_name=f"{strategy_name}_walkforward",
                strategy_name=strategy_name,
                parameters=strategy_params,
                data_info={"symbols": symbols, "start": start, "end": end, "provider": provider},
            )
            print(f"  Report written to : {output_path}")
        except Exception as e:
            print(f"  Warning: report generation skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
