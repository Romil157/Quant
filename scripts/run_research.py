#!/usr/bin/env python
"""Run a walk-forward research experiment with experiment tracking.

Examples
--------
python scripts/run_research.py --config configs/research.yaml --strategy dual_momentum
python scripts/run_research.py --symbols SPY QQQ IWM --strategy momentum
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from quant.backtest.engine import BacktestConfig
from quant.data import download_data
from quant.research import (
    WalkForwardConfig,
    WalkForwardValidator,
)
from quant.strategies import STRATEGY_REGISTRY, create_strategy


def _strategy_factory(name: str):
    """Return a callable ``params -> Strategy`` for the registry strategy."""
    def factory(params: dict[str, Any]):
        # WalkForward passes a params dict including 'lookback', 'top_n', etc.
        # Filter to keys the strategy accepts by relying on create_strategy.
        try:
            return create_strategy(name, **params)
        except TypeError:
            # Fall back to defaults if the chosen params don't fit.
            return create_strategy(name)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research analysis")
    parser.add_argument("--config", default="configs/research.yaml", help="Config file")
    parser.add_argument("--experiment", default="walk_forward", help="Experiment name")
    parser.add_argument("--symbols", nargs="+", help="Override symbol universe")
    parser.add_argument("--strategy", default="dual_momentum", help="Strategy registry name")
    parser.add_argument("--provider", default=None, help="Data provider override")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    research_cfg = cfg.get("research", {})
    symbols = args.symbols or research_cfg.get("universe", ["SPY", "QQQ", "IWM"])
    start = research_cfg.get("start_date", "2018-01-01")
    end = research_cfg.get("end_date", "2024-12-31")
    provider = args.provider or cfg.get("provider", "mock")

    if args.strategy not in STRATEGY_REGISTRY:
        valid = ", ".join(sorted(STRATEGY_REGISTRY))
        print(f"Error: unknown strategy {args.strategy!r}. Valid: {valid}", file=sys.stderr)
        sys.exit(2)

    wf_cfg = research_cfg.get("walk_forward", {})
    walk_forward_config = WalkForwardConfig(
        train_window=int(wf_cfg.get("train_window", 252)),
        validation_window=int(wf_cfg.get("validation_window", 63)),
        test_window=int(wf_cfg.get("test_window", 63)),
        step=int(wf_cfg.get("step", 63)),
        expanding=bool(wf_cfg.get("expanding", False)),
    )

    # Minimal parameter grid for the chosen strategy. These are intentionally
    # tiny to keep the script runnable on mock data; extend via config for real
    # parameter sweeps.
    param_grid: dict[str, list] = {
        "lookback": [63, 126],
        "top_n": [3, 5],
    }

    print(f"Running research: strategy={args.strategy} symbols={symbols}")
    print(f"  date range: {start} -> {end}  provider={provider}")
    print(f"  walk-forward: train={walk_forward_config.train_window} "
          f"val={walk_forward_config.validation_window} test={walk_forward_config.test_window} "
          f"step={walk_forward_config.step}")

    try:
        data = download_data(symbols=symbols, start_date=start, end_date=end, provider=provider)
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Error: no data returned", file=sys.stderr)
        sys.exit(1)

    backtest_config = BacktestConfig(initial_capital=100_000, start_date=start, end_date=end)
    validator = WalkForwardValidator(
        config=walk_forward_config,
        backtest_config=backtest_config,
        param_grid=param_grid,
    )

    factory = _strategy_factory(args.strategy)

    try:
        result = validator.validate(data, factory)
    except Exception as e:
        print(f"Walk-forward validation failed: {e}", file=sys.stderr)
        raise

    print()
    print("=" * 70)
    print(f"Walk-Forward Results: {args.experiment}")
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

    # Optional experiment tracking.
    tracking_cfg = research_cfg.get("experiment_tracking", {})
    if tracking_cfg.get("enabled"):
        try:
            from quant.research import ExperimentTracker

            tracker = ExperimentTracker(db_path=tracking_cfg.get("db_path", "data/metadata/experiments.sqlite"))
            exp = tracker.create_experiment(
                name=args.experiment,
                strategy=args.strategy,
                dataset=",".join(symbols),
                parameters={"param_grid": param_grid, "walk_forward": wf_cfg},
                start_date=datetime.fromisoformat(start),
                end_date=datetime.fromisoformat(end),
                notes=f"Walk-forward validation: {len(result.folds)} folds",
            )
            tracker.update_experiment(
                exp.experiment_id,
                metrics=result.aggregate_metrics,
                status="completed",
            )
            print(f"  Experiment recorded: id={exp.experiment_id} "
                  f"db={tracker.db_path}")
        except Exception as e:
            print(f"  Warning: experiment tracking skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
