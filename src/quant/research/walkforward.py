"""Walk-forward validation and parameter optimization."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestConfig, BacktestEngine, Strategy

logger = logging.getLogger(__name__)



@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    train_window: int = 252        # Training window in bars
    validation_window: int = 63    # Validation window in bars
    test_window: int = 63          # Test window in bars
    step: int = 63                 # Step size between folds
    expanding: bool = False        # Expanding vs rolling window
    anchor: str = "train_end"      # Anchor point: "train_end", "validation_end"
    min_train_size: int = 100      # Minimum training size
    purge_gap: int = 0             # Gap between train/validation (embargo)


@dataclass
class FoldResult:
    """Result of a single walk-forward fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict
    validation_metrics: dict
    test_metrics: dict
    best_params: dict
    best_score: float


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    folds: list[FoldResult]
    aggregate_metrics: dict
    parameter_stability: dict
    config: WalkForwardConfig


class WalkForwardValidator:
    """Walk-forward validation for strategy optimization."""

    def __init__(
        self,
        config: WalkForwardConfig,
        backtest_config: BacktestConfig,
        param_grid: dict[str, list],
        scoring: Callable[[dict], float] = lambda m: m.get('sharpe_ratio', 0),
    ):
        self.config = config
        self.backtest_config = backtest_config
        self.param_grid = param_grid
        self.scoring = scoring

    def validate(
        self,
        data: dict[str, pd.DataFrame],
        strategy_factory: Callable[[dict], Strategy],
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            strategy_factory: Function that creates Strategy from params dict

        Returns:
            WalkForwardResult with all fold results
        """
        # Get common timeline
        common_index = self._get_common_index(data)

        # Generate fold boundaries
        folds = self._generate_folds(common_index)

        results = []
        for fold_id, (train_start, train_end, val_start, val_end, test_start, test_end) in enumerate(folds):

            # Slice data for this fold
            train_data = self._slice_data(data, train_start, train_end)
            val_data = self._slice_data(data, val_start, val_end)
            test_data = self._slice_data(data, test_start, test_end)

            if not train_data or len(train_data) < self.config.min_train_size:
                continue

            # Parameter optimization on train + validation
            best_params, best_score = self._optimize_params(
                train_data, val_data, strategy_factory
            )

            # Evaluate on test
            test_metrics = self._evaluate_params(
                test_data, best_params, strategy_factory
            )

            # Also evaluate on train and validation for reference
            train_metrics = self._evaluate_params(train_data, best_params, strategy_factory)
            val_metrics = self._evaluate_params(val_data, best_params, strategy_factory)

            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                validation_start=val_start,
                validation_end=val_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                best_params=best_params,
                best_score=best_score,
            )
            results.append(fold_result)

        # Aggregate results
        agg_metrics = self._aggregate_metrics(results)
        param_stability = self._analyze_parameter_stability(results)

        return WalkForwardResult(
            folds=results,
            aggregate_metrics=agg_metrics,
            parameter_stability=param_stability,
            config=self.config,
        )

    def _get_common_index(self, data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get common timeline across all symbols."""
        if not data:
            return pd.DatetimeIndex([])

        base_symbol = list(data.keys())[0]
        base_index = data[base_symbol].index

        # Intersect with all symbols
        common = base_index
        for _symbol, df in data.items():
            common = common.intersection(df.index)

        return common.sort_values()

    def _generate_folds(
        self,
        index: pd.DatetimeIndex,
    ) -> list[tuple]:
        """Generate walk-forward fold boundaries."""
        n = len(index)
        folds = []

        start = self.config.train_window

        for i in range(start, n - self.config.validation_window - self.config.test_window + 1, self.config.step):
            train_start_idx = 0 if self.config.expanding else i - self.config.train_window
            train_end_idx = i - self.config.purge_gap - 1

            if train_end_idx < train_start_idx:
                continue

            val_start_idx = train_end_idx + 1 + self.config.purge_gap
            val_end_idx = val_start_idx + self.config.validation_window - 1

            test_start_idx = val_end_idx + 1 + self.config.purge_gap
            test_end_idx = test_start_idx + self.config.test_window - 1

            if test_end_idx >= n:
                break

            folds.append((
                index[train_start_idx], index[train_end_idx],
                index[val_start_idx], index[val_end_idx],
                index[test_start_idx], index[test_end_idx],
            ))

        return folds

    def _slice_data(
        self,
        data: dict[str, pd.DataFrame],
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Slice data to date range."""
        result = {}
        for symbol, df in data.items():
            mask = (df.index >= start) & (df.index <= end)
            sliced = df.loc[mask]
            if len(sliced) > 0:
                result[symbol] = sliced
        return result

    def _optimize_params(
        self,
        train_data: dict[str, pd.DataFrame],
        val_data: dict[str, pd.DataFrame],
        strategy_factory: Callable[[dict], Strategy],
    ) -> tuple[dict, float]:
        """Grid search for best parameters."""
        import itertools

        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        best_params = None
        best_score = -np.inf

        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo, strict=False))

            try:
                strategy = strategy_factory(params)

                # Run on combined train + validation
                combined_data = {**train_data}
                for k, v in val_data.items():
                    if k in combined_data:
                        combined_data[k] = pd.concat([combined_data[k], v])
                    else:
                        combined_data[k] = v

                engine = BacktestEngine(self.backtest_config)
                engine.set_strategy(strategy)
                results = engine.run(combined_data)

                score = self.scoring(results)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.warning("Optimization failed for params %s: %s", params, e)
                continue

        return best_params or {}, best_score

    def _evaluate_params(
        self,
        data: dict[str, pd.DataFrame],
        params: dict,
        strategy_factory: Callable[[dict], Strategy],
    ) -> dict:
        """Evaluate parameters on data."""
        if not data:
            return {}

        try:
            strategy = strategy_factory(params)
            engine = BacktestEngine(self.backtest_config)
            engine.set_strategy(strategy)
            results = engine.run(data)

            return {
                'total_return': results.get('total_return', 0),
                'sharpe_ratio': self._calc_sharpe(results),
                'max_drawdown': self._calc_max_dd(results),
                'num_trades': len(results.get('fills', [])),
            }
        except Exception as e:
            logger.warning("Evaluation failed for params %s: %s", params, e)
            return {}

    def _calc_sharpe(self, results: dict) -> float:
        """Calculate Sharpe from results."""
        returns = results.get('returns')
        if returns is None or len(returns) < 2:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    def _calc_max_dd(self, results: dict) -> float:
        """Calculate max drawdown from equity curve."""
        equity = results.get('equity_curve')
        if equity is None or len(equity) == 0:
            return 0.0
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return float(abs(dd.min()))

    def _aggregate_metrics(self, folds: list[FoldResult]) -> dict:
        """Aggregate metrics across folds."""
        if not folds:
            return {}

        test_returns = [f.test_metrics.get('total_return', 0) for f in folds]
        test_sharpes = [f.test_metrics.get('sharpe_ratio', 0) for f in folds]
        test_dds = [f.test_metrics.get('max_drawdown', 0) for f in folds]

        return {
            'mean_return': np.mean(test_returns),
            'std_return': np.std(test_returns),
            'mean_sharpe': np.mean(test_sharpes),
            'std_sharpe': np.std(test_sharpes),
            'mean_max_dd': np.mean(test_dds),
            'max_max_dd': np.max(test_dds),
            'win_rate': np.mean([r > 0 for r in test_returns]),
            'num_folds': len(folds),
        }

    def _analyze_parameter_stability(self, folds: list[FoldResult]) -> dict:
        """Analyze parameter stability across folds."""
        if not folds:
            return {}

        param_names = list(folds[0].best_params.keys()) if folds[0].best_params else []
        stability = {}

        for param in param_names:
            values = [f.best_params[param] for f in folds if param in f.best_params and f.best_params[param] is not None]
            if values:
                stability[param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf'),
                }

        return stability


class ParameterSweep:
    """Parameter sweep/grid search with cross-validation."""

    def __init__(
        self,
        param_grid: dict[str, list],
        backtest_config: BacktestConfig,
        scoring: Callable[[dict], float] = lambda m: m.get('sharpe_ratio', 0),
    ):
        self.param_grid = param_grid
        self.backtest_config = backtest_config
        self.scoring = scoring

    def run(
        self,
        data: dict[str, pd.DataFrame],
        strategy_factory: Callable[[dict], Strategy],
    ) -> pd.DataFrame:
        """Run parameter sweep."""
        import itertools

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        results = []

        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo, strict=False))

            try:
                strategy = strategy_factory(params)
                engine = BacktestEngine(self.backtest_config)
                engine.set_strategy(strategy)
                bt_results = engine.run(data)

                score = self.scoring(bt_results)

                row = params.copy()
                row['score'] = score
                row['total_return'] = bt_results.get('total_return', 0)
                row['sharpe'] = self._calc_sharpe(bt_results)
                row['max_dd'] = self._calc_max_dd(bt_results)
                row['num_trades'] = len(bt_results.get('fills', []))

                results.append(row)

            except Exception as e:
                logger.warning("Parameter sweep failed for params %s: %s", params, e)
                continue


        return pd.DataFrame(results)

    def _calc_sharpe(self, results: dict) -> float:
        returns = results.get('returns')
        if returns is None or len(returns) < 2:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    def _calc_max_dd(self, results: dict) -> float:
        equity = results.get('equity_curve')
        if equity is None or len(equity) == 0:
            return 0.0
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return float(abs(dd.min()))
