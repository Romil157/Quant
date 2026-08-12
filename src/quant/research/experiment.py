"""Experiment tracking with SQLite backend."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quant.backtest.engine import BacktestConfig


@dataclass
class Experiment:
    """Experiment record."""
    experiment_id: str
    name: str
    strategy: str
    dataset: str
    parameters: dict
    start_date: datetime
    end_date: datetime
    random_seed: int
    git_commit: str | None
    created_at: datetime
    status: str = "running"  # running, completed, failed
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    notes: str = ""


class ExperimentTracker:
    """SQLite-based experiment tracker."""

    def __init__(self, db_path: str = "data/metadata/experiments.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    random_seed INTEGER NOT NULL,
                    git_commit TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    metrics TEXT,  -- JSON
                    config TEXT,   -- JSON
                    notes TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,  -- equity_curve, trades, positions, report
                    artifact_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_strategy
                ON experiments(strategy)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_date
                ON experiments(created_at)
            """)

            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_experiment(
        self,
        name: str,
        strategy: str,
        dataset: str,
        parameters: dict,
        start_date: datetime,
        end_date: datetime,
        random_seed: int = 42,
        git_commit: str | None = None,
        config: dict | None = None,
        notes: str = "",
    ) -> Experiment:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        exp = Experiment(
            experiment_id=experiment_id,
            name=name,
            strategy=strategy,
            dataset=dataset,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed,
            git_commit=git_commit,
            created_at=now,
            config=config or {},
            notes=notes,
        )

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO experiments (
                    experiment_id, name, strategy, dataset, parameters,
                    start_date, end_date, random_seed, git_commit,
                    created_at, status, metrics, config, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.experiment_id,
                exp.name,
                exp.strategy,
                exp.dataset,
                json.dumps(exp.parameters),
                exp.start_date.isoformat(),
                exp.end_date.isoformat(),
                exp.random_seed,
                exp.git_commit,
                exp.created_at.isoformat(),
                exp.status,
                json.dumps(exp.metrics),
                json.dumps(exp.config),
                exp.notes,
            ))
            conn.commit()

        return exp

    def update_experiment(
        self,
        experiment_id: str,
        metrics: dict | None = None,
        status: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update experiment with results."""
        with self._get_conn() as conn:
            updates = []
            params = []

            if metrics is not None:
                updates.append("metrics = ?")
                params.append(json.dumps(metrics))
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)

            if not updates:
                return False

            params.append(experiment_id)
            query = f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?"
            cursor = conn.execute(query, params)
            conn.commit()
            return bool(cursor.rowcount > 0)

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_experiment(row)

    def list_experiments(
        self,
        strategy: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Experiment]:
        """List experiments with optional filters."""
        with self._get_conn() as conn:
            query = "SELECT * FROM experiments WHERE 1=1"
            params: list[str | int] = []

            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.append(limit)
            params.append(offset)

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_experiment(row) for row in rows]

    def get_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        strategy: str | None = None,
        top_n: int = 10,
    ) -> list[Experiment]:
        """Get top experiments by metric."""
        exps = self.list_experiments(strategy=strategy, status="completed", limit=1000)

        def get_metric(exp: Experiment) -> float:
            val = exp.metrics.get(metric)
            return float(val) if val is not None else -float('inf')

        sorted_exps = sorted(exps, key=get_metric, reverse=True)
        return sorted_exps[:top_n]

    def add_artifact(
        self,
        experiment_id: str,
        artifact_type: str,
        artifact_path: str,
    ) -> int:
        """Record an artifact for an experiment."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO experiment_artifacts (experiment_id, artifact_type, artifact_path, created_at)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, artifact_type, artifact_path, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid or 0

    def get_artifacts(self, experiment_id: str) -> list[dict]:
        """Get artifacts for an experiment."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM experiment_artifacts WHERE experiment_id = ?
            """, (experiment_id,)).fetchall()
            return [dict(row) for row in rows]

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        """Convert database row to Experiment object."""
        return Experiment(
            experiment_id=row['experiment_id'],
            name=row['name'],
            strategy=row['strategy'],
            dataset=row['dataset'],
            parameters=json.loads(row['parameters']),
            start_date=datetime.fromisoformat(row['start_date']),
            end_date=datetime.fromisoformat(row['end_date']),
            random_seed=row['random_seed'],
            git_commit=row['git_commit'],
            created_at=datetime.fromisoformat(row['created_at']),
            status=row['status'],
            metrics=json.loads(row['metrics']) if row['metrics'] else {},
            config=json.loads(row['config']) if row['config'] else {},
            notes=row['notes'] or "",
        )

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its artifacts."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM experiment_artifacts WHERE experiment_id = ?", (experiment_id,))
            cursor = conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
            conn.commit()
            return bool(cursor.rowcount > 0)


class ExperimentRunner:
    """High-level experiment runner with automatic tracking."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        backtest_config: BacktestConfig,
    ):
        self.tracker = tracker
        self.backtest_config = backtest_config

    def run_experiment(
        self,
        name: str,
        strategy_name: str,
        dataset: str,
        parameters: dict,
        data: dict,
        strategy_factory,
        random_seed: int = 42,
        git_commit: str | None = None,
        config: dict | None = None,
        notes: str = "",
    ) -> Experiment:
        """Run a complete experiment with tracking."""
        # Get date range from data
        start_date, end_date = self._get_date_range(data)

        # Create experiment
        exp = self.tracker.create_experiment(
            name=name,
            strategy=strategy_name,
            dataset=dataset,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed,
            git_commit=git_commit,
            config=config,
            notes=notes,
        )

        try:
            # Run backtest
            strategy = strategy_factory(parameters)

            # Import here to avoid circular
            from quant.backtest.engine import BacktestEngine

            engine = BacktestEngine(self.backtest_config)
            engine.set_strategy(strategy)
            results = engine.run(data)

            # Calculate metrics
            metrics = self._calculate_metrics(results)

            # Update experiment
            self.tracker.update_experiment(
                exp.experiment_id,
                metrics=metrics,
                status="completed",
            )

            # Save artifacts
            self._save_artifacts(exp.experiment_id, results)

            result = self.tracker.get_experiment(exp.experiment_id)
            if result is None:
                raise RuntimeError(f"Experiment {exp.experiment_id} not found after creation")
            return result

        except Exception as e:
            self.tracker.update_experiment(
                exp.experiment_id,
                status="failed",
                notes=f"Error: {str(e)}",
            )
            raise

    def _get_date_range(self, data: dict) -> tuple[datetime, datetime]:
        """Extract date range from data."""
        all_dates = []
        for df in data.values():
            all_dates.extend(df.index.tolist())

        if not all_dates:
            return datetime.now(), datetime.now()

        all_dates = sorted(all_dates)
        return all_dates[0], all_dates[-1]

    def _calculate_metrics(self, results: dict) -> dict:
        """Calculate standard metrics from backtest results."""
        returns = results.get('returns')
        equity = results.get('equity_curve')

        if returns is None or len(returns) < 2:
            return {}

        import numpy as np

        total_return = results.get('total_return', 0)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Drawdown
        if equity is not None and len(equity) > 0:
            peak = equity.expanding().max()
            dd = (equity - peak) / peak
            max_dd = float(abs(dd.min()))
        else:
            max_dd = 0.0

        # Sortino
        downside = returns[returns < 0]
        sortino = float(returns.mean() * 252 / (downside.std() * np.sqrt(252))) if len(downside) > 0 and downside.std() > 0 else 0

        # Calmar
        calmar = float(returns.mean() * 252 / max_dd) if max_dd > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'num_trades': len(results.get('fills', [])),
            'win_rate': float((returns > 0).mean()) if len(returns) > 0 else 0,
        }

    def _save_artifacts(self, experiment_id: str, results: dict) -> None:
        """Save artifacts (equity curve, trades, etc.) as Parquet files."""
        from pathlib import Path

        artifact_dir = Path("reports/experiments") / experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Equity curve
        equity = results.get('equity_curve')
        if equity is not None:
            path = artifact_dir / "equity_curve.parquet"
            equity.to_frame('equity').to_parquet(path)
            self.tracker.add_artifact(experiment_id, "equity_curve", str(path))

        # Returns
        returns = results.get('returns')
        if returns is not None:
            path = artifact_dir / "returns.parquet"
            returns.to_frame('returns').to_parquet(path)
            self.tracker.add_artifact(experiment_id, "returns", str(path))

        # Trades/fills
        fills = results.get('fills')
        if fills:
            path = artifact_dir / "fills.parquet"
            pd.DataFrame([{
                'order_id': f.order_id,
                'symbol': f.symbol,
                'side': f.side.value,
                'quantity': f.quantity,
                'price': f.price,
                'timestamp': f.timestamp,
                'commission': f.commission,
            } for f in fills]).to_parquet(path)
            self.tracker.add_artifact(experiment_id, "fills", str(path))

        # Account history
        acct = results.get('account_history')
        if acct:
            path = artifact_dir / "account_history.parquet"
            pd.DataFrame([{
                'timestamp': a.timestamp,
                'cash': a.cash,
                'total_value': a.total_value,
                'gross_exposure': a.gross_exposure,
                'net_exposure': a.net_exposure,
            } for a in acct]).to_parquet(path)
            self.tracker.add_artifact(experiment_id, "account_history", str(path))
