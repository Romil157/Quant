"""Unit tests for experiment tracking."""
import os
import tempfile
from datetime import datetime

from quant.research.experiment import (
    ExperimentTracker,
)


def test_experiment_creation():
    """Test experiment creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_experiments.sqlite")
        tracker = ExperimentTracker(db_path)

        exp = tracker.create_experiment(
            name="Test Experiment",
            strategy="MomentumStrategy",
            dataset="SP500",
            parameters={"lookback": 20, "top_n": 10},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            random_seed=42,
        )

        assert exp.experiment_id is not None
        assert exp.name == "Test Experiment"
        assert exp.strategy == "MomentumStrategy"
        assert exp.parameters["lookback"] == 20
        assert exp.status == "running"


def test_experiment_update():
    """Test experiment update."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_experiments.sqlite")
        tracker = ExperimentTracker(db_path)

        exp = tracker.create_experiment(
            name="Test",
            strategy="TestStrategy",
            dataset="TestData",
            parameters={},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        # Update with metrics
        success = tracker.update_experiment(
            exp.experiment_id,
            metrics={"sharpe_ratio": 1.5, "total_return": 0.2},
            status="completed",
        )

        assert success

        # Retrieve and verify
        updated = tracker.get_experiment(exp.experiment_id)
        assert updated.status == "completed"
        assert updated.metrics["sharpe_ratio"] == 1.5
        assert updated.metrics["total_return"] == 0.2


def test_list_experiments():
    """Test listing experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_experiments.sqlite")
        tracker = ExperimentTracker(db_path)

        # Create multiple experiments
        for i in range(3):
            tracker.create_experiment(
                name=f"Exp {i}",
                strategy="Momentum" if i % 2 == 0 else "MeanReversion",
                dataset="SP500",
                parameters={"param": i},
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
            )

        # List all
        all_exps = tracker.list_experiments()
        assert len(all_exps) == 3

        # Filter by strategy
        momentum = tracker.list_experiments(strategy="Momentum")
        assert len(momentum) == 2

        meanrev = tracker.list_experiments(strategy="MeanReversion")
        assert len(meanrev) == 1


def test_experiment_artifacts():
    """Test experiment artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_experiments.sqlite")
        tracker = ExperimentTracker(db_path)

        exp = tracker.create_experiment(
            name="Test",
            strategy="TestStrategy",
            dataset="TestData",
            parameters={},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        # Add artifact
        artifact_id = tracker.add_artifact(
            exp.experiment_id,
            "equity_curve",
            "/path/to/equity.parquet",
        )

        assert artifact_id > 0

        # Get artifacts
        artifacts = tracker.get_artifacts(exp.experiment_id)
        assert len(artifacts) == 1
        assert artifacts[0]['artifact_type'] == "equity_curve"
        assert artifacts[0]['artifact_path'] == "/path/to/equity.parquet"


def test_best_experiments():
    """Test getting best experiments by metric."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_experiments.sqlite")
        tracker = ExperimentTracker(db_path)

        for i in range(5):
            exp = tracker.create_experiment(
                name=f"Exp {i}",
                strategy="Momentum",
                dataset="SP500",
                parameters={"lookback": 10 + i * 10},
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
            )

            # Add varying metrics
            tracker.update_experiment(
                exp.experiment_id,
                metrics={"sharpe_ratio": 0.5 + i * 0.2, "total_return": 0.1 + i * 0.05},
                status="completed",
            )

        # Get top 3 by Sharpe
        best = tracker.get_best_experiments(metric="sharpe_ratio", top_n=3)
        assert len(best) == 3

        # Should be sorted descending
        assert best[0].metrics["sharpe_ratio"] > best[1].metrics["sharpe_ratio"]
        assert best[1].metrics["sharpe_ratio"] > best[2].metrics["sharpe_ratio"]


def test_delete_experiment():
    """Test experiment deletion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_experiments.sqlite")
        tracker = ExperimentTracker(db_path)

        exp = tracker.create_experiment(
            name="Test",
            strategy="TestStrategy",
            dataset="TestData",
            parameters={},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        # Add artifact
        tracker.add_artifact(exp.experiment_id, "test", "/path")

        # Delete
        deleted = tracker.delete_experiment(exp.experiment_id)
        assert deleted

        # Verify deleted
        retrieved = tracker.get_experiment(exp.experiment_id)
        assert retrieved is None

        # Artifacts should also be deleted
        artifacts = tracker.get_artifacts(exp.experiment_id)
        assert len(artifacts) == 0
