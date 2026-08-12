"""Unit tests for research report generation."""
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from quant.research.report import ReportConfig, ResearchReport


def create_sample_results() -> dict:
    """Create sample backtest results."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    equity = pd.Series(100000 * np.exp(np.cumsum(returns)), index=dates)

    # Mock fills
    class MockFill:
        def __init__(self, symbol, side, quantity, price, timestamp, commission):
            self.symbol = symbol
            self.side = type('Side', (), {'value': side})()
            self.quantity = quantity
            self.price = price
            self.timestamp = timestamp
            self.commission = commission
            self.value = quantity * price

    fills = [
        MockFill("AAPL", "buy", 100, 150.0, datetime(2023, 6, 1), 1.0),
        MockFill("AAPL", "sell", 100, 155.0, datetime(2023, 6, 15), 1.0),
        MockFill("MSFT", "buy", 50, 300.0, datetime(2023, 7, 1), 1.0),
    ]

    # Mock account history
    class MockAccount:
        def __init__(self, timestamp, total_value, gross_exp, net_exp, cash):
            self.timestamp = timestamp
            self.total_value = total_value
            self.gross_exposure = gross_exp
            self.net_exposure = net_exp
            self.cash = cash
            self.positions = {}

    account_history = [
        MockAccount(datetime(2023, 6, 1), 100000, 0.5, 0.3, 100000),
        MockAccount(datetime(2023, 6, 15), 102000, 0.6, 0.4, 98000),
        MockAccount(datetime(2023, 6, 30), 101500, 0.55, 0.35, 99000),
    ]

    return {
        'returns': returns,
        'equity_curve': equity,
        'fills': fills,
        'account_history': account_history,
        'total_return': 0.15,
    }


def test_report_config():
    """Test report configuration."""
    config = ReportConfig(
        output_dir="test_reports",
        include_plots=True,
        include_trades=True,
        format="html",
    )

    assert config.include_plots is True
    assert config.format == "html"


def test_research_report_generation():
    """Test research report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ReportConfig(
            output_dir=tmpdir,
            include_plots=True,
            include_trades=True,
            format="html",
        )

        report = ResearchReport(config)
        results = create_sample_results()

        path = report.generate_backtest_report(
            results=results,
            experiment_name="Test Strategy",
            strategy_name="MomentumStrategy",
            parameters={"lookback": 20, "top_n": 10},
            data_info={"symbols": ["AAPL", "MSFT"], "period": "2023"},
        )

        # Check file created
        import os
        assert os.path.exists(path)
        assert path.suffix == ".html"

        # Check file has content
        with open(path) as f:
            content = f.read()

        assert "Test Strategy" in content
        assert "MomentumStrategy" in content
        assert "Equity Curve" in content


def test_report_json_format():
    """Test JSON report format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ReportConfig(
            output_dir=tmpdir,
            format="json",
        )

        report = ResearchReport(config)
        results = create_sample_results()

        path = report.generate_backtest_report(
            results=results,
            experiment_name="Test",
            strategy_name="TestStrategy",
            parameters={},
            data_info={},
        )

        import json
        import os
        assert os.path.exists(path)
        assert path.suffix == ".json"

        with open(path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "summary" in data
        assert "performance" in data
        assert data["metadata"]["experiment_name"] == "Test"


def test_report_markdown_format():
    """Test Markdown report format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ReportConfig(
            output_dir=tmpdir,
            format="markdown",
        )

        report = ResearchReport(config)
        results = create_sample_results()

        path = report.generate_backtest_report(
            results=results,
            experiment_name="Test",
            strategy_name="TestStrategy",
            parameters={"lookback": 20},
            data_info={"symbols": ["AAPL"]},
        )

        import os
        assert os.path.exists(path)
        assert path.suffix == ".md"

        with open(path) as f:
            content = f.read()

        assert "# Research Report: Test" in content
        assert "Total Return" in content
        assert "Sharpe Ratio" in content


def test_metrics_calculation():
    """Test metrics calculation."""
    config = ReportConfig()
    report = ResearchReport(config)

    # Simple test data
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    equity = pd.Series(100000 * np.exp(np.cumsum(returns)), index=dates)
    fills = []

    metrics = report._calculate_metrics(returns, equity, fills)

    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics


def test_drawdown_duration():
    """Test drawdown duration calculation."""
    config = ReportConfig()
    report = ResearchReport(config)

    # Create drawdown series with known durations
    # First DD: 4 bars (-0.01, -0.02, -0.01, 0) -> actually the 0 ends it
    # Wait: the function counts consecutive negative values
    dd = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.04, -0.02])

    duration = report._calc_dd_duration(dd)

    # All 6 values are negative, so duration = 6
    assert duration == 6
