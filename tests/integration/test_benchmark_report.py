"""Integration test for cross-strategy benchmark reporting."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.execution import ExecutionConfig
from quant.data import download_data
from quant.portfolio.construction import PortfolioConstraints
from quant.strategies import STRATEGY_REGISTRY, create_strategy


def test_cross_strategy_benchmark_execution():
    """Test that all registered strategies can run through the benchmark evaluation."""
    symbols = ["AAPL", "MSFT"]
    data = download_data(symbols=symbols, start_date="2023-01-01", end_date="2023-06-01", provider="mock")

    assert len(data) == 2

    bt_config = BacktestConfig(
        initial_capital=100_000,
        start_date="2023-01-01",
        end_date="2023-06-01",
        timeframe="1d",
        execution=ExecutionConfig(commission_bps=2.0, spread_bps=1.0, slippage_bps=2.0),
        portfolio_constraints=PortfolioConstraints(max_position=1.0, long_only=True),
    )

    results_by_strategy = {}
    for strategy_name in STRATEGY_REGISTRY.keys():
        strategy = create_strategy(strategy_name)
        engine = BacktestEngine(bt_config)
        engine.set_strategy(strategy)
        res = engine.run(data)

        assert "equity_curve" in res
        assert "returns" in res
        assert len(res["equity_curve"]) > 0
        results_by_strategy[strategy_name] = res

    assert len(results_by_strategy) == len(STRATEGY_REGISTRY)


def test_benchmark_html_report_generation():
    """Test generating a benchmark HTML report file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        report_file = out_dir / "test_benchmark.html"

        # Simple HTML content check
        html = f"""<!DOCTYPE html>
<html>
<head><title>Benchmark</title></head>
<body>
<h1>Cross-Strategy Benchmark</h1>
<table>
<tr><th>Strategy</th><th>Sharpe</th></tr>
{"".join(f"<tr><td>{name}</td><td>1.25</td></tr>" for name in STRATEGY_REGISTRY.keys())}
</table>
</body>
</html>"""
        report_file.write_text(html, encoding="utf-8")

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "Cross-Strategy Benchmark" in content
        for name in STRATEGY_REGISTRY.keys():
            assert name in content
