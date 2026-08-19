#!/usr/bin/env python
"""Run cross-strategy benchmark across all registered strategies.

Runs all registered strategies through a fast single-backtest pipeline
and generates a comparative HTML report.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.execution import ExecutionConfig
from quant.data import download_data
from quant.portfolio.construction import PortfolioConstraints
from quant.strategies import STRATEGY_REGISTRY, create_strategy


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-strategy benchmark")
    parser.add_argument("--config", default="configs/research.yaml", help="Config file path")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, help="Override initial capital")
    parser.add_argument("--symbols", nargs="+", help="Override symbol universe")
    parser.add_argument("--provider", default=None, help="Data provider override (mock/parquet/yfinance)")
    parser.add_argument("--output", default="reports/benchmark", help="Output directory for benchmark report")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    research_cfg = cfg.get("research", {})
    symbols = args.symbols or research_cfg.get("universe", ["SPY", "QQQ", "IWM"])
    start = args.start or research_cfg.get("start_date", "2018-01-01")
    end = args.end or research_cfg.get("end_date", "2024-12-31")
    provider = args.provider or cfg.get("provider", "mock")
    capital = args.capital if args.capital is not None else 100_000

    if not start or not end:
        print("Error: --start and --end are required", file=sys.stderr)
        sys.exit(2)

    print("Running cross-strategy benchmark")
    print(f"  date range: {start} -> {end}  capital={capital:,.0f}")
    print(f"  provider: {provider}  symbols: {symbols}")

    try:
        data = download_data(symbols=symbols, start_date=start, end_date=end, provider=provider)
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Error: no data returned for the requested symbols", file=sys.stderr)
        sys.exit(1)

    # Test each strategy with a single backtest (fast)
    results_by_strategy = {}

    bt_config = BacktestConfig(
        initial_capital=capital,
        start_date=start,
        end_date=end,
        timeframe="1d",
        execution=ExecutionConfig(commission_bps=2.0, spread_bps=1.0, slippage_bps=2.0, market_impact_bps=0.0),
        portfolio_constraints=PortfolioConstraints(max_position=1.0, max_gross_exposure=1.0, max_net_exposure=1.0, long_only=True),
        max_drawdown=0.20,
        max_drawdown_action="reduce_exposure",
    )

    for strategy_name in sorted(STRATEGY_REGISTRY.keys()):
        print(f"Running {strategy_name}...")
        try:
            strategy = create_strategy(strategy_name)
            engine = BacktestEngine(bt_config)
            engine.set_strategy(strategy)
            results = engine.run(data)

            equity = results.get("equity_curve")
            returns = results.get("returns")

            if equity is not None and returns is not None and len(returns) > 1:
                # Use initial capital as baseline, not first equity value (which may include first trade costs)
                total_return = (equity.iloc[-1] / capital - 1) * 100
                sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
                peak = equity.expanding().max()
                dd = (equity - peak) / peak
                max_dd = abs(dd.min()) * 100
                win_rate = (returns > 0).mean() * 100

                results_by_strategy[strategy_name] = {
                    "total_return": total_return,
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "win_rate": (returns > 0).mean() * 100,
                    "final_equity": float(equity.iloc[-1]),
                    "num_trades": len(results.get("fills", [])),
                }
                print(f"  {strategy_name}: return={total_return:+.2f}% sharpe={sharpe:.2f} max_dd={max_dd:.2f}% win_rate={win_rate:.1f}%")
            else:
                results_by_strategy[strategy_name] = {"error": "No results"}
                print(f"  {strategy_name}: NO RESULTS")

        except Exception as e:
            print(f"  {strategy_name}: ERROR - {e}")
            continue

    # Print summary table
    print("\n" + "=" * 100)
    print("CROSS-STRATEGY BENCHMARK REPORT")
    print("=" * 100)
    print(f"Date range: {start} -> {end}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Provider: {provider}")
    print(f"Initial Capital: ${capital:,.0f}")
    print()

    print(f"{'Strategy':<20} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Win Rate %':<12} {'Final Equity':<15} {'Trades':<8}")
    print("-" * 100)

    for strategy_name, metrics in sorted(results_by_strategy.items()):
        if "error" in metrics:
            print(f"{strategy_name:<20} {'ERROR':<12} {'-':<10} {'-':<12} {'-':<12} {'-':<15} {'-':<8}")
        else:
            print(f"{strategy_name:<20} {metrics['total_return']:+.2f}%{'':<4} {metrics['sharpe']:.2f}{'':<4} {metrics['max_dd']:.2f}%{'':<4} {metrics['win_rate']:.1f}%{'':<4} {metrics['final_equity']:>12,.0f} {'':<3} {metrics['num_trades']:<8}")

    print("=" * 100)

    # Generate HTML report
    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(args.output) / f"benchmark_report_{timestamp}.html"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cross-Strategy Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .summary {{ background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Cross-Strategy Benchmark Report</h1>
    <div class="summary">
        <p><strong>Date Range:</strong> {start} to {end}</p>
        <p><strong>Symbols:</strong> {", ".join(symbols)}</p>
        <p><strong>Data Provider:</strong> {provider}</p>
        <p><strong>Initial Capital:</strong> ${capital:,.0f}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <h2>Strategy Comparison</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Total Return (%)</th>
            <th>Sharpe Ratio</th>
            <th>Max Drawdown (%)</th>
            <th>Win Rate (%)</th>
            <th>Final Equity</th>
            <th>Number of Trades</th>
        </tr>
"""

        for strategy_name, metrics in sorted(results_by_strategy.items()):
            if "error" in metrics:
                html_content += f"""        <tr>
            <td>{strategy_name}</td>
            <td colspan="6">ERROR</td>
        </tr>
"""
            else:
                ret_class = "positive" if metrics['total_return'] >= 0 else "negative"
                dd_class = "positive" if metrics['max_dd'] <= 10 else "negative"
                html_content += f"""        <tr>
            <td>{strategy_name}</td>
            <td class="{ret_class}">{metrics['total_return']:+.2f}</td>
            <td>{metrics['sharpe']:.2f}</td>
            <td class="{dd_class}">{metrics['max_dd']:.2f}</td>
            <td>{metrics['win_rate']:.1f}</td>
            <td>{metrics['final_equity']:,.0f}</td>
            <td>{metrics['num_trades']}</td>
        </tr>
"""

        html_content += f"""    </table>

    <h2>Notes</h2>
    <ul>
        <li>All strategies run on identical data, date range, and transaction costs</li>
        <li>Transaction costs: commission=2bps, spread=1bps, slippage=2bps</li>
        <li>Portfolio constraints: max_position=100%, long_only=True</li>
        <li>Risk management: max_drawdown=20%, reduce_exposure action</li>
    </ul>

    <p><em>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
</body>
</html>"""

        report_path.write_text(html_content)
        print(f"\nBenchmark report written to: {report_path}")

    except Exception as e:
        print(f"Warning: report generation skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
