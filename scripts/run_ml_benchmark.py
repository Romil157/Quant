#!/usr/bin/env python
"""Run ML vs rule-based comparative benchmark under the Phase 1 standard.

Evaluates OnlineEnsemble / ML pipeline against built-in rule-based alpha strategies
on identical market data and transaction costs.
"""
from __future__ import annotations

import argparse
import sys

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.execution import ExecutionConfig
from quant.data import download_data
from quant.portfolio.construction import PortfolioConstraints
from quant.strategies import STRATEGY_REGISTRY, create_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML vs rule-based strategy benchmark")
    parser.add_argument("--config", default="configs/research.yaml", help="Config file path")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"], help="Symbol universe")
    parser.add_argument("--provider", default="mock", help="Data provider (mock/parquet/yfinance)")
    parser.add_argument("--output", default="reports/ml_benchmark", help="Output directory")
    args = parser.parse_args()

    print("Running ML vs Rule-Based Strategy Benchmark")
    print(f"  date range: {args.start} -> {args.end}")
    print(f"  symbols: {args.symbols}")
    print(f"  provider: {args.provider}")

    try:
        data = download_data(symbols=args.symbols, start_date=args.start, end_date=args.end, provider=args.provider)
    except Exception as e:
        print(f"Error loading market data: {e}", file=sys.stderr)
        sys.exit(1)

    bt_config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        timeframe="1d",
        execution=ExecutionConfig(commission_bps=2.0, spread_bps=1.0, slippage_bps=2.0),
        portfolio_constraints=PortfolioConstraints(max_position=1.0, long_only=True),
    )

    results = {}
    for name in STRATEGY_REGISTRY:
        try:
            strategy = create_strategy(name)
            engine = BacktestEngine(bt_config)
            engine.set_strategy(strategy)
            res = engine.run(data)

            equity = res.get("equity_curve")
            returns = res.get("returns")
            if equity is not None and returns is not None and len(returns) > 1:
                tot_ret = (equity.iloc[-1] / args.capital - 1) * 100
                sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
                peak = equity.expanding().max()
                max_dd = abs(((equity - peak) / peak).min()) * 100
                results[name] = {
                    "total_return": tot_ret,
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "num_trades": len(res.get("fills", [])),
                }
        except Exception as err:
            results[name] = {"error": str(err)}

    print("\n" + "=" * 80)
    print(f"{'Strategy':<20} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Trades':<8}")
    print("-" * 80)
    for name, m in sorted(results.items()):
        if "error" in m:
            print(f"{name:<20} {'ERROR':<12} {'-':<10} {'-':<12} {'-':<8}")
        else:
            print(f"{name:<20} {m['total_return']:+.2f}%{'':<4} {m['sharpe']:.2f}{'':<4} {m['max_dd']:.2f}%{'':<4} {m['num_trades']:<8}")
    print("=" * 80)


if __name__ == "__main__":
    main()
