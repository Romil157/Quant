#!/usr/bin/env python
"""Run a backtest with the specified configuration.

Examples
--------
# Buy and hold on mock data:
python scripts/run_backtest.py --config configs/backtest.yaml --strategy buy_and_hold

# Momentum with a short lookback:
python scripts/run_backtest.py --strategy momentum --lookback 21 --top-n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
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


def _parse_strategy_kwargs(args: argparse.Namespace, strategy_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI strategy overrides on top of YAML strategy params."""
    params: dict[str, Any] = dict(strategy_cfg.get("params", {}))

    # CLI overrides for common strategy kwargs (only applied if provided).
    if args.lookback is not None:
        params["lookback"] = args.lookback
    if args.top_n is not None:
        params["top_n"] = args.top_n
    if args.rebalance_freq is not None:
        params["rebalance_freq"] = args.rebalance_freq

    # Drop keys the chosen strategy does not accept (best effort, schema-free).
    return params


def _filter_params(strategy_cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the strategy's __init__ actually accepts.

    Avoids passing YAML momentum params (e.g. ``lookback``) to e.g.
    :class:`BuyAndHoldStrategy`.
    """
    import inspect

    sig = inspect.signature(strategy_cls.__init__)
    accepted = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    return {k: v for k, v in params.items() if k in accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", default="configs/backtest.yaml", help="Config file path")
    parser.add_argument("--strategy", help="Strategy registry name (e.g. buy_and_hold, momentum)")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, help="Override initial capital")
    parser.add_argument("--symbols", nargs="+", help="Override symbol universe")
    parser.add_argument("--provider", default=None, help="Data provider override (mock/parquet/yfinance)")
    parser.add_argument("--lookback", type=int, help="Strategy lookback override")
    parser.add_argument("--top-n", type=int, help="Strategy top_n override")
    parser.add_argument("--rebalance-freq", type=int, help="Strategy rebalance frequency override")
    parser.add_argument("--report", action="store_true", help="Generate an HTML research report")
    parser.add_argument(
        "--save-artifacts",
        default=None,
        help="Directory to write equity_curve.parquet + returns.parquet + fills.parquet (for scripts/generate_report.py)",
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

    strategy_name = args.strategy or strategy_cfg.get("name", "buy_and_hold")
    if strategy_name not in STRATEGY_REGISTRY:
        valid = ", ".join(sorted(STRATEGY_REGISTRY))
        print(f"Error: unknown strategy {strategy_name!r}. Valid: {valid}", file=sys.stderr)
        sys.exit(2)

    start = args.start or bt_cfg.get("start")
    end = args.end or bt_cfg.get("end")
    if not start or not end:
        print("Error: --start and --end (or backtest.start/end in config) are required", file=sys.stderr)
        sys.exit(2)

    capital = args.capital if args.capital is not None else float(bt_cfg.get("initial_capital", 100_000))

    symbols = args.symbols or cfg.get("symbols") or ["AAPL", "MSFT", "GOOGL"]
    provider = args.provider or cfg.get("provider", "mock")

    strategy_params = _parse_strategy_kwargs(args, strategy_cfg)
    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    filtered_params = _filter_params(strategy_cls, strategy_params)
    try:
        strategy = create_strategy(strategy_name, **filtered_params)
    except TypeError as e:
        print(f"Error: strategy {strategy_name!r} rejected parameters {filtered_params}: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Running backtest: strategy={strategy_name} provider={provider} symbols={symbols}")
    print(f"  date range: {start} -> {end}  capital={capital:,.0f}")

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

    engine = BacktestEngine(bt_config)
    engine.set_strategy(strategy)

    try:
        results = engine.run(data)
    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        raise

    final_equity = float(results.get("final_equity", capital))
    total_return = float(results.get("total_return", 0.0))
    n_orders = len(results.get("orders", []))
    n_fills = len(results.get("fills", []))

    print()
    print("=" * 60)
    print("Backtest Summary")
    print("=" * 60)
    print(f"  Strategy          : {strategy_name}")
    print(f"  Symbols           : {', '.join(symbols)}")
    print(f"  Final equity      : {final_equity:,.2f}")
    print(f"  Total return      : {total_return * 100:+.2f}%")
    print(f"  Max drawdown hit  : {bool(results.get('max_drawdown_hit', False))}")
    print(f"  Orders / fills    : {n_orders} / {n_fills}")
    print("=" * 60)

    if args.report:
        try:
            from quant.research import ReportConfig, ResearchReport

            report_cfg = ReportConfig(
                output_dir=Path(cfg.get("output", {}).get("results_dir", "reports/backtest")),
                format="html",
            )
            report = ResearchReport(report_cfg)
            output_path = report.generate_backtest_report(
                results=results,
                experiment_name=f"{strategy_name}_backtest",
                strategy_name=strategy_name,
                parameters=strategy_params,
                data_info={"symbols": symbols, "start": start, "end": end, "provider": provider},
            )
            print(f"  Report written to : {output_path}")
        except Exception as e:
            print(f"  Warning: report generation skipped: {e}", file=sys.stderr)

    if args.save_artifacts:
        artifact_dir = Path(args.save_artifacts)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            equity_series = results.get("equity_curve")
            if equity_series is not None:
                equity_series.to_frame("equity").to_parquet(artifact_dir / "equity_curve.parquet")
            returns_series = results.get("returns")
            if returns_series is not None:
                returns_series.to_frame("returns").to_parquet(artifact_dir / "returns.parquet")
            fills_list = results.get("fills", [])
            if fills_list:
                pd.DataFrame([{
                    "order_id": f.order_id,
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "timestamp": f.timestamp,
                    "commission": f.commission,
                } for f in fills_list]).to_parquet(artifact_dir / "fills.parquet")
            account = results.get("account_history", [])
            if account:
                pd.DataFrame([{
                    "timestamp": a.timestamp,
                    "cash": a.cash,
                    "total_value": a.total_value,
                    "gross_exposure": a.gross_exposure,
                    "net_exposure": a.net_exposure,
                } for a in account]).to_parquet(artifact_dir / "account_history.parquet")
            print(f"  Artifacts saved to : {artifact_dir}")
        except Exception as e:
            print(f"  Warning: artifact save skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
