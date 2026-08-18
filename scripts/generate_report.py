#!/usr/bin/env python
"""Generate a research report from a backtest's saved artifacts.

Expects the artifact layout produced by ``ExperimentRunner._save_artifacts``:
    <results_dir>/
        equity_curve.parquet
        returns.parquet
        fills.parquet        (optional)
        account_history.parquet  (optional)

Examples
--------
python scripts/generate_report.py --results reports/experiments/abc12345 --format html
python scripts/generate_report.py --results reports/backtest/latest --format markdown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quant.backtest.types import Fill, OrderSide
from quant.research import ReportConfig, ResearchReport


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}", file=sys.stderr)
        return None


def _reconstruct_fills(df: pd.DataFrame | None) -> list[Fill]:
    if df is None or df.empty:
        return []
    fills: list[Fill] = []
    for row in df.itertuples(index=False):
        side_raw = getattr(row, "side", "buy")
        side = OrderSide.BUY if str(side_raw).lower() == "buy" else OrderSide.SELL
        fills.append(
            Fill(
                order_id=str(getattr(row, "order_id", "")),
                symbol=str(getattr(row, "symbol", "")),
                side=side,
                quantity=float(getattr(row, "quantity", 0.0)),
                price=float(getattr(row, "price", 0.0)),
                timestamp=pd.Timestamp(getattr(row, "timestamp", None)).to_pydatetime(),
                commission=float(getattr(row, "commission", 0.0)),
            )
        )
    return fills


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research report")
    parser.add_argument("--results", required=True, help="Directory containing backtest artifacts")
    parser.add_argument("--output", default="reports", help="Output directory for the report")
    parser.add_argument(
        "--format",
        choices=["html", "pdf", "markdown"],
        default="html",
        help="Report format (pdf not implemented; falls back to html)",
    )
    parser.add_argument("--name", default="backtest_report", help="Experiment name in the report")
    parser.add_argument("--strategy", default="unknown", help="Strategy name in the report metadata")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(2)

    equity_df = _read_parquet(results_dir / "equity_curve.parquet")
    returns_df = _read_parquet(results_dir / "returns.parquet")

    if equity_df is None or returns_df is None:
        print(
            "Error: equity_curve.parquet and returns.parquet are required in the results directory",
            file=sys.stderr,
        )
        sys.exit(2)

    # Coerce to pd.Series keyed by date/index.
    equity = equity_df.iloc[:, 0] if len(equity_df.columns) >= 1 else pd.Series(dtype=float)
    returns = returns_df.iloc[:, 0] if len(returns_df.columns) >= 1 else pd.Series(dtype=float)

    fills_df = _read_parquet(results_dir / "fills.parquet")
    account_df = _read_parquet(results_dir / "account_history.parquet")

    fills = _reconstruct_fills(fills_df)
    account_history: list[Any] = []

    if account_df is not None and not account_df.empty:
        from quant.backtest.types import Account

        for row in account_df.itertuples(index=False):
            account_history.append(
                Account(
                    timestamp=pd.Timestamp(getattr(row, "timestamp", None)).to_pydatetime(),
                    cash=float(getattr(row, "cash", 0.0)),
                    positions={},
                    total_value=float(getattr(row, "total_value", 0.0)),
                    gross_exposure=float(getattr(row, "gross_exposure", 0.0)),
                    net_exposure=float(getattr(row, "net_exposure", 0.0)),
                )
            )

    results = {
        "equity_curve": equity,
        "returns": returns,
        "fills": fills,
        "account_history": account_history,
    }

    fmt = "html" if args.format == "pdf" else args.format
    if args.format == "pdf":
        print("Note: pdf format not implemented, falling back to html.", file=sys.stderr)

    report_cfg = ReportConfig(output_dir=Path(args.output), format=fmt)
    report = ResearchReport(report_cfg)

    output_path = report.generate_backtest_report(
        results=results,
        experiment_name=args.name,
        strategy_name=args.strategy,
        parameters={},
        data_info={"results_dir": str(results_dir)},
    )

    print(f"Report written: {output_path}")


if __name__ == "__main__":
    main()
