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

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from quant.analytics.significance import (
    calculate_block_bootstrap_ci,
    calculate_psr,
    compute_deflated_sharpe,
)
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
    parser.add_argument("--bootstraps", type=int, default=500, help="Number of block bootstrap iterations")
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
    results_by_strategy: dict[str, dict[str, Any]] = {}
    strategy_returns: dict[str, Any] = {}

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
                strategy_returns[strategy_name] = returns
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
                    "win_rate": win_rate,
                    "final_equity": float(equity.iloc[-1]),
                    "num_trades": len(results.get("fills", [])),
                    "returns": returns,
                }
                print(f"  {strategy_name}: return={total_return:+.2f}% sharpe={sharpe:.2f} max_dd={max_dd:.2f}% win_rate={win_rate:.1f}%")
            else:
                results_by_strategy[strategy_name] = {"error": "No results"}
                print(f"  {strategy_name}: NO RESULTS")

        except Exception as e:
            print(f"  {strategy_name}: ERROR - {e}")
            continue

    # Compute Statistical Significance (PSR, DSR, Block Bootstrap CIs)
    print("\nComputing statistical significance layer (PSR, DSR, Block Bootstrap CIs)...")
    dsr_results = compute_deflated_sharpe(strategy_returns, confidence_threshold=0.95) if strategy_returns else {}
    psr_results = {name: calculate_psr(rets, benchmark_sr=0.0) for name, rets in strategy_returns.items()}
    ci_results = {
        name: calculate_block_bootstrap_ci(rets, block_length=20, n_bootstraps=args.bootstraps, random_seed=42)
        for name, rets in strategy_returns.items()
    }

    # Print summary table
    print("\n" + "=" * 130)
    print("CROSS-STRATEGY BENCHMARK & STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 130)
    print(f"Date range: {start} -> {end} | Provider: {provider} | Initial Capital: ${capital:,.0f}")
    print(f"Multiple Testing Correction: Deflated Sharpe Ratio (DSR) over N={len(STRATEGY_REGISTRY)} strategies")
    print(f"Bootstrap Resampling: Circular Block Bootstrap (L=20 days, B={args.bootstraps} resamples, 95% CI)")
    print("-" * 130)

    header = f"{'Strategy':<18} {'Return %':<10} {'Sharpe [95% CI]':<24} {'Max DD % [95% CI]':<24} {'PSR (SR*>0)':<13} {'DSR (N=7)':<12} {'Significance (95%)':<20}"
    print(header)
    print("-" * 130)

    for strategy_name in sorted(STRATEGY_REGISTRY.keys()):
        metrics = results_by_strategy.get(strategy_name, {})
        if "error" in metrics or strategy_name not in strategy_returns:
            print(f"{strategy_name:<18} {'ERROR':<10} {'-':<24} {'-':<24} {'-':<13} {'-':<12} {'ERROR':<20}")
        else:
            psr_val = psr_results[strategy_name].psr
            dsr_val = dsr_results[strategy_name].dsr
            is_sig = dsr_results[strategy_name].is_significant
            sig_text = "PASS (p>=0.95)" if is_sig else "Not Sig (DSR<0.95)"

            sharpe_pt = metrics["sharpe"]
            sharpe_ci = ci_results[strategy_name]["sharpe"]
            sharpe_str = f"{sharpe_pt:.2f} [{sharpe_ci.lower_ci:.2f}, {sharpe_ci.upper_ci:.2f}]"

            dd_pt = metrics["max_dd"]
            dd_ci = ci_results[strategy_name]["max_dd"]
            dd_str = f"{dd_pt:.1f}% [{dd_ci.lower_ci * 100:.1f}%, {dd_ci.upper_ci * 100:.1f}%]"

            print(
                f"{strategy_name:<18} {metrics['total_return']:+.2f}%{'':<3} "
                f"{sharpe_str:<24} {dd_str:<24} "
                f"{psr_val * 100:>5.1f}%{'':<6} {dsr_val * 100:>5.1f}%{'':<5} {sig_text:<20}"
            )

    print("=" * 130)

    # Generate HTML report
    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"benchmark_report_{timestamp}.html"

        # Calculate sample expected max Sharpe for display
        exp_max_sr = next(iter(dsr_results.values())).expected_max_sharpe if dsr_results else 0.0

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cross-Strategy Benchmark & Statistical Significance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
        .container {{ max-width: 1300px; margin: 0 auto; background: white; padding: 32px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
        h1 {{ color: #0f172a; margin-top: 0; font-size: 26px; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        h2 {{ color: #334155; margin-top: 32px; font-size: 20px; }}
        .summary {{ background: #f1f5f9; padding: 20px; border-radius: 8px; margin: 20px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
        .summary-card {{ background: white; padding: 14px; border-radius: 6px; border: 1px solid #e2e8f0; }}
        .summary-card strong {{ display: block; font-size: 12px; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }}
        .summary-card span {{ font-size: 16px; font-weight: 600; color: #0f172a; }}
        table {{ border-collapse: collapse; width: 100%; margin: 24px 0; font-size: 14px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 12px 14px; text-align: center; }}
        th {{ background-color: #0f172a; color: white; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        tr:hover {{ background-color: #f1f5f9; }}
        .badge-pass {{ background: #dcfce7; color: #15803d; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; display: inline-block; }}
        .badge-fail {{ background: #fee2e2; color: #b91c1c; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; display: inline-block; }}
        .ci-sub {{ font-size: 11px; color: #64748b; display: block; margin-top: 2px; }}
        .positive {{ color: #16a34a; font-weight: 600; }}
        .negative {{ color: #dc2626; font-weight: 600; }}
        .methodology {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 0 8px 8px 0; margin-top: 24px; }}
        .methodology h3 {{ margin-top: 0; color: #1e40af; font-size: 16px; }}
        .methodology ul {{ margin: 0; padding-left: 20px; color: #1e3a8a; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Cross-Strategy Benchmark & Statistical Significance Report</h1>
        <div class="summary">
            <div class="summary-card">
                <strong>Date Range</strong>
                <span>{start} &rarr; {end}</span>
            </div>
            <div class="summary-card">
                <strong>Symbols Tested</strong>
                <span>{", ".join(symbols)}</span>
            </div>
            <div class="summary-card">
                <strong>Data Provider</strong>
                <span>{provider}</span>
            </div>
            <div class="summary-card">
                <strong>Initial Capital</strong>
                <span>${capital:,.0f}</span>
            </div>
            <div class="summary-card">
                <strong>Multiple Testing Correction</strong>
                <span>DSR (N={len(STRATEGY_REGISTRY)} trials, E[max SR₀]={exp_max_sr:.2f})</span>
            </div>
            <div class="summary-card">
                <strong>Bootstrap Model</strong>
                <span>Circular Block (L=20d, B={args.bootstraps})</span>
            </div>
        </div>

        <h2>Strategy Comparison & Statistical Confidence</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>CAGR [95% CI]</th>
                <th>Sharpe Ratio [95% CI]</th>
                <th>Max DD [95% CI]</th>
                <th>Win Rate</th>
                <th>PSR (SR* &gt; 0)</th>
                <th>DSR (N={len(STRATEGY_REGISTRY)})</th>
                <th>Significance (95%)</th>
            </tr>
"""

        for strategy_name in sorted(STRATEGY_REGISTRY.keys()):
            metrics = results_by_strategy.get(strategy_name, {})
            if "error" in metrics or strategy_name not in strategy_returns:
                html_content += f"""            <tr>
                <td><strong>{strategy_name}</strong></td>
                <td colspan="8" style="color: #94a3b8;">ERROR / NO RESULTS</td>
            </tr>\n"""
            else:
                ret_class = "positive" if metrics['total_return'] >= 0 else "negative"
                psr_obj = psr_results[strategy_name]
                dsr_obj = dsr_results[strategy_name]
                ci_obj = ci_results[strategy_name]

                badge = '<span class="badge-pass">&check; PASS (p&ge;0.95)</span>' if dsr_obj.is_significant else '<span class="badge-fail">&cross; Not Significant (DSR&lt;0.95)</span>'

                cagr_pt = ci_obj["cagr"].point_estimate * 100
                cagr_low = ci_obj["cagr"].lower_ci * 100
                cagr_high = ci_obj["cagr"].upper_ci * 100

                sr_pt = metrics["sharpe"]
                sr_low = ci_obj["sharpe"].lower_ci
                sr_high = ci_obj["sharpe"].upper_ci

                dd_pt = metrics["max_dd"]
                dd_low = ci_obj["max_dd"].lower_ci * 100
                dd_high = ci_obj["max_dd"].upper_ci * 100

                html_content += f"""            <tr>
                <td style="text-align: left; font-weight: 600;">{strategy_name}</td>
                <td class="{ret_class}">{metrics['total_return']:+.2f}%</td>
                <td>
                    {cagr_pt:+.2f}%
                    <span class="ci-sub">[{cagr_low:+.1f}%, {cagr_high:+.1f}%]</span>
                </td>
                <td>
                    <strong>{sr_pt:.2f}</strong>
                    <span class="ci-sub">[{sr_low:.2f}, {sr_high:.2f}]</span>
                </td>
                <td>
                    {dd_pt:.2f}%
                    <span class="ci-sub">[{dd_low:.1f}%, {dd_high:.1f}%]</span>
                </td>
                <td>{metrics['win_rate']:.1f}%</td>
                <td><strong>{psr_obj.psr * 100:.1f}%</strong></td>
                <td><strong>{dsr_obj.dsr * 100:.1f}%</strong></td>
                <td>{badge}</td>
            </tr>\n"""

        html_content += f"""        </table>

        <div class="methodology">
            <h3>Statistical Rigor & Multiple Testing Methodology</h3>
            <ul>
                <li><strong>Probabilistic Sharpe Ratio (PSR)</strong>: Measures the probability that the strategy's true Sharpe ratio exceeds zero (SR* = 0), accounting for non-normal return distributions (skewness and excess kurtosis) and sample size n.</li>
                <li><strong>Deflated Sharpe Ratio (DSR)</strong>: Extends PSR by replacing the zero benchmark with E[max SR₀] (the expected maximum Sharpe ratio under the null hypothesis of zero true skill across N={len(STRATEGY_REGISTRY)} independent strategies). Corrects for selection bias, data mining, and multiple testing.</li>
                <li><strong>Circular Block Bootstrap (CBB)</strong>: Resamples daily returns in overlapping blocks of L=20 trading days (B={args.bootstraps} iterations) to preserve autocorrelation and compute non-parametric 95% empirical confidence intervals.</li>
                <li><strong>Execution Assumptions</strong>: Commission=2.0 bps, Spread=1.0 bps, Slippage=2.0 bps, Long-only max position=100%, Max drawdown risk trigger=20%.</li>
            </ul>
        </div>

        <p style="text-align: right; color: #94a3b8; font-size: 12px; margin-top: 30px;"><em>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | AegisQuant Statistical Layer</em></p>
    </div>
</body>
</html>"""

        report_path.write_text(html_content, encoding="utf-8")
        print(f"\nBenchmark report written to: {report_path}")

    except Exception as e:
        print(f"Warning: report generation skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
