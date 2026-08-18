"""Research report generation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quant.risk.engine import RiskEngine


@dataclass
class ReportConfig:
    """Report generation configuration."""
    output_dir: Path = Path("reports")
    include_plots: bool = True
    include_trades: bool = True
    include_risk: bool = True
    benchmark_symbols: list[str] = field(default_factory=list)
    format: str = "html"  # html, json, markdown

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


class ResearchReport:
    """Generate comprehensive research reports."""

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.risk_engine = RiskEngine()

    def generate_backtest_report(
        self,
        results: dict,
        experiment_name: str,
        strategy_name: str,
        parameters: dict,
        data_info: dict,
    ) -> Path:
        """Generate a complete backtest research report."""

        # Calculate all metrics
        returns = results.get('returns')
        equity = results.get('equity_curve')
        fills = results.get('fills', [])
        account_history = results.get('account_history', [])

        # Ensure equity/returns have DatetimeIndex for monthly resampling
        def _ensure_dtindex(series: pd.Series, account_history: list) -> pd.Series:
            if not isinstance(series, pd.Series) or isinstance(series.index, pd.DatetimeIndex):
                return series
            if account_history:
                timestamps = [a.timestamp for a in account_history]
                if len(timestamps) == len(series):
                    return series.set_axis(pd.DatetimeIndex(timestamps))
            # Fallback: business-day range ending at today
            return series.set_axis(pd.bdate_range(end=pd.Timestamp.now(), periods=len(series)))

        equity = _ensure_dtindex(equity, account_history)
        returns = _ensure_dtindex(returns, account_history)

        if returns is None or equity is None:
            raise ValueError("Results must contain returns and equity_curve")

        # Core metrics
        metrics = self._calculate_metrics(returns, equity, fills)

        # Risk metrics
        risk_metrics: dict[str, Any] | None = None
        if self.config.include_risk:
            # Mock positions for risk engine
            class MockPos:
                def __init__(self, mv): self.market_value = mv

            positions: dict[str, Any] = {}
            if account_history:
                last_acct = account_history[-1]
                for sym, pos in last_acct.positions.items():
                    if pos.quantity != 0:
                        positions[sym] = MockPos(pos.market_value)

            risk_metrics_obj = self.risk_engine.calculate_metrics(
                portfolio_values=equity,
                returns=returns,
                positions=positions,
                prices={},  # Would need actual prices
            )
            risk_metrics = asdict(risk_metrics_obj)

        # Generate plots
        plots = {}
        if self.config.include_plots:
            plots = self._generate_plots(returns, equity, fills, account_history, metrics)

        # Build report
        report = {
            'metadata': {
                'experiment_name': experiment_name,
                'strategy_name': strategy_name,
                'parameters': parameters,
                'data_info': data_info,
                'generated_at': datetime.now().isoformat(),
            },
            'summary': self._create_summary(metrics, risk_metrics),
            'performance': metrics,
            'risk': risk_metrics.__dict__ if hasattr(risk_metrics, '__dict__') else risk_metrics,
            'trades': self._create_trade_summary(fills) if self.config.include_trades else [],
            'plots': plots,
        }

        # Save report
        output_path = self._save_report(report, experiment_name)

        return output_path

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        fills: list,
    ) -> dict:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {}

        # Returns
        total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1)
        # CAGR: handle case where total_return <= -1 (equity would be negative/zero)
        # In practice this shouldn't happen with long-only + max_position < 1, but
        # mock data can violate it. Clamp to -99% return as worst case.
        cagr_base = max(1 + total_return, 0.01)
        cagr = float(cagr_base ** (252 / len(returns)) - 1)
        ann_vol = float(returns.std() * np.sqrt(252))

        # Risk-adjusted
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Drawdown
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        max_dd = float(abs(dd.min()))
        dd_duration = self._calc_dd_duration(dd)

        # Downside
        downside = returns[returns < 0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0
        sortino = float(returns.mean() * 252 / downside_vol) if downside_vol > 0 else 0
        calmar = float(returns.mean() * 252 / max_dd) if max_dd > 0 else 0

        # VaR/CVaR
        var_95 = float(-np.percentile(returns, 5))
        var_99 = float(-np.percentile(returns, 1))
        cvar_95 = float(-returns[returns <= -var_95].mean()) if (returns <= -var_95).any() else var_95

        # Trading
        num_trades = len(fills)
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

        if fills:
            # Calculate trade PnL (simplified)
            trade_pnls = [f.quantity * f.price for f in fills]  # Placeholder
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')

        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'max_dd_duration': dd_duration,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'downside_volatility': downside_vol,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
        }

    def _calc_dd_duration(self, dd: pd.Series) -> int:
        """Calculate max drawdown duration in bars."""
        in_dd = dd < 0
        if not in_dd.any():
            return 0

        durations = []
        current = 0
        for is_dd in in_dd:
            if is_dd:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)

        return int(max(durations)) if durations else 0

    def _create_summary(self, metrics: dict, risk_metrics: Any) -> dict:
        """Create executive summary."""
        return {
            'total_return_pct': f"{metrics.get('total_return', 0) * 100:.2f}%",
            'cagr_pct': f"{metrics.get('cagr', 0) * 100:.2f}%",
            'sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'sortino': f"{metrics.get('sortino_ratio', 0):.2f}",
            'max_dd_pct': f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
            'num_trades': metrics.get('num_trades', 0),
            'win_rate_pct': f"{metrics.get('win_rate', 0) * 100:.1f}%",
        }

    def _create_trade_summary(self, fills: list) -> list[dict]:
        """Create trade-level summary."""
        if not fills:
            return []

        trades = []
        for f in fills:
            trades.append({
                'timestamp': f.timestamp.isoformat() if hasattr(f.timestamp, 'isoformat') else str(f.timestamp),
                'symbol': f.symbol,
                'side': f.side.value if hasattr(f.side, 'value') else str(f.side),
                'quantity': f.quantity,
                'price': f.price,
                'commission': f.commission,
                'value': f.value,
            })

        return trades

    def _generate_plots(
        self,
        returns: pd.Series,
        equity: pd.Series,
        fills: list,
        account_history: list,
        metrics: dict,
    ) -> dict:
        """Generate Plotly figures as JSON."""
        plots = {}

        # 1. Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode='lines', name='Equity',
            line={"color": 'blue', "width": 1}
        ))
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            template='plotly_white',
            height=400,
        )
        plots['equity_curve'] = fig.to_json()

        # 2. Drawdown
        peak = equity.expanding().max()
        dd = (equity - peak) / peak * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode='lines', name='Drawdown',
            fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
            line={"color": 'red', "width": 1}
        ))
        fig.update_layout(
            title='Drawdown (%)',
            xaxis_title='Date',
            yaxis_title='Drawdown %',
            template='plotly_white',
            height=300,
        )
        plots['drawdown'] = fig.to_json()

        # 3. Rolling Sharpe (63-day)
        rolling_sharpe = returns.rolling(63).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            mode='lines', name='Rolling Sharpe (63d)',
            line={"color": 'green', "width": 1}
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title='Rolling Sharpe Ratio (63-day)',
            xaxis_title='Date',
            yaxis_title='Sharpe',
            template='plotly_white',
            height=300,
        )
        plots['rolling_sharpe'] = fig.to_json()

        # 4. Monthly returns heatmap
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_pct = monthly * 100

        # Pivot for heatmap
        monthly_df = monthly_pct.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
            y=pivot.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 2),
            texttemplate='%{text:.2f}%',
            textfont={"size": 10},
        ))
        fig.update_layout(
            title='Monthly Returns (%)',
            template='plotly_white',
            height=400,
        )
        plots['monthly_returns'] = fig.to_json()

        # 5. Returns distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Daily Returns',
            marker_color='blue',
            opacity=0.7,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.add_vline(x=returns.mean() * 100, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {returns.mean()*100:.3f}%")
        fig.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Return %',
            yaxis_title='Frequency',
            template='plotly_white',
            height=300,
        )
        plots['returns_dist'] = fig.to_json()

        # 6. Exposure over time (if account history available)
        if account_history:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            times = [a.timestamp for a in account_history]
            gross = [a.gross_exposure for a in account_history]
            net = [a.net_exposure for a in account_history]
            equity_vals = [a.total_value for a in account_history]

            fig.add_trace(go.Scatter(x=times, y=gross, name='Gross Exposure', line={"color": 'red'}), secondary_y=False)
            fig.add_trace(go.Scatter(x=times, y=net, name='Net Exposure', line={"color": 'blue'}), secondary_y=False)
            fig.add_trace(go.Scatter(x=times, y=equity_vals, name='Equity', line={"color": 'green'}), secondary_y=True)

            fig.update_layout(title='Exposure & Equity', template='plotly_white', height=400)
            fig.update_yaxes(title_text="Exposure", secondary_y=False)
            fig.update_yaxes(title_text="Equity", secondary_y=True)

            plots['exposure'] = fig.to_json()

        return plots

    def _save_report(self, report: dict, experiment_name: str) -> Path:
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = experiment_name.replace(' ', '_').replace('/', '_')

        if self.config.format == "json":
            path = self.config.output_dir / f"{safe_name}_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        elif self.config.format == "html":
            path = self.config.output_dir / f"{safe_name}_{timestamp}.html"
            html = self._render_html(report)
            with open(path, 'w') as f:
                f.write(html)

        elif self.config.format == "markdown":
            path = self.config.output_dir / f"{safe_name}_{timestamp}.md"
            md = self._render_markdown(report)
            with open(path, 'w') as f:
                f.write(md)

        else:
            path = self.config.output_dir / f"{safe_name}_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        return path

    def _render_html(self, report: dict) -> str:
        """Render report as HTML."""
        meta = report.get('metadata', {})
        summary = report.get('summary', {})
        perf = report.get('performance', {})
        plots = report.get('plots', {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research Report: {meta.get('experiment_name', 'Untitled')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
        .plot-container {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Research Report: {meta.get('experiment_name', 'Untitled')}</h1>
        <p><strong>Strategy:</strong> {meta.get('strategy_name', 'N/A')}</p>
        <p><strong>Generated:</strong> {meta.get('generated_at', 'N/A')}</p>

        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_return_pct', 'N/A')}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('cagr_pct', 'N/A')}</div>
                <div class="metric-label">CAGR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('sharpe', 'N/A')}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('sortino', 'N/A')}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('max_dd_pct', 'N/A')}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('num_trades', 'N/A')}</div>
                <div class="metric-label">Number of Trades</div>
            </div>
        </div>

        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Return</td><td class="{'positive' if perf.get('total_return', 0) >= 0 else 'negative'}">{perf.get('total_return', 0)*100:.2f}%</td></tr>
            <tr><td>CAGR</td><td class="{'positive' if perf.get('cagr', 0) >= 0 else 'negative'}">{perf.get('cagr', 0)*100:.2f}%</td></tr>
            <tr><td>Annual Volatility</td><td>{perf.get('annual_volatility', 0)*100:.2f}%</td></tr>
            <tr><td>Sharpe Ratio</td><td class="{'positive' if perf.get('sharpe_ratio', 0) >= 1 else 'negative'}">{perf.get('sharpe_ratio', 0):.2f}</td></tr>
            <tr><td>Sortino Ratio</td><td>{perf.get('sortino_ratio', 0):.2f}</td></tr>
            <tr><td>Calmar Ratio</td><td>{perf.get('calmar_ratio', 0):.2f}</td></tr>
            <tr><td>Max Drawdown</td><td class="negative">{perf.get('max_drawdown', 0)*100:.2f}%</td></tr>
            <tr><td>Max DD Duration</td><td>{perf.get('max_dd_duration', 0)} bars</td></tr>
            <tr><td>VaR (95%)</td><td class="negative">{perf.get('var_95', 0)*100:.2f}%</td></tr>
            <tr><td>CVaR (95%)</td><td class="negative">{perf.get('cvar_95', 0)*100:.2f}%</td></tr>
            <tr><td>Win Rate</td><td>{perf.get('win_rate', 0)*100:.1f}%</td></tr>
            <tr><td>Profit Factor</td><td>{perf.get('profit_factor', 0):.2f}</td></tr>
            <tr><td>Number of Trades</td><td>{perf.get('num_trades', 0)}</td></tr>
        </table>

        <h2>Charts</h2>
"""

        # Add plots
        for name, plot_json in plots.items():
            div_id = f"plot_{name}"
            html += f"""
        <div class="plot-container">
            <h3>{name.replace('_', ' ').title()}</h3>
            <div id="{div_id}"></div>
        </div>
        <script>
            var plotData = {plot_json};
            Plotly.newPlot('{div_id}', plotData.data, plotData.layout, {{responsive: true}});
        </script>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def _render_markdown(self, report: dict) -> str:
        """Render report as Markdown."""
        meta = report.get('metadata', {})
        summary = report.get('summary', {})
        perf = report.get('performance', {})

        md = f"""# Research Report: {meta.get('experiment_name', 'Untitled')}

**Strategy:** {meta.get('strategy_name', 'N/A')}
**Generated:** {meta.get('generated_at', 'N/A')}

## Summary

| Metric | Value |
|--------|-------|
| Total Return | {summary.get('total_return_pct', 'N/A')} |
| CAGR | {summary.get('cagr_pct', 'N/A')} |
| Sharpe Ratio | {summary.get('sharpe', 'N/A')} |
| Sortino Ratio | {summary.get('sortino', 'N/A')} |
| Max Drawdown | {summary.get('max_dd_pct', 'N/A')} |
| Number of Trades | {summary.get('num_trades', 'N/A')} |
| Win Rate | {summary.get('win_rate_pct', 'N/A')} |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | {perf.get('total_return', 0)*100:.2f}% |
| CAGR | {perf.get('cagr', 0)*100:.2f}% |
| Annual Volatility | {perf.get('annual_volatility', 0)*100:.2f}% |
| Sharpe Ratio | {perf.get('sharpe_ratio', 0):.2f} |
| Sortino Ratio | {perf.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {perf.get('calmar_ratio', 0):.2f} |
| Max Drawdown | {perf.get('max_drawdown', 0)*100:.2f}% |
| Max DD Duration | {perf.get('max_dd_duration', 0)} bars |
| VaR (95%) | {perf.get('var_95', 0)*100:.2f}% |
| CVaR (95%) | {perf.get('cvar_95', 0)*100:.2f}% |
| Downside Volatility | {perf.get('downside_volatility', 0)*100:.2f}% |
| Win Rate | {perf.get('win_rate', 0)*100:.1f}% |
| Avg Win | {perf.get('avg_win', 0):.4f} |
| Avg Loss | {perf.get('avg_loss', 0):.4f} |
| Profit Factor | {perf.get('profit_factor', 0):.2f} |
| Number of Trades | {perf.get('num_trades', 0)} |

## Parameters

```json
{json.dumps(meta.get('parameters', {}), indent=2)}
```

## Data Info

```json
{json.dumps(meta.get('data_info', {}), indent=2)}
```
"""
        return md
