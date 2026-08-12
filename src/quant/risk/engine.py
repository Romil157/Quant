"""Risk engine with stress testing and portfolio analytics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    market_shock: float = 0.0          # Market return shock (e.g., -0.20 for -20%)
    vol_shock: float = 1.0             # Volatility multiplier (e.g., 2.0 for 2x)
    correlation_shock: float = 1.0     # Correlation multiplier
    liquidity_shock: float = 1.0       # Liquidity multiplier (spread widening)
    sector_shocks: dict[str, float] = field(default_factory=dict)  # Sector-specific shocks


@dataclass
class RiskLimits:
    """Portfolio risk limits."""
    max_drawdown: float = 0.20
    max_var_95: float = 0.05           # Max 95% VaR as fraction of portfolio
    max_cvar_95: float = 0.07          # Max 95% CVaR
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.5
    max_position: float = 0.10
    max_sector_exposure: float = 0.30
    max_factor_exposure: dict[str, float] = field(default_factory=dict)
    max_turnover: float = 1.0
    max_concentration: float = 0.20    # Max single position
    max_leverage: float = 1.5


@dataclass
class RiskMetrics:
    """Calculated risk metrics."""
    portfolio_value: float
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    concentration: float
    largest_position: float
    sector_exposures: dict[str, float]
    factor_exposures: dict[str, float]


class RiskEngine:
    """Risk calculation and monitoring engine."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.stress_scenarios = self._default_stress_scenarios()

    def _default_stress_scenarios(self) -> list[StressScenario]:
        """Create default stress scenarios."""
        return [
            StressScenario("Market Crash -5%", market_shock=-0.05),
            StressScenario("Market Crash -10%", market_shock=-0.10),
            StressScenario("Market Crash -20%", market_shock=-0.20),
            StressScenario("Market Crash -30%", market_shock=-0.30),
            StressScenario("Vol Spike 50%", vol_shock=1.5),
            StressScenario("Vol Spike 100%", vol_shock=2.0),
            StressScenario("Vol Spike 200%", vol_shock=3.0),
            StressScenario("Liquidity Crisis 50%", liquidity_shock=1.5),
            StressScenario("Liquidity Crisis 100%", liquidity_shock=2.0),
            StressScenario("Correlation Breakdown", correlation_shock=1.5),
            StressScenario("1987 Crash", market_shock=-0.22, vol_shock=3.0, correlation_shock=2.0),
            StressScenario("2008 Crisis", market_shock=-0.40, vol_shock=2.5, correlation_shock=1.8, liquidity_shock=2.0),
            StressScenario("COVID Crash", market_shock=-0.35, vol_shock=4.0, correlation_shock=1.5),
            StressScenario("Flash Crash", market_shock=-0.10, vol_shock=5.0, liquidity_shock=3.0),
        ]

    def calculate_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        positions: dict,
        prices: dict[str, float],
        benchmark_returns: pd.Series | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""

        # Portfolio value and returns
        port_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0
        port_returns = returns.dropna()

        # Volatility (annualized)
        vol = port_returns.std() * np.sqrt(252) if len(port_returns) > 1 else 0

        # VaR and CVaR
        var_95 = self._calculate_var(port_returns, 0.95)
        var_99 = self._calculate_var(port_returns, 0.99)
        cvar_95 = self._calculate_cvar(port_returns, 0.95)
        cvar_99 = self._calculate_cvar(port_returns, 0.99)

        # Drawdown
        max_dd, current_dd = self._calculate_drawdown(portfolio_values)

        # Risk-adjusted ratios
        sharpe = self._calculate_sharpe(port_returns)
        sortino = self._calculate_sortino(port_returns)
        calmar = self._calculate_calmar(port_returns, max_dd)

        # Beta and correlation vs benchmark
        beta, corr = self._calculate_beta_correlation(port_returns, benchmark_returns)

        # Exposures
        gross_exp = sum(abs(p.market_value) for p in positions.values())
        net_exp = sum(p.market_value for p in positions.values())
        leverage = gross_exp / port_value if port_value > 0 else 0

        # Concentration
        position_values = {s: abs(p.market_value) for s, p in positions.items()}
        sum(position_values.values())
        largest_pos = max(position_values.values()) / port_value if port_value > 0 and position_values else 0
        concentration = sum((v/port_value)**2 for v in position_values.values()) if port_value > 0 else 0

        # Sector exposures
        sector_exp: dict[str, float] = {}
        if sector_map:
            for symbol, pos in positions.items():
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    sector_exp[sector] = sector_exp.get(sector, 0.0) + abs(pos.market_value)
            sector_exp = {s: v/port_value for s, v in sector_exp.items()} if port_value > 0 else {}

        return RiskMetrics(
            portfolio_value=port_value,
            volatility=vol,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            beta=beta,
            correlation=corr,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            leverage=leverage,
            concentration=concentration,
            largest_position=largest_pos,
            sector_exposures=sector_exp,
            factor_exposures={},  # TODO: Factor model integration
        )

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk (historical)."""
        if len(returns) == 0:
            return 0.0
        return float(-np.percentile(returns, (1 - confidence) * 100))

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        var = self._calculate_var(returns, confidence)
        tail = returns[returns <= -var]
        return -tail.mean() if len(tail) > 0 else var

    def _calculate_drawdown(self, values: pd.Series) -> tuple[float, float]:
        """Calculate max and current drawdown."""
        if len(values) == 0:
            return 0.0, 0.0
        peak = values.expanding().max()
        dd = (values - peak) / peak
        max_dd = dd.min()
        current_dd = dd.iloc[-1]
        return abs(max_dd), abs(current_dd)

    def _calculate_sharpe(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns.mean() * 252 - risk_free
        return float(excess / (returns.std() * np.sqrt(252)))

    def _calculate_sortino(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate Sortino ratio (annualized)."""
        if len(returns) < 2:
            return 0.0
        excess = returns.mean() * 252 - risk_free
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        return float(excess / downside_std) if downside_std > 0 else 0.0

    def _calculate_calmar(self, returns: pd.Series, max_dd: float) -> float:
        """Calculate Calmar ratio."""
        if max_dd == 0:
            return 0.0
        ann_return = returns.mean() * 252
        return float(ann_return / max_dd)

    def _calculate_beta_correlation(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None,
    ) -> tuple[float, float]:
        """Calculate beta and correlation vs benchmark."""
        if benchmark is None or len(returns) < 2 or len(benchmark) < 2:
            return 1.0, 0.0

        # Align series
        aligned = pd.concat([returns, benchmark], axis=1, join='inner').dropna()
        if len(aligned) < 2:
            return 1.0, 0.0

        r_port = aligned.iloc[:, 0]
        r_bench = aligned.iloc[:, 1]

        corr = r_port.corr(r_bench)
        beta = r_port.cov(r_bench) / r_bench.var() if r_bench.var() > 0 else 1.0

        return beta, corr

    def run_stress_tests(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        positions: dict,
        prices: dict[str, float],
        sector_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Run all stress scenarios."""
        base_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0
        base_returns = returns.dropna()

        results = []
        for scenario in self.stress_scenarios:
            stressed_value = self._apply_scenario(
                base_value, base_returns, positions, prices, scenario
            )
            pnl = stressed_value - base_value
            pnl_pct = pnl / base_value if base_value > 0 else 0

            results.append({
                'scenario': scenario.name,
                'base_value': base_value,
                'stressed_value': stressed_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'market_shock': scenario.market_shock,
                'vol_shock': scenario.vol_shock,
            })

        return pd.DataFrame(results)

    def _apply_scenario(
        self,
        base_value: float,
        returns: pd.Series,
        positions: dict,
        prices: dict[str, float],
        scenario: StressScenario,
    ) -> float:
        """Apply stress scenario to portfolio."""
        if base_value == 0:
            return 0.0

        # Market shock: apply to all positions
        market_impact = base_value * scenario.market_shock

        # Volatility shock: increase VaR
        if len(returns) > 0:
            stressed_returns = returns * scenario.vol_shock
            var_increase = self._calculate_var(stressed_returns, 0.95) - self._calculate_var(returns, 0.95)
            vol_impact = abs(var_increase) * base_value
        else:
            vol_impact = 0

        # Liquidity shock: widen spreads
        liquidity_impact = 0
        for symbol, pos in positions.items():
            if symbol in prices:
                notional = abs(pos.market_value)
                # Estimate spread cost increase
                base_spread = 0.001  # 10 bps default
                stressed_spread = base_spread * scenario.liquidity_shock
                liquidity_impact += notional * (stressed_spread - base_spread)

        # Sector shocks
        sector_impact = 0
        if scenario.sector_shocks:
            # Would need sector mapping
            pass

        total_pnl = market_impact - vol_impact - liquidity_impact + sector_impact
        return base_value + total_pnl

    def check_limits(self, metrics: RiskMetrics) -> dict[str, bool]:
        """Check all risk limits."""
        return {
            'max_drawdown': metrics.max_drawdown <= self.limits.max_drawdown,
            'var_95': metrics.var_95 <= self.limits.max_var_95,
            'cvar_95': metrics.cvar_95 <= self.limits.max_cvar_95,
            'gross_exposure': metrics.gross_exposure <= self.limits.max_gross_exposure,
            'net_exposure': metrics.net_exposure <= self.limits.max_net_exposure,
            'max_position': metrics.largest_position <= self.limits.max_position,
            'concentration': metrics.concentration <= self.limits.max_concentration,
            'leverage': metrics.leverage <= self.limits.max_leverage,
            'sector_exposure': all(
                v <= self.limits.max_sector_exposure
                for v in metrics.sector_exposures.values()
            ),
        }

    def get_limit_breaches(self, metrics: RiskMetrics) -> list[str]:
        """Get list of breached limits."""
        checks = self.check_limits(metrics)
        return [k for k, v in checks.items() if not v]
