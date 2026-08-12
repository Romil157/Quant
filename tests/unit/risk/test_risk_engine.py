"""Unit tests for risk engine."""
import numpy as np
import pandas as pd

from quant.risk.engine import (
    RiskEngine,
    RiskLimits,
    StressScenario,
)


def test_risk_engine_basic():
    """Test basic risk engine."""
    engine = RiskEngine()
    assert engine.limits.max_drawdown == 0.20
    assert len(engine.stress_scenarios) > 0


def test_calculate_metrics():
    """Test risk metrics calculation."""
    engine = RiskEngine()

    # Create sample portfolio data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    portfolio_values = pd.Series(100000 * np.exp(np.cumsum(returns)))

    # Mock positions
    class MockPosition:
        def __init__(self, mv):
            self.market_value = mv

    positions = {
        'AAPL': MockPosition(20000),
        'MSFT': MockPosition(15000),
        'GOOGL': MockPosition(10000),
    }

    prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500}

    metrics = engine.calculate_metrics(
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        prices=prices,
    )

    assert metrics.portfolio_value > 0
    assert metrics.volatility >= 0
    assert metrics.var_95 >= 0
    assert metrics.cvar_95 >= metrics.var_95
    assert metrics.max_drawdown >= 0
    assert metrics.current_drawdown >= 0


def test_var_calculation():
    """Test VaR calculation."""
    engine = RiskEngine()

    returns = pd.Series([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])

    var_95 = engine._calculate_var(returns, 0.95)
    var_99 = engine._calculate_var(returns, 0.99)

    # VaR 99% should be more extreme than VaR 95%
    assert var_99 >= var_95
    assert var_95 > 0


def test_cvar_calculation():
    """Test CVaR calculation."""
    engine = RiskEngine()

    returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02])

    cvar_95 = engine._calculate_cvar(returns, 0.95)
    var_95 = engine._calculate_var(returns, 0.95)

    # CVaR should be >= VaR
    assert cvar_95 >= var_95


def test_drawdown_calculation():
    """Test drawdown calculation."""
    engine = RiskEngine()

    values = pd.Series([100, 110, 105, 120, 115, 100, 90, 95, 100])

    max_dd, current_dd = engine._calculate_drawdown(values)

    # Peak was 120, trough was 90, max DD = 30/120 = 25%
    assert abs(max_dd - 0.25) < 0.01
    # Current is 100, peak is 120, current DD = 20/120 = 16.67%
    assert abs(current_dd - 0.1667) < 0.01


def test_sharpe_calculation():
    """Test Sharpe ratio calculation."""
    engine = RiskEngine()

    # Positive returns
    returns = pd.Series([0.001] * 252)
    sharpe = engine._calculate_sharpe(returns)
    assert sharpe > 0

    # Zero variance
    returns_zero = pd.Series([0.0] * 252)
    sharpe_zero = engine._calculate_sharpe(returns_zero)
    assert sharpe_zero == 0


def test_sortino_calculation():
    """Test Sortino ratio calculation."""
    engine = RiskEngine()

    returns = pd.Series([0.001] * 200 + [-0.005] * 52)
    sortino = engine._calculate_sortino(returns)
    assert sortino >= 0


def test_calmar_calculation():
    """Test Calmar ratio calculation."""
    engine = RiskEngine()

    returns = pd.Series([0.0005] * 252)
    max_dd = 0.10
    calmar = engine._calculate_calmar(returns, max_dd)

    # Annual return = 0.0005 * 252 = 0.126, Calmar = 0.126 / 0.10 = 1.26
    assert abs(calmar - 1.26) < 0.1


def test_beta_correlation():
    """Test beta and correlation calculation."""
    engine = RiskEngine()

    np.random.seed(42)
    bench = pd.Series(np.random.normal(0.0005, 0.01, 252))
    port = bench * 1.2 + np.random.normal(0, 0.005, 252)

    beta, corr = engine._calculate_beta_correlation(port, bench)

    assert beta > 1.0  # Should be ~1.2
    assert corr > 0.9  # Should be highly correlated


def test_stress_scenarios():
    """Test default stress scenarios."""
    engine = RiskEngine()

    scenarios = engine.stress_scenarios
    assert len(scenarios) >= 10

    names = [s.name for s in scenarios]
    assert "Market Crash -5%" in names
    assert "Market Crash -10%" in names
    assert "Market Crash -20%" in names
    assert "2008 Crisis" in names
    assert "COVID Crash" in names


def test_run_stress_tests():
    """Test running stress tests."""
    engine = RiskEngine()

    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    portfolio_values = pd.Series(100000 * np.exp(np.cumsum(returns)))

    class MockPosition:
        def __init__(self, mv):
            self.market_value = mv

    positions = {
        'AAPL': MockPosition(20000),
        'MSFT': MockPosition(15000),
    }
    prices = {'AAPL': 150, 'MSFT': 300}

    results = engine.run_stress_tests(
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        prices=prices,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(engine.stress_scenarios)
    assert 'scenario' in results.columns
    assert 'pnl_pct' in results.columns

    # All scenarios should have negative PnL (stress)
    assert all(results['pnl_pct'] <= 0)


def test_check_limits():
    """Test risk limit checking."""
    limits = RiskLimits(
        max_drawdown=0.10,
        max_var_95=0.03,
        max_position=0.10,
        max_gross_exposure=1.0,
    )
    engine = RiskEngine(limits)

    # Create metrics within limits
    class MockMetrics:
        max_drawdown = 0.05
        var_95 = 0.02
        cvar_95 = 0.03
        gross_exposure = 0.8
        net_exposure = 0.3
        largest_position = 0.08
        concentration = 0.15
        leverage = 1.0
        sector_exposures = {'Tech': 0.20, 'Health': 0.15}

    metrics = MockMetrics()
    checks = engine.check_limits(metrics)

    assert all(checks.values())
    assert len(engine.get_limit_breaches(metrics)) == 0


def test_limit_breaches():
    """Test limit breach detection."""
    limits = RiskLimits(
        max_drawdown=0.10,
        max_var_95=0.03,
        max_position=0.10,
    )
    engine = RiskEngine(limits)

    class MockMetrics:
        max_drawdown = 0.15  # BREACH
        var_95 = 0.02
        cvar_95 = 0.03
        gross_exposure = 0.8
        net_exposure = 0.3
        largest_position = 0.15  # BREACH
        concentration = 0.15
        leverage = 1.0
        sector_exposures = {}

    metrics = MockMetrics()
    breaches = engine.get_limit_breaches(metrics)

    assert 'max_drawdown' in breaches
    assert 'max_position' in breaches
    assert len(breaches) == 2


def test_custom_stress_scenario():
    """Test custom stress scenario."""
    scenario = StressScenario(
        name="Custom Test",
        market_shock=-0.15,
        vol_shock=2.0,
        sector_shocks={'Tech': -0.25, 'Energy': 0.10},
    )

    assert scenario.name == "Custom Test"
    assert scenario.market_shock == -0.15
    assert scenario.vol_shock == 2.0
    assert scenario.sector_shocks['Tech'] == -0.25
