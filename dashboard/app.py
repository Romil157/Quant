"""Quant Research Dashboard (Streamlit).

A lightweight research dashboard that pulls market data via the platform's
data layer, runs buy-and-hold (or another simple strategy), and renders the
equity curve, drawdown, return distribution, and headline risk metrics.

Run with::

    streamlit run dashboard/app.py

Or via the helper menu::

    ./run.sh  (option 8)    # macOS / Linux
    run.bat  (option 8)     # Windows
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from quant.analytics import (
    calculate_cvar,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_var,
)
from quant.data import download_data
from quant.strategies import STRATEGY_REGISTRY, create_strategy


@st.cache_data(ttl=600)
def load_data(symbols: tuple[str, ...], start: str, end: str, provider: str) -> dict[str, pd.DataFrame]:
    """Streamlit-cached loader. The cache key must use tuples since lists are unhashable."""
    return download_data(list(symbols), start_date=start, end_date=end, provider=provider)


def render_equity_curve(equity: pd.Series):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date", yaxis_title="Portfolio Value",
        template="plotly_white", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown(equity: pd.Series):
    import plotly.graph_objects as go

    dd = calculate_drawdown(equity) * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
        line={"color": "red", "width": 1},
    ))
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date", yaxis_title="Drawdown",
        template="plotly_white", height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_returns_distribution(returns: pd.Series):
    import plotly.express as px

    fig = px.histogram(returns * 100, nbins=50, labels={"value": "Daily return (%)"},
                       title="Daily Returns Distribution",
                       template="plotly_white")
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.add_vline(x=returns.mean() * 100, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {returns.mean()*100:.3f}%")
    fig.update_layout(yaxis_title="Frequency", height=300)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Quant Research Dashboard", layout="wide")
    st.title("Quant Research Dashboard")
    st.caption("Research-only — not a live trading UI.")

    with st.sidebar:
        st.header("Configuration")
        default_symbols = ["AAPL", "MSFT", "GOOGL"]
        symbols = st.multiselect("Symbols", default_symbols, default=default_symbols)
        today = datetime.today().date()
        start = st.date_input("Start date", today - timedelta(days=365 * 2))
        end = st.date_input("End date", today)
        provider = st.selectbox("Data provider", ["mock", "parquet", "yfinance"], index=0)
        strategy_name = st.selectbox("Strategy", sorted(STRATEGY_REGISTRY), index=0)
        run_btn = st.button("Run")

    if not symbols:
        st.warning("Select at least one symbol.")
        return

    if not run_btn and "dashboard_run" not in st.session_state:
        st.info("Click **Run** in the sidebar to load data and compute analytics.")
        return

    st.session_state["dashboard_run"] = True
    start_str = start.isoformat()
    end_str = end.isoformat()
    try:
        data = load_data(tuple(symbols), start_str, end_str, provider)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Build an equity curve from the closes (a simple long-only price index).
    closes = pd.DataFrame({s: df["close"] for s, df in data.items()}).dropna()
    if closes.empty:
        st.warning("No price data after merging; pick a wider date range or different symbols.")
        return
    returns = closes.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)
    equity = (1 + portfolio_returns).cumprod() * 100_000

    # --- Metric cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Final equity", f"{equity.iloc[-1]:,.2f}")
    with col2:
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        st.metric("Total return", f"{total_return:+.2f}%")
    with col3:
        st.metric("Sharpe", f"{calculate_sharpe(portfolio_returns):.2f}")
    with col4:
        st.metric("Sortino", f"{calculate_sortino(portfolio_returns):.2f}")
    with col5:
        st.metric("Max DD", f"{calculate_max_drawdown(equity) * 100:.2f}%")

    # --- Tabs ---
    tab_equity, tab_dd, tab_dist, tab_risk, tab_strategy = st.tabs(
        ["Equity Curve", "Drawdown", "Returns Distribution", "Risk Metrics", "Strategy Signals"]
    )

    with tab_equity:
        render_equity_curve(equity)
    with tab_dd:
        render_drawdown(equity)
    with tab_dist:
        render_returns_distribution(portfolio_returns)
    with tab_risk:
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("Daily VaR 95%", f"{calculate_var(portfolio_returns) * 100:.2f}%")
        with rcol2:
            st.metric("Daily CVaR 95%", f"{calculate_cvar(portfolio_returns) * 100:.2f}%")
        with rcol3:
            st.metric("Volatility (ann.)", f"{portfolio_returns.std() * (252 ** 0.5) * 100:.2f}%")
        st.dataframe(portfolio_returns.describe().to_frame("Daily returns"))

    with tab_strategy:
        st.write(f"Available strategies: `{', '.join(sorted(STRATEGY_REGISTRY))}`")
        if strategy_name not in STRATEGY_REGISTRY:
            st.warning(f"Unknown strategy {strategy_name!r}.")
            return
        try:
            strategy = create_strategy(strategy_name)
            st.write(f"Strategy class: `{type(strategy).__name__}`")
        except Exception as e:
            st.error(f"Could not instantiate strategy: {e}")


if __name__ == "__main__":
    main()
