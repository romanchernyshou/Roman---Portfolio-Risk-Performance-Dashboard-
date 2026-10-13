"""Plotly figure builders. Each function returns a ``go.Figure`` and never calls Streamlit."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_LAYOUT = {"margin": {"l": 10, "r": 10, "t": 40, "b": 10}, "legend": {"orientation": "h", "y": -0.15}}


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**_LAYOUT)
    return fig


def cumulative_returns_chart(portfolio: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    """Cumulative return of the portfolio (and optionally the benchmark)."""
    if portfolio.empty:
        return _empty("Not enough data")
    fig = go.Figure()
    fig.add_scatter(x=portfolio.index, y=portfolio, name="Portfolio", line={"width": 2.5})
    if benchmark is not None and not benchmark.empty:
        fig.add_scatter(
            x=benchmark.index, y=benchmark, name="S&P 500 (SPY)", line={"dash": "dot"}
        )
    fig.update_layout(title="Cumulative return", yaxis_tickformat=".0%", hovermode="x unified", **_LAYOUT)
    return fig


def annual_returns_chart(table: pd.DataFrame) -> go.Figure:
    """Bar chart of calendar-year returns from ``metrics.monthly_return_table``."""
    if table.empty or "Total" not in table:
        return _empty("Not enough data")
    fig = px.bar(x=table.index.astype(str), y=table["Total"], labels={"x": "Year", "y": "Return"})
    fig.update_traces(marker_color=["#2e7d32" if v >= 0 else "#c62828" for v in table["Total"]])
    fig.update_layout(title="Calendar-year returns", yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def top_holdings_chart(weights: pd.Series, top_n: int, threshold: float) -> go.Figure:
    """Horizontal bars of the top-N weights with the concentration threshold marked."""
    top = weights.sort_values(ascending=False).head(top_n).sort_values()
    fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Weight", "y": ""})
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#c62828",
        annotation_text=f"Limit {threshold:.0%}",
    )
    fig.update_layout(title=f"Top {len(top)} holdings", xaxis_tickformat=".0%", **_LAYOUT)
    return fig


def allocation_treemap(weights: pd.Series, groups: dict[str, str] | None = None) -> go.Figure:
    """Treemap by holding, optionally nested under sector/asset class."""
    df = weights.rename("weight").rename_axis("holding").reset_index()
    if groups:
        df["group"] = df["holding"].map(groups).fillna("Unknown")
        fig = px.treemap(df, path=["group", "holding"], values="weight")
    else:
        fig = px.treemap(df, path=["holding"], values="weight")
    fig.update_traces(textinfo="label+percent root")
    fig.update_layout(**_LAYOUT)
    return fig


def allocation_pie(weights: pd.Series, title: str) -> go.Figure:
    """Donut chart of weights (index = labels)."""
    fig = px.pie(names=weights.index, values=weights.values, hole=0.45)
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def rolling_volatility_chart(series: dict[str, pd.Series], window: int) -> go.Figure:
    """Rolling annualized volatility lines. ``series`` maps label -> volatility series."""
    non_empty = {k: v for k, v in series.items() if not v.empty}
    if not non_empty:
        return _empty(f"Need more than {window} trading days of history")
    fig = go.Figure()
    for name, s in non_empty.items():
        is_port = name == "Portfolio"
        fig.add_scatter(
            x=s.index, y=s, name=name,
            line={"width": 3 if is_port else 1}, opacity=1 if is_port else 0.55,
        )
    fig.update_layout(
        title=f"Rolling {window}-day annualized volatility",
        yaxis_tickformat=".0%", hovermode="x unified", **_LAYOUT,
    )
    return fig


def drawdown_chart(portfolio_dd: pd.Series, benchmark_dd: pd.Series | None = None) -> go.Figure:
    """Underwater chart."""
    if portfolio_dd.empty:
        return _empty("Not enough data")
    fig = go.Figure()
    fig.add_scatter(
        x=portfolio_dd.index, y=portfolio_dd, name="Portfolio", fill="tozeroy", line={"width": 1.5}
    )
    if benchmark_dd is not None and not benchmark_dd.empty:
        fig.add_scatter(
            x=benchmark_dd.index, y=benchmark_dd, name="S&P 500 (SPY)", line={"dash": "dot"}
        )
    fig.update_layout(title="Drawdown (underwater)", yaxis_tickformat=".0%", hovermode="x unified", **_LAYOUT)
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Correlation heatmap with values printed in each cell."""
    if corr.shape[0] < 2:
        return _empty("Need at least two holdings for correlation analysis")
    fig = px.imshow(
        corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto"
    )
    fig.update_layout(title="Correlation of daily returns", **_LAYOUT)
    return fig
