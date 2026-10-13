"""Streamlit UI for the portfolio analytics dashboard (UI only; logic lives in src/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts, data, metrics

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_portfolio.csv"
RANGES = ["1M", "6M", "YTD", "1Y", "3Y", "5Y", "Max"]

st.set_page_config(page_title="Portfolio Analytics", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Cached data access (thin wrappers so the data layer stays Streamlit-free)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner="Downloading price history...")
def cached_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return data.download_prices(list(tickers))


@st.cache_data(ttl=86400, show_spinner="Looking up sectors...")
def cached_sectors(tickers: tuple[str, ...]) -> dict[str, str]:
    return data.fetch_sectors(list(tickers))


def slice_range(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep only rows inside the selected date range, ending at the last available date."""
    end = df.index[-1]
    offsets = {
        "1M": pd.DateOffset(months=1),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5),
    }
    if label == "Max":
        return df
    start = pd.Timestamp(end.year, 1, 1) if label == "YTD" else end - offsets[label]
    return df[df.index >= start]


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.2%}"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


# --------------------------------------------------------------------------- #
# Sidebar: inputs
# --------------------------------------------------------------------------- #
st.sidebar.title("Portfolio inputs")
mode = st.sidebar.radio("Holdings source", ["Sample portfolio", "Upload CSV", "Manual entry"])

holdings: pd.DataFrame | None = None
try:
    if mode == "Sample portfolio":
        holdings = data.load_holdings_csv(str(SAMPLE_PATH))
    elif mode == "Upload CSV":
        upload = st.sidebar.file_uploader("CSV with: ticker, shares [, cost_basis, purchase_date]", type="csv")
        if upload is not None:
            holdings = data.load_holdings_csv(upload)
    else:
        template = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "shares": [10.0, 5.0], "cost_basis": [None, None]}
        )
        edited = st.sidebar.data_editor(template, num_rows="dynamic", key="manual")
        holdings = data.parse_holdings(edited)
except data.HoldingsError as exc:
    st.sidebar.error(str(exc))

date_range = st.sidebar.selectbox("Date range", RANGES, index=3)
risk_free = st.sidebar.number_input("Risk-free rate (annual, %)", 0.0, 20.0, 4.0, 0.25) / 100
threshold = st.sidebar.slider("Concentration warning threshold", 0.05, 1.0, 0.20, 0.05)
top_n = st.sidebar.slider("Top-N holdings to show", 3, 20, 10)

st.title("📈 Portfolio Analytics Dashboard")
st.caption("Hypothetical performance of your *current* holdings over the selected window. Not investment advice.")

if holdings is None or holdings.empty:
    st.info("Choose a holdings source in the sidebar to get started.")
    st.stop()

# --------------------------------------------------------------------------- #
# Load and prepare data
# --------------------------------------------------------------------------- #
tickers = holdings["ticker"].tolist()
try:
    raw_prices = cached_prices(tuple(dict.fromkeys([*tickers, data.BENCHMARK])))
    prices_all, missing, warnings = data.clean_prices(raw_prices, [*tickers, data.BENCHMARK])
except (data.DataError, ValueError) as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

if data.BENCHMARK in missing:
    st.error("Benchmark (SPY) data is unavailable right now. Please try again later.")
    st.stop()

holdings = holdings[holdings["ticker"].isin(prices_all.columns)].reset_index(drop=True)
if holdings.empty:
    st.error("None of your tickers returned usable price data.")
    st.stop()
for w in warnings:
    st.warning(w)

prices = slice_range(prices_all, date_range)
if len(prices) < 3:
    st.error("Too few trading days in this range. Pick a longer date range.")
    st.stop()

hold_tickers = holdings["ticker"].tolist()
shares = holdings.set_index("ticker")["shares"]
asset_returns = metrics.daily_returns(prices[hold_tickers])
port_returns = metrics.daily_returns(metrics.portfolio_value(prices, shares))
bench_returns = metrics.daily_returns(prices[data.BENCHMARK])

latest = prices[hold_tickers].iloc[-1]
values = latest * shares
weights = metrics.weights_from_values(values)

st.caption(
    f"Data window: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d} "
    f"({len(prices)} trading days). History starts when the newest holding began trading."
)

# Headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio value", f"${values.sum():,.0f}")
c2.metric("Total return", pct(metrics.total_return(port_returns)))
c3.metric("Annualized volatility", pct(metrics.annualized_volatility(port_returns)))
c4.metric("Max drawdown", pct(metrics.max_drawdown(port_returns)))

tabs = st.tabs(
    ["Returns", "S&P 500 benchmark", "Concentration", "Allocation", "Volatility", "Drawdowns", "Correlation"]
)

# 1. Returns ---------------------------------------------------------------- #
with tabs[0]:
    st.plotly_chart(charts.cumulative_returns_chart(metrics.cumulative_returns(port_returns)), width="stretch")
    table = metrics.monthly_return_table(port_returns)
    left, right = st.columns([3, 2])
    left.subheader("Monthly returns")
    left.dataframe(table.style.format("{:.1%}", na_rep=""), width="stretch")
    right.plotly_chart(charts.annual_returns_chart(table), width="stretch")

# 2. Benchmark -------------------------------------------------------------- #
with tabs[1]:
    beta, alpha = metrics.beta_alpha(port_returns, bench_returns, risk_free)
    cols = st.columns(5)
    cols[0].metric("Excess return vs SPY", pct(metrics.excess_return(port_returns, bench_returns)))
    cols[1].metric("Beta", num(beta))
    cols[2].metric("Alpha (annual)", pct(alpha))
    cols[3].metric("Tracking error", pct(metrics.tracking_error(port_returns, bench_returns)))
    cols[4].metric(
        "Sharpe (Portfolio / SPY)",
        f"{num(metrics.sharpe_ratio(port_returns, risk_free))} / "
        f"{num(metrics.sharpe_ratio(bench_returns, risk_free))}",
    )
    st.plotly_chart(
        charts.cumulative_returns_chart(
            metrics.cumulative_returns(port_returns), metrics.cumulative_returns(bench_returns)
        ),
        width="stretch",
    )

# 3. Concentration ---------------------------------------------------------- #
with tabs[2]:
    h = metrics.hhi(weights)
    cols = st.columns(3)
    cols[0].metric("HHI (0-1)", num(h, 3))
    cols[1].metric("Effective # of holdings", num(metrics.effective_n(weights), 1))
    cols[2].metric("Largest position", f"{weights.idxmax()} ({weights.max():.1%})")
    breaches = metrics.concentration_breaches(weights, threshold)
    if len(weights) == 1:
        st.warning("Single-holding portfolio: fully concentrated.")
    if not breaches.empty:
        st.warning(
            f"Positions above the {threshold:.0%} threshold: "
            + ", ".join(f"{t} ({w:.1%})" for t, w in breaches.items())
        )
    st.plotly_chart(charts.top_holdings_chart(weights, top_n, threshold), width="stretch")

    detail = pd.DataFrame({"Shares": shares, "Price": latest, "Value": values, "Weight": weights})
    cost = holdings.set_index("ticker")["cost_basis"]
    if cost.notna().any():
        detail["Cost basis"] = cost
        detail["Unrealized P&L"] = (latest - cost) * shares
        detail["Unrealized %"] = latest / cost - 1
    st.dataframe(
        detail.sort_values("Value", ascending=False).style.format(
            {"Shares": "{:,.2f}", "Price": "${:,.2f}", "Value": "${:,.0f}", "Weight": "{:.1%}",
             "Cost basis": "${:,.2f}", "Unrealized P&L": "${:,.0f}", "Unrealized %": "{:.1%}"},
            na_rep="",
        ),
        width="stretch",
    )

# 4. Allocation ------------------------------------------------------------- #
with tabs[3]:
    try:
        sectors = cached_sectors(tuple(hold_tickers))
    except Exception:  # noqa: BLE001 - sector data is non-critical
        sectors = dict.fromkeys(hold_tickers, "Unknown")
        st.warning("Sector lookup failed; showing 'Unknown'.")
    by_sector = weights.groupby(weights.index.map(sectors)).sum().sort_values(ascending=False)
    a, b = st.columns(2)
    a.plotly_chart(charts.allocation_treemap(weights, sectors), width="stretch")
    b.plotly_chart(charts.allocation_pie(by_sector, "By sector / asset class"), width="stretch")

# 5. Volatility ------------------------------------------------------------- #
with tabs[4]:
    window = st.radio("Rolling window", [30, 90], horizontal=True)
    series = {"Portfolio": metrics.rolling_volatility(port_returns, window)}
    for t in hold_tickers:
        series[t] = metrics.rolling_volatility(asset_returns[t], window)
    st.plotly_chart(charts.rolling_volatility_chart(series, window), width="stretch")
    vol_table = pd.Series(
        {t: metrics.annualized_volatility(asset_returns[t]) for t in hold_tickers}, name="Annualized volatility"
    ).sort_values(ascending=False)
    st.dataframe(vol_table.to_frame().style.format("{:.1%}"), width="stretch")

# 6. Drawdowns -------------------------------------------------------------- #
with tabs[5]:
    st.plotly_chart(
        charts.drawdown_chart(metrics.drawdown_series(port_returns), metrics.drawdown_series(bench_returns)),
        width="stretch",
    )
    periods = metrics.drawdown_periods(port_returns, 5)
    st.subheader("Top 5 worst drawdowns")
    if periods.empty:
        st.success("No drawdowns in this window.")
    else:
        show = periods.rename(
            columns={"days_to_trough": "Days to trough", "total_days": "Total days"}
        ).rename(columns=str.capitalize)
        for col in ("Peak", "Trough", "Recovery"):
            show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d").fillna("not recovered")
        st.dataframe(show.style.format({"Depth": "{:.1%}"}), hide_index=True, width="stretch")

# 7. Correlation ------------------------------------------------------------ #
with tabs[6]:
    corr = metrics.correlation_matrix(asset_returns)
    st.plotly_chart(charts.correlation_heatmap(corr), width="stretch")
    pairs = metrics.correlation_pairs(corr)
    if pairs.empty:
        st.info("Add at least two holdings to see pairwise correlations.")
    else:
        left_col, right_col = st.columns(2)
        left_col.subheader("Most correlated")
        left_col.dataframe(pairs.head(5).style.format({"correlation": "{:.2f}"}), hide_index=True, width="stretch")
        right_col.subheader("Least correlated")
        right_col.dataframe(
            pairs.tail(5).iloc[::-1].style.format({"correlation": "{:.2f}"}), hide_index=True, width="stretch"
        )

st.divider()
st.caption("For educational purposes only; not investment advice. Prices via Yahoo Finance (yfinance).")
