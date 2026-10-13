"""Pure financial metric calculations.

Nothing in this module imports Streamlit or touches the network, so every function
can be unit-tested with small deterministic fixtures.

Conventions
-----------
* Returns are simple (arithmetic) daily returns, e.g. ``0.01`` means +1%.
* Annualization uses ``TRADING_DAYS`` (252) periods per year.
* Risk-free rates are passed as annual decimals (``0.04`` means 4%).
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily returns from a price series/frame, dropping the leading NaN row."""
    return prices.pct_change().dropna(how="all")


def portfolio_value(prices: pd.DataFrame, shares: pd.Series) -> pd.Series:
    """Buy-and-hold portfolio value: ``sum(shares_i * price_i)`` for each date.

    Tickers present in ``shares`` but missing from ``prices`` are ignored.
    """
    cols = [t for t in shares.index if t in prices.columns]
    if not cols:
        raise ValueError("No overlapping tickers between prices and shares.")
    return (prices[cols] * shares[cols]).sum(axis=1)


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cumulative return curve starting from the first return observation."""
    return (1.0 + returns).cumprod() - 1.0


def total_return(returns: pd.Series) -> float:
    """Compounded total return over the whole series (NaN if empty)."""
    if len(returns) == 0:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return (CAGR-style) using 252 trading days per year."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / n) - 1.0


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Calendar table: rows are years, columns Jan..Dec plus a ``Total`` column.

    Partial months/years are compounded over the days that are available.
    """
    if len(returns) == 0:
        return pd.DataFrame()
    growth = 1.0 + returns
    monthly = growth.groupby([returns.index.year, returns.index.month]).prod() - 1.0
    table = monthly.unstack(level=1).reindex(columns=range(1, 13))
    table.columns = [calendar.month_abbr[m] for m in table.columns]
    table["Total"] = growth.groupby(returns.index.year).prod() - 1.0
    table.index.name = "Year"
    return table


# --------------------------------------------------------------------------- #
# Risk / benchmark statistics
# --------------------------------------------------------------------------- #
def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns (sample, ddof=1)."""
    if len(returns.dropna()) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio: mean daily excess return / daily std * sqrt(252)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / TRADING_DAYS
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std < 1e-12:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def beta_alpha(
    portfolio: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.0
) -> tuple[float, float]:
    """CAPM beta and annualized Jensen's alpha versus a benchmark.

    ``beta = cov(Rp - rf, Rb - rf) / var(Rb - rf)``
    ``alpha = (mean(Rp - rf) - beta * mean(Rb - rf)) * 252``
    Returns ``(nan, nan)`` when there is not enough overlapping data.
    """
    df = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan"), float("nan")
    rf = risk_free_rate / TRADING_DAYS
    p = df.iloc[:, 0] - rf
    b = df.iloc[:, 1] - rf
    var_b = b.var(ddof=1)
    if not np.isfinite(var_b) or var_b < 1e-16:
        return float("nan"), float("nan")
    beta = float(p.cov(b) / var_b)
    alpha = float((p.mean() - beta * b.mean()) * TRADING_DAYS)
    return beta, alpha


def tracking_error(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Annualized standard deviation of active (portfolio minus benchmark) returns."""
    df = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    active = df.iloc[:, 0] - df.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def excess_return(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Total compounded portfolio return minus total compounded benchmark return."""
    df = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if df.empty:
        return float("nan")
    return total_return(df.iloc[:, 0]) - total_return(df.iloc[:, 1])


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def weights_from_values(values: pd.Series) -> pd.Series:
    """Normalize position values to weights summing to 1."""
    total = values.sum()
    if total == 0 or not np.isfinite(total):
        raise ValueError("Total portfolio value must be non-zero.")
    return values / total


def hhi(weights: pd.Series) -> float:
    """Herfindahl-Hirschman Index on a 0-1 scale: ``sum(w_i^2)``."""
    return float((weights**2).sum())


def effective_n(weights: pd.Series) -> float:
    """Effective number of equal-weighted holdings: ``1 / HHI``."""
    h = hhi(weights)
    return float("nan") if h == 0 else 1.0 / h


def concentration_breaches(weights: pd.Series, threshold: float) -> pd.Series:
    """Positions whose weight is strictly above ``threshold`` (largest first)."""
    return weights[weights > threshold].sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Volatility, drawdowns, correlation
# --------------------------------------------------------------------------- #
def rolling_volatility(
    returns: pd.Series | pd.DataFrame, window: int
) -> pd.Series | pd.DataFrame:
    """Rolling annualized volatility over ``window`` trading days."""
    out = returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return out.dropna(how="all")


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak of wealth (starting wealth = 1). Values are <= 0."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline (negative number, 0.0 if none)."""
    if len(returns) == 0:
        return float("nan")
    return float(drawdown_series(returns).min())


def drawdown_periods(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """The ``top_n`` deepest drawdown episodes.

    Columns: ``peak``, ``trough``, ``recovery`` (NaT if not yet recovered), ``depth``,
    ``days_to_trough`` and ``total_days`` (peak to recovery, or to the last date).
    """
    cols = ["peak", "trough", "recovery", "depth", "days_to_trough", "total_days"]
    if len(returns) == 0:
        return pd.DataFrame(columns=cols)

    dd = drawdown_series(returns)
    idx = dd.index
    episodes: list[dict] = []
    in_dd = False
    peak_i = trough_i = 0

    def _record(recovery_i: int | None) -> None:
        end = idx[recovery_i] if recovery_i is not None else idx[-1]
        episodes.append(
            {
                "peak": idx[peak_i],
                "trough": idx[trough_i],
                "recovery": idx[recovery_i] if recovery_i is not None else pd.NaT,
                "depth": float(dd.iloc[trough_i]),
                "days_to_trough": (idx[trough_i] - idx[peak_i]).days,
                "total_days": (end - idx[peak_i]).days,
            }
        )

    for i, val in enumerate(dd.to_numpy()):
        if not in_dd:
            if val < 0:
                in_dd, peak_i, trough_i = True, max(i - 1, 0), i
        elif val >= 0:
            _record(i)
            in_dd = False
        elif val < dd.iloc[trough_i]:
            trough_i = i
    if in_dd:
        _record(None)

    if not episodes:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(episodes).sort_values("depth").head(top_n).reset_index(drop=True)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of asset returns."""
    return returns.corr()


def correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Unique asset pairs with their correlation, sorted from most to least correlated."""
    cols = ["asset_a", "asset_b", "correlation"]
    if corr.shape[0] < 2:
        return pd.DataFrame(columns=cols)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    rows = [
        (corr.index[i], corr.columns[j], float(corr.iat[i, j]))
        for i, j in zip(*np.where(mask), strict=True)
        if np.isfinite(corr.iat[i, j])
    ]
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )
