"""Unit tests for src.metrics using small deterministic fixtures."""

import numpy as np
import pandas as pd
import pytest

from src import metrics as m


def _series(values, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


# ---- returns --------------------------------------------------------------- #
def test_daily_returns_and_cumulative():
    prices = _series([100, 110, 99])
    r = m.daily_returns(prices)
    assert r.tolist() == pytest.approx([0.10, -0.10])
    assert m.cumulative_returns(r).iloc[-1] == pytest.approx(-0.01)
    assert m.total_return(r) == pytest.approx(-0.01)


def test_portfolio_value_buy_and_hold():
    prices = pd.DataFrame({"A": [10, 20], "B": [5, 5]}, index=pd.bdate_range("2024-01-01", periods=2))
    shares = pd.Series({"A": 2, "B": 4, "ZZZ": 100})  # ZZZ has no prices -> ignored
    assert m.portfolio_value(prices, shares).tolist() == [40.0, 60.0]
    with pytest.raises(ValueError):
        m.portfolio_value(prices, pd.Series({"ZZZ": 1}))


def test_annualized_return():
    r = _series([0.0] * 126)  # flat
    assert m.annualized_return(r) == pytest.approx(0.0)
    r = _series([(1.21) ** (1 / 252) - 1] * 252)  # exactly +21% over one year
    assert m.annualized_return(r) == pytest.approx(0.21)
    assert np.isnan(m.annualized_return(_series([])))


def test_monthly_return_table():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    r = pd.Series([0.10, 0.10, -0.50, 0.0], index=idx)
    table = m.monthly_return_table(r)
    assert table.loc[2024, "Jan"] == pytest.approx(0.21)
    assert table.loc[2024, "Feb"] == pytest.approx(-0.50)
    assert table.loc[2024, "Total"] == pytest.approx(1.21 * 0.5 - 1)
    assert np.isnan(table.loc[2024, "Mar"])
    assert m.monthly_return_table(r.iloc[0:0]).empty


# ---- volatility / sharpe --------------------------------------------------- #
def test_annualized_volatility_matches_numpy():
    r = _series([0.01, -0.01, 0.02, -0.02, 0.015])
    assert m.annualized_volatility(r) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))
    assert np.isnan(m.annualized_volatility(_series([0.01])))


def test_rolling_volatility_window():
    r = _series([0.01, -0.01] * 5)
    rv = m.rolling_volatility(r, 4)
    assert len(rv) == len(r) - 3
    assert rv.iloc[0] == pytest.approx(np.std(r.iloc[:4], ddof=1) * np.sqrt(252))
    assert m.rolling_volatility(r, 50).empty  # window longer than history


def test_sharpe_ratio():
    r = _series([0.01, -0.01, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert m.sharpe_ratio(r, 0.0) == pytest.approx(expected)
    assert m.sharpe_ratio(r, 0.05) < expected
    assert np.isnan(m.sharpe_ratio(_series([0.01] * 5)))  # zero variance


# ---- beta / alpha ---------------------------------------------------------- #
def test_beta_alpha_levered_benchmark():
    b = _series([0.01, -0.02, 0.015, 0.005, -0.01])
    beta, alpha = m.beta_alpha(2 * b, b)
    assert beta == pytest.approx(2.0)
    assert alpha == pytest.approx(0.0, abs=1e-12)


def test_beta_alpha_constant_outperformance():
    b = _series([0.01, -0.02, 0.015, 0.005, -0.01])
    beta, alpha = m.beta_alpha(b + 0.001, b)
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.001 * 252)


def test_beta_alpha_insufficient_data():
    b, a = m.beta_alpha(_series([0.01]), _series([0.01]))
    assert np.isnan(b) and np.isnan(a)
    b, a = m.beta_alpha(_series([0.01, 0.02, 0.03]), _series([0.01] * 3))  # zero benchmark var
    assert np.isnan(b) and np.isnan(a)


def test_tracking_error_and_excess_return():
    b = _series([0.01, 0.01, 0.01, 0.01])
    p = b + pd.Series([0.001, -0.001, 0.001, -0.001], index=b.index)
    assert m.tracking_error(p, b) == pytest.approx(
        np.std([0.001, -0.001, 0.001, -0.001], ddof=1) * np.sqrt(252)
    )
    assert m.tracking_error(b, b) == pytest.approx(0.0)
    assert m.excess_return(b + 0.01, b) == pytest.approx(1.02**4 - 1.01**4)


# ---- concentration --------------------------------------------------------- #
def test_hhi_and_effective_n():
    equal = pd.Series([0.25] * 4, index=list("ABCD"))
    assert m.hhi(equal) == pytest.approx(0.25)
    assert m.effective_n(equal) == pytest.approx(4.0)
    single = pd.Series([1.0], index=["A"])
    assert m.hhi(single) == pytest.approx(1.0)
    assert m.effective_n(single) == pytest.approx(1.0)


def test_weights_and_breaches():
    w = m.weights_from_values(pd.Series({"A": 60.0, "B": 30.0, "C": 10.0}))
    assert w.sum() == pytest.approx(1.0)
    assert m.concentration_breaches(w, 0.25).index.tolist() == ["A", "B"]
    assert m.concentration_breaches(w, 0.60).empty  # strictly greater than
    with pytest.raises(ValueError):
        m.weights_from_values(pd.Series({"A": 0.0}))


# ---- drawdowns ------------------------------------------------------------- #
def test_drawdown_series_and_max():
    r = _series([0.10, -0.50, 0.20])
    dd = m.drawdown_series(r)
    assert dd.tolist() == pytest.approx([0.0, -0.5, -0.4])
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    assert m.max_drawdown(_series([0.01, 0.02])) == 0.0


def test_drawdown_starting_below_initial_value():
    r = _series([-0.10, 0.05])
    assert m.max_drawdown(r) == pytest.approx(-0.10)


def test_drawdown_periods_recovered_and_open():
    r = _series([0.10, -0.10, 0.20, -0.30, 0.05])
    periods = m.drawdown_periods(r, top_n=5)
    assert len(periods) == 2
    worst = periods.iloc[0]
    assert worst["depth"] == pytest.approx(-0.30)
    assert pd.isna(worst["recovery"])  # still underwater at the end
    first = periods.iloc[1]
    assert first["depth"] == pytest.approx(-0.10)
    assert first["recovery"] == r.index[2]
    assert first["days_to_trough"] == (r.index[1] - r.index[0]).days
    assert len(m.drawdown_periods(r, top_n=1)) == 1
    assert m.drawdown_periods(_series([0.01, 0.02])).empty
    assert m.drawdown_periods(_series([])).empty


# ---- correlation ----------------------------------------------------------- #
def test_correlation_matrix_and_pairs():
    a = _series([0.01, -0.02, 0.03, -0.01, 0.02])
    df = pd.DataFrame({"A": a, "B": 2 * a, "C": -a})
    corr = m.correlation_matrix(df)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert corr.loc["A", "C"] == pytest.approx(-1.0)
    pairs = m.correlation_pairs(corr)
    assert len(pairs) == 3  # unique pairs only
    assert pairs.iloc[0]["correlation"] == pytest.approx(1.0)
    assert pairs.iloc[-1]["correlation"] == pytest.approx(-1.0)
    assert m.correlation_pairs(df[["A"]].corr()).empty  # single holding
