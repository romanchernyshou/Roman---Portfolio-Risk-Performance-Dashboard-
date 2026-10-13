"""Tests for src.data. No network access: yfinance is mocked."""

import io

import numpy as np
import pandas as pd
import pytest

from src import data


# ---- holdings -------------------------------------------------------------- #
def test_parse_holdings_normalizes_and_merges_duplicates():
    df = pd.DataFrame(
        {
            " Ticker ": [" aapl", "AAPL", "msft"],
            "Shares": [10, 30, 5],
            "cost_basis": [100.0, 200.0, None],
        }
    )
    out = data.parse_holdings(df).set_index("ticker")
    assert out.loc["AAPL", "shares"] == 40
    assert out.loc["AAPL", "cost_basis"] == pytest.approx(175.0)  # share-weighted
    assert np.isnan(out.loc["MSFT", "cost_basis"])


def test_parse_holdings_errors():
    with pytest.raises(data.HoldingsError, match="Missing required"):
        data.parse_holdings(pd.DataFrame({"ticker": ["A"]}))
    with pytest.raises(data.HoldingsError, match="positive"):
        data.parse_holdings(pd.DataFrame({"ticker": ["A"], "shares": [-1]}))
    with pytest.raises(data.HoldingsError, match="No holdings"):
        data.parse_holdings(pd.DataFrame({"ticker": [None], "shares": [1]}))


def test_load_holdings_csv_from_buffer_and_bad_input():
    csv = io.StringIO("ticker,shares,purchase_date\nVTI,3,2021-01-05\n")
    out = data.load_holdings_csv(csv)
    assert out.loc[0, "ticker"] == "VTI"
    assert out.loc[0, "purchase_date"] == pd.Timestamp("2021-01-05")
    with pytest.raises(data.HoldingsError):
        data.load_holdings_csv(io.StringIO(""))


# ---- prices ---------------------------------------------------------------- #
def _fake_multi(tickers, n=5):
    idx = pd.bdate_range("2024-01-01", periods=n, tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    return pd.DataFrame(np.arange(n * len(cols), dtype=float).reshape(n, -1) + 1, index=idx, columns=cols)


def test_download_prices_multi_ticker(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: _fake_multi(["A", "SPY"]))
    out = data.download_prices(["a", "SPY", "A"])
    assert list(out.columns) == ["A", "SPY"]
    assert out.index.tz is None


def test_download_prices_single_ticker_flat_columns(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=3)
    raw = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [1, 1, 1]}, index=idx)
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: raw)
    out = data.download_prices(["AAPL"])
    assert list(out.columns) == ["AAPL"]


def test_download_prices_failure_and_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(data.yf, "download", boom)
    with pytest.raises(data.DataError, match="failed"):
        data.download_prices(["AAPL"])

    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(data.DataError, match="No price data"):
        data.download_prices(["AAPL"])


def test_clean_prices_drops_missing_and_ffills_gaps():
    idx = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4, 5, 6],  # one-day gap -> forward filled
            "B": [np.nan, np.nan, 3, 4, 5, 6],  # later start -> early rows trimmed
            "GHOST": [np.nan] * 6,  # no data -> excluded
        },
        index=idx,
        dtype=float,
    )
    clean, missing, warnings = data.clean_prices(prices, ["A", "B", "GHOST"], min_history=10)
    assert missing == ["GHOST"]
    assert list(clean.columns) == ["A", "B"]
    assert len(clean) == 4 and clean.index[0] == idx[2]
    assert clean.loc[idx[2], "A"] == 2  # forward-filled
    assert len(warnings) == 2  # missing ticker + short history


def test_clean_prices_errors():
    idx = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(data.DataError):
        data.clean_prices(pd.DataFrame({"A": [np.nan] * 3}, index=idx), ["A"])
    with pytest.raises(data.DataError):
        data.clean_prices(pd.DataFrame({"A": [np.nan, np.nan, 1.0]}, index=idx), ["A"])


# ---- sectors --------------------------------------------------------------- #
def test_fetch_sectors_with_fallbacks(monkeypatch):
    class FakeTicker:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            if self.t == "BAD":
                raise RuntimeError("rate limited")
            return {
                "AAPL": {"sector": "Technology"},
                "VTI": {"quoteType": "ETF"},
            }.get(self.t, {})

    monkeypatch.setattr(data.yf, "Ticker", FakeTicker)
    out = data.fetch_sectors(["AAPL", "VTI", "BAD", "NONE"])
    assert out == {
        "AAPL": "Technology",
        "VTI": "ETF / Fund",
        "BAD": "Unknown",
        "NONE": "Unknown",
    }
