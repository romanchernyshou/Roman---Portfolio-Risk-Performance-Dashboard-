"""End-to-end smoke test: run the Streamlit script headlessly with synthetic prices."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import data

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _fake_prices(tickers, period="max"):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", "2024-12-31")
    rets = rng.normal(0.0004, 0.012, size=(len(idx), len(tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list(tickers))


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(data, "download_prices", _fake_prices)
    monkeypatch.setattr(data, "fetch_sectors", lambda t: dict.fromkeys(t, "Technology"))


def test_app_runs_with_sample_portfolio(patched):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert len(at.tabs) == 7


def test_app_shows_error_when_download_fails(monkeypatch):
    def boom(tickers, period="max"):
        raise data.DataError("offline")

    monkeypatch.setattr(data, "download_prices", boom)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Could not load market data" in e.value for e in at.error)
