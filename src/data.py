"""Data loading: holdings parsing, price download (yfinance) and sector lookup.

Network access is isolated in :func:`download_prices` and :func:`fetch_sectors`
so the rest of the app (and the test suite) can run offline.
"""

from __future__ import annotations

from typing import IO

import pandas as pd
import yfinance as yf

BENCHMARK = "SPY"
REQUIRED_COLUMNS = ("ticker", "shares")


class HoldingsError(ValueError):
    """Raised when the holdings input is invalid."""


class DataError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable."""


# --------------------------------------------------------------------------- #
# Holdings
# --------------------------------------------------------------------------- #
def parse_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a holdings table.

    Required columns: ``ticker``, ``shares``. Optional: ``cost_basis`` (per share) and
    ``purchase_date``. Duplicate tickers are merged (shares summed, cost basis
    share-weighted, earliest purchase date kept).
    """
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise HoldingsError(f"Missing required column(s): {', '.join(missing)}.")

    out = out.dropna(subset=["ticker"])
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out[out["ticker"] != ""]
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce")
    if out.empty:
        raise HoldingsError("No holdings found.")
    if out["shares"].isna().any() or (out["shares"] <= 0).any():
        bad = out.loc[out["shares"].isna() | (out["shares"] <= 0), "ticker"].tolist()
        raise HoldingsError(f"Shares must be positive numbers (check: {', '.join(bad)}).")

    out["cost_basis"] = (
        pd.to_numeric(out["cost_basis"], errors="coerce") if "cost_basis" in out else float("nan")
    )
    out["purchase_date"] = (
        pd.to_datetime(out["purchase_date"], errors="coerce")
        if "purchase_date" in out
        else pd.NaT
    )

    def _merge(g: pd.DataFrame) -> pd.Series:
        total = g["shares"].sum()
        priced = g.dropna(subset=["cost_basis"])
        cost = (
            (priced["cost_basis"] * priced["shares"]).sum() / priced["shares"].sum()
            if not priced.empty
            else float("nan")
        )
        return pd.Series(
            {"shares": total, "cost_basis": cost, "purchase_date": g["purchase_date"].min()}
        )

    merged = pd.DataFrame(
        [{"ticker": t, **_merge(g).to_dict()} for t, g in out.groupby("ticker", sort=False)]
    )
    return merged.reset_index(drop=True)


def load_holdings_csv(source: str | IO) -> pd.DataFrame:
    """Read a holdings CSV from a path or file-like object and validate it."""
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to the UI
        raise HoldingsError(f"Could not read CSV: {exc}") from exc
    return parse_holdings(raw)


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull adjusted close prices out of a yfinance result (handles single/multi ticker)."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close = raw
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def download_prices(tickers: list[str], period: str = "max") -> pd.DataFrame:
    """Download split/dividend-adjusted daily close prices.

    Raises :class:`DataError` if the request fails or returns nothing.
    """
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    try:
        raw = yf.download(
            tickers, period=period, auto_adjust=True, progress=False, threads=True
        )
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"Price download failed: {exc}") from exc
    close = _extract_close(raw, tickers)
    if close.empty:
        raise DataError("No price data returned. Check your internet connection and tickers.")
    return close


def clean_prices(
    prices: pd.DataFrame, required: list[str], min_history: int = 60, ffill_limit: int = 5
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Align prices to a common trading calendar.

    * Tickers with no data at all are dropped (reported in ``missing``).
    * Small gaps (e.g. non-trading days for one asset) are forward-filled.
    * Rows before every remaining ticker has data are removed.

    Returns ``(clean_prices, missing_tickers, warnings)``.
    """
    present = [t for t in required if t in prices.columns and prices[t].notna().any()]
    missing = [t for t in required if t not in present]
    if not present:
        raise DataError("None of the requested tickers returned price data.")

    clean = prices[present].dropna(how="all").ffill(limit=ffill_limit).dropna()
    if len(clean) < 2:
        raise DataError("Not enough overlapping price history to compute returns.")

    warnings: list[str] = []
    if missing:
        warnings.append(f"No data found for: {', '.join(missing)}. They were excluded.")
    if len(clean) < min_history:
        warnings.append(
            f"Only {len(clean)} overlapping trading days available; "
            "rolling statistics and ratios may be unreliable."
        )
    return clean, missing, warnings


# --------------------------------------------------------------------------- #
# Sectors
# --------------------------------------------------------------------------- #
def fetch_sectors(tickers: list[str]) -> dict[str, str]:
    """Best-effort sector / asset-class lookup. Falls back to ``"Unknown"``."""
    result: dict[str, str] = {}
    for t in tickers:
        label = "Unknown"
        try:
            info = yf.Ticker(t).info or {}
            if info.get("sector"):
                label = str(info["sector"])
            elif str(info.get("quoteType", "")).upper() in {"ETF", "MUTUALFUND"}:
                label = "ETF / Fund"
        except Exception:  # noqa: BLE001 - lookups are non-critical
            pass
        result[t] = label
    return result
