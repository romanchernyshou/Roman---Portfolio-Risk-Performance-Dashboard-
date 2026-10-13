# Portfolio Analytics Dashboard

[![CI](https://github.com/YOUR_USER/portfolio-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/portfolio-dashboard/actions)

A Streamlit dashboard for analyzing an equity/ETF portfolio: historical returns, S&P 500 benchmarking,
concentration, allocation, volatility, drawdowns and correlation. Built with Python, Streamlit, Pandas,
NumPy and Plotly. Prices come from Yahoo Finance via `yfinance`.

> _Screenshots: add images to `docs/` and reference them here, e.g. `![Returns tab](docs/returns.png)`._

## Features

| Tab | What you get |
| --- | --- |
| Returns | Cumulative return, monthly return table, calendar-year bars; range 1M / 6M / YTD / 1Y / 3Y / 5Y / Max |
| S&P 500 benchmark | Portfolio vs. SPY, excess return, beta, alpha, tracking error, Sharpe |
| Concentration | Top-N weights, HHI, effective number of holdings, configurable threshold warning, P&L table |
| Allocation | Treemap by holding nested in sector, plus sector / asset-class donut |
| Volatility | Rolling 30/90-day annualized volatility, portfolio and per holding |
| Drawdowns | Underwater chart, max drawdown, top 5 worst drawdowns with durations and recovery dates |
| Correlation | Heatmap plus the most and least correlated pairs |

## Quick start

```bash
git clone https://github.com/YOUR_USER/portfolio-dashboard.git
cd portfolio-dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens with `data/sample_portfolio.csv`. Switch the sidebar to **Upload CSV** or **Manual entry**
to use your own holdings.

### Holdings CSV format

```csv
ticker,shares,cost_basis,purchase_date
AAPL,40,150.00,2021-03-15
VTI,35,200.00,2020-12-01
```

`ticker` and `shares` are required. `cost_basis` (per share) and `purchase_date` are optional.
Duplicate tickers are merged (shares summed, cost basis share-weighted).

## Project layout

```
app.py              Streamlit UI only
src/metrics.py      Pure financial calculations (no Streamlit, no network)
src/data.py         Holdings parsing, yfinance downloads, price cleaning, sector lookup
src/charts.py       Plotly figure builders
tests/              pytest suite (offline; yfinance is mocked)
data/               Sample portfolio
.github/workflows/  CI: ruff + pytest
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Metric definitions

Let `r_t` be the simple daily return, `rf` the annual risk-free rate, and `N = 252` trading days per year.

| Metric | Definition |
| --- | --- |
| Portfolio value | `sum(shares_i * price_i,t)` (buy-and-hold, current share counts) |
| Cumulative return | `prod(1 + r_t) - 1` |
| Annualized return | `prod(1 + r_t)^(N / n) - 1` for `n` observations |
| Annualized volatility | `std(r_t, ddof=1) * sqrt(N)` |
| Sharpe ratio | `mean(r_t - rf/N) / std(r_t - rf/N) * sqrt(N)` |
| Beta | `cov(Rp - rf/N, Rb - rf/N) / var(Rb - rf/N)` |
| Alpha (annualized) | `(mean(Rp - rf/N) - beta * mean(Rb - rf/N)) * N` |
| Tracking error | `std(Rp - Rb, ddof=1) * sqrt(N)` |
| Excess return | Total portfolio return minus total SPY return over the window |
| HHI | `sum(w_i^2)`, from 1/n (diversified) to 1 (single holding) |
| Effective # of holdings | `1 / HHI` |
| Drawdown | `wealth_t / max(1, running peak of wealth) - 1` |
| Drawdown duration | Calendar days from peak to recovery (or to the last date if not yet recovered) |

## Assumptions

- **Current-holdings backtest.** Today's share counts are applied to the whole window. Purchase dates and
  actual trade history are *not* used to reconstruct past positions; `cost_basis` is only used for the
  unrealized P&L table.
- **Common history.** Prices are aligned across all holdings and SPY, so the window starts when the newest
  holding began trading. Gaps of up to 5 days are forward-filled.
- **Adjusted prices.** Uses Yahoo's split- and dividend-adjusted closes, so returns are total returns.
- **Risk-free rate** defaults to 4.0% annual and is adjustable in the sidebar. Annualization uses 252 days.
- **Sector data** is best-effort from Yahoo; ETFs/funds are labeled "ETF / Fund" and unknowns "Unknown".
- Benchmark is **SPY** (an S&P 500 ETF) as a proxy for the index.

## Limitations and disclaimer

Yahoo Finance data is unofficial, may be delayed or incorrect, and can be rate-limited. Results ignore
taxes, fees, cash flows, FX and survivorship bias, and past performance does not predict future results.
This project is for **educational purposes only and is not investment advice**.

## License

MIT, see [LICENSE](LICENSE).
