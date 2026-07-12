# risk-toolkit

[![CI](https://github.com/bulut396/Python-Risk-Management-Library/actions/workflows/ci.yml/badge.svg)](https://github.com/bulut396/Python-Risk-Management-Library/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A lightweight Python library for calculating financial risk and performance metrics for a portfolio or a single asset.

## Features

- **Returns helper** — convert a price series into simple daily returns
- **Annualized volatility** — standard deviation scaled to a yearly figure
- **Sharpe ratio** — risk-adjusted return vs. a risk-free rate
- **Sortino ratio** — risk-adjusted return penalizing only downside volatility
- **Beta** — sensitivity of an asset to a market benchmark
- **Maximum drawdown** — largest peak-to-trough decline, with peak/trough locations
- **Value at Risk** — both historical (empirical) and parametric (Gaussian) estimators

## Installation

```bash
pip install git+https://github.com/bulut396/Python-Risk-Management-Library
```

Or clone the repository and install locally:

```bash
pip install .
```

## Quick Start

```python
from risk_toolkit import (
    prices_to_returns,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    historical_var,
)

prices = [100, 101.5, 100.8, 103.2, 102.5, 105.0, 104.1, 106.7]
returns = prices_to_returns(prices)

print("Volatility :", annualized_volatility(returns))
print("Sharpe     :", sharpe_ratio(returns, risk_free_rate=0.04))
print("Max DD     :", max_drawdown(returns)["max_drawdown"])
print("VaR (95%)  :", historical_var(returns, confidence_level=0.95))
```

## Available Metrics

| Metric                | Function                                  | What it measures                                              |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Daily returns         | `prices_to_returns(prices)`               | Simple period-over-period returns from a price series         |
| Annualized volatility | `annualized_volatility(returns)`          | Yearly standard deviation of returns                          |
| Sharpe ratio          | `sharpe_ratio(returns, risk_free_rate)`   | Excess return per unit of total volatility                    |
| Sortino ratio         | `sortino_ratio(returns, risk_free_rate)`  | Excess return per unit of downside volatility                 |
| Beta                  | `beta(asset_returns, market_returns)`     | Sensitivity of an asset to market movements                   |
| Maximum drawdown      | `max_drawdown(returns)`                    | Largest peak-to-trough loss, plus peak and trough locations   |
| Historical VaR        | `historical_var(returns, confidence_level)` | Empirical loss threshold at a confidence level              |
| Parametric VaR        | `parametric_var(returns, confidence_level)` | Gaussian loss threshold at a confidence level               |
| Historical CVaR        | `historical_cvar(returns, confidence_level)` | Empirical average loss beyond the VaR threshold (Expected Shortfall) |
| Parametric CVaR        | `parametric_cvar(returns, confidence_level)` | Gaussian average loss beyond the VaR threshold (Expected Shortfall)  |
| Calmar ratio           | `calmar_ratio(returns)`                    | Annualized return relative to maximum drawdown               |

## Running Tests

```bash
pip install pytest
pytest
```

## Version History

- **0.2.0** — Added Conditional VaR (historical & parametric) and Calmar Ratio.
- **0.1.0** — Initial release: volatility, Sharpe, Sortino, beta, max drawdown, VaR.

## License

Released under the [MIT License](LICENSE).
