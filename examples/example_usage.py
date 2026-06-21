"""Example usage of risk_toolkit.

Run this script directly to see every metric computed on a realistic, simulated
price series. The data is generated with a fixed random seed, so the output is
reproducible and the script runs fully offline (no network or API keys).

    python examples/example_usage.py
"""

from typing import List, Tuple

import numpy as np

from risk_toolkit import (
    annualized_volatility,
    beta,
    calmar_ratio,
    historical_cvar,
    historical_var,
    max_drawdown,
    parametric_cvar,
    parametric_var,
    prices_to_returns,
    sharpe_ratio,
    sortino_ratio,
)

# Fixed seed -> deterministic, reproducible output.
SEED = 2
# ~3 months of trading days.
N_DAYS = 63


def _to_prices(returns: np.ndarray, start: float) -> List[float]:
    """Compound a series of daily returns into a price path."""
    prices = [start]
    for r in returns:
        prices.append(round(prices[-1] * (1.0 + r), 2))
    return prices


def generate_sample_prices() -> Tuple[List[float], List[float]]:
    """Generate a realistic asset price path and a market benchmark.

    The market follows a mild upward drift with daily noise. The asset is driven
    by the market (beta ~1.1) plus its own idiosyncratic noise, and includes a
    deliberate multi-day correction (~10% drawdown) so the drawdown metric has
    something meaningful to report. This mimics how a real stock trades far
    better than a smooth straight line would.

    Returns
    -------
    (asset_prices, market_prices) : tuple of list of float
        Two equal-length daily closing-price series, oldest to newest.
    """
    rng = np.random.default_rng(SEED)

    # Market: small positive daily drift (~0.07%) with ~0.8% daily volatility.
    market_returns = rng.normal(0.0007, 0.008, N_DAYS - 1)

    # Asset: market exposure (beta) + idiosyncratic alpha/noise.
    beta_true = 1.1
    idiosyncratic = rng.normal(0.0006, 0.007, N_DAYS - 1)
    asset_returns = beta_true * market_returns + idiosyncratic

    # Inject a sustained ~6-7% mid-series correction (a realistic pullback).
    asset_returns[30:37] += -0.009

    asset_prices = _to_prices(asset_returns, start=100.0)
    market_prices = _to_prices(market_returns, start=250.0)
    return asset_prices, market_prices


def main() -> None:
    asset_prices, market_prices = generate_sample_prices()
    asset_returns = prices_to_returns(asset_prices)
    market_returns = prices_to_returns(market_prices)

    annual_rf = 0.04  # 4% annual risk-free rate

    print("=" * 52)
    print(" risk_toolkit - example metrics (simulated stock)")
    print("=" * 52)
    print(f"Trading days (prices)         : {len(asset_prices)}")
    print(f"Return observations           : {len(asset_returns)}")
    print(f"Price range                   : {min(asset_prices):.2f} -> {max(asset_prices):.2f}")
    print()

    print(f"Annualized volatility         : {annualized_volatility(asset_returns):.4f}")
    print(f"Sharpe ratio (rf=4%)          : {sharpe_ratio(asset_returns, annual_rf):.4f}")
    print(f"Sortino ratio (rf=4%)         : {sortino_ratio(asset_returns, annual_rf):.4f}")
    print(f"Beta vs. market               : {beta(asset_returns, market_returns):.4f}")
    print()

    dd = max_drawdown(asset_returns)
    print("Maximum drawdown:")
    print(f"  max_drawdown                : {dd['max_drawdown']:.4%}")
    print(f"  peak_index                  : {dd['peak_index']}")
    print(f"  trough_index                : {dd['trough_index']}")
    print()

    print(f"Historical VaR (95%)          : {historical_var(asset_returns, 0.95):.4%}")
    print(f"Parametric VaR (95%)          : {parametric_var(asset_returns, 0.95):.4%}")
    print(f"Historical CVaR (95%)         : {historical_cvar(asset_returns, 0.95):.4%}")
    print(f"Parametric CVaR (95%)         : {parametric_cvar(asset_returns, 0.95):.4%}")
    print(f"Calmar ratio                  : {calmar_ratio(asset_returns):.4f}")
    print("=" * 52)

    # ----------------------------------------------------------------------- #
    # Using real data instead of the simulated prices above.
    #
    # Install yfinance (`pip install yfinance`) and uncomment the block below
    # to pull real closing prices. No API key is required.
    #
    # import yfinance as yf
    #
    # data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    # prices = data["Close"].dropna()
    # asset_returns = prices_to_returns(prices)
    # print(sharpe_ratio(asset_returns, annual_rf))
    # ----------------------------------------------------------------------- #


if __name__ == "__main__":
    main()
