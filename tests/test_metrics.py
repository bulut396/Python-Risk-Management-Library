"""Tests for risk_toolkit.metrics and the returns helper."""

import math

import numpy as np
import pandas as pd
import pytest

from risk_toolkit import (
    annualized_volatility,
    beta,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from risk_toolkit.utils import prices_to_returns


# --------------------------------------------------------------------------- #
# prices_to_returns
# --------------------------------------------------------------------------- #
def test_prices_to_returns_known_values():
    returns = prices_to_returns([100, 110, 99])
    # (110-100)/100 = 0.10 ; (99-110)/110 = -0.10
    assert list(returns) == pytest.approx([0.10, -0.10])


def test_prices_to_returns_accepts_series():
    prices = pd.Series([100.0, 105.0, 105.0])
    returns = prices_to_returns(prices)
    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.0)


def test_prices_to_returns_rejects_short_input():
    with pytest.raises(ValueError):
        prices_to_returns([100])


def test_prices_to_returns_rejects_non_positive():
    with pytest.raises(ValueError):
        prices_to_returns([100, 0, 50])
    with pytest.raises(ValueError):
        prices_to_returns([100, -5, 50])


# --------------------------------------------------------------------------- #
# annualized_volatility
# --------------------------------------------------------------------------- #
def test_volatility_is_positive():
    returns = prices_to_returns([100, 102, 101, 105, 103])
    assert annualized_volatility(returns) > 0


def test_volatility_known_value():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01])
    # Sample std (ddof=1) of this series is ~0.011547005
    expected = returns.std(ddof=1) * math.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_volatility_rejects_single_value():
    with pytest.raises(ValueError):
        annualized_volatility([0.01])


# --------------------------------------------------------------------------- #
# sharpe_ratio
# --------------------------------------------------------------------------- #
def test_sharpe_known_value():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    excess = returns - 0.0
    expected = (excess.mean() / excess.std(ddof=1)) * math.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_uses_per_period_risk_free():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    per_period = 0.0504 / 252
    excess = returns - per_period
    expected = (excess.mean() / excess.std(ddof=1)) * math.sqrt(252)
    assert sharpe_ratio(returns, risk_free_rate=0.0504) == pytest.approx(expected)


def test_sharpe_zero_std_raises():
    with pytest.raises(ValueError):
        sharpe_ratio([0.01, 0.01, 0.01])


# --------------------------------------------------------------------------- #
# sortino_ratio
# --------------------------------------------------------------------------- #
def test_sortino_known_value():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    excess = returns - 0.0
    downside = excess.clip(upper=0.0)
    downside_dev = math.sqrt((downside ** 2).mean())
    expected = (excess.mean() / downside_dev) * math.sqrt(252)
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_sortino_at_least_as_large_as_sharpe_for_skewed_upside():
    # With limited downside, the Sortino denominator is smaller than the
    # Sharpe denominator, so Sortino >= Sharpe here.
    returns = pd.Series([0.02, 0.03, -0.005, 0.04])
    assert sortino_ratio(returns) >= sharpe_ratio(returns)


def test_sortino_no_downside_raises():
    with pytest.raises(ValueError):
        sortino_ratio([0.01, 0.02, 0.03])


# --------------------------------------------------------------------------- #
# beta
# --------------------------------------------------------------------------- #
def test_beta_identical_series_is_one():
    market = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    assert beta(market, market) == pytest.approx(1.0)


def test_beta_double_market_is_two():
    market = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    asset = market * 2.0
    assert beta(asset, market) == pytest.approx(2.0)


def test_beta_matches_numpy_definition():
    rng = np.random.default_rng(42)
    market = pd.Series(rng.normal(0, 0.02, 50))
    asset = 0.5 * market + pd.Series(rng.normal(0, 0.01, 50))
    cov = np.cov(asset, market, ddof=1)[0, 1]
    expected = cov / market.var(ddof=1)
    assert beta(asset, market) == pytest.approx(expected)


def test_beta_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        beta([0.01, 0.02, 0.03], [0.01, 0.02])


# --------------------------------------------------------------------------- #
# calmar_ratio
# --------------------------------------------------------------------------- #
def test_calmar_ratio_known_value():
    # Cumulative wealth: 1.10, 0.88, 0.968, 1.0164 -> total return 0.0164,
    # max drawdown -0.20 (from peak 1.10 to trough 0.88).
    # With periods_per_year == n_periods, annualization is a no-op, so the
    # expected Calmar ratio is simply total_return / abs(max_drawdown).
    returns = pd.Series([0.10, -0.20, 0.10, 0.05])
    expected = 0.0164 / 0.20
    assert calmar_ratio(returns, periods_per_year=4) == pytest.approx(expected, rel=1e-3)


def test_calmar_ratio_zero_drawdown_raises():
    returns = pd.Series([0.01, 0.02, 0.03])  # monotonically increasing, no decline
    with pytest.raises(ValueError):
        calmar_ratio(returns)


def test_calmar_ratio_rejects_short_input():
    with pytest.raises(ValueError):
        calmar_ratio([0.01])
