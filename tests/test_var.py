"""Tests for risk_toolkit.var."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_toolkit import historical_cvar, historical_var, parametric_cvar, parametric_var


def test_historical_var_known_percentile():
    returns = pd.Series([-0.05, -0.02, 0.0, 0.01, 0.03])
    # 5th percentile (95% confidence) via numpy's default interpolation.
    expected = np.percentile(returns.to_numpy(), 5)
    assert historical_var(returns, 0.95) == pytest.approx(expected)


def test_historical_var_is_a_loss():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.02, 1000))
    assert historical_var(returns, 0.95) < 0


def test_historical_var_rejects_bad_confidence():
    with pytest.raises(ValueError):
        historical_var([0.01, -0.01], 1.5)


def test_historical_var_empty_raises():
    with pytest.raises(ValueError):
        historical_var([], 0.95)


def test_parametric_var_known_value():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
    mean = returns.mean()
    std = returns.std(ddof=1)
    expected = mean + norm.ppf(0.05) * std
    assert parametric_var(returns, 0.95) == pytest.approx(expected)


def test_parametric_var_is_a_loss_for_zero_mean():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0, 0.02, 1000))
    assert parametric_var(returns, 0.95) < 0


def test_parametric_var_rejects_bad_confidence():
    with pytest.raises(ValueError):
        parametric_var([0.01, -0.01], 0.0)


def test_parametric_var_single_value_raises():
    with pytest.raises(ValueError):
        parametric_var([0.01], 0.95)


def test_historical_cvar_known_value():
    # Sorted ascending: -0.05, -0.03, -0.01, 0.02, 0.04.
    # 20th percentile (80% confidence) interpolates to -0.034 between the two
    # lowest values, so only -0.05 falls at-or-below the threshold -> CVaR = -0.05.
    returns = pd.Series([-0.05, -0.03, -0.01, 0.02, 0.04])
    assert historical_cvar(returns, confidence_level=0.8) == pytest.approx(-0.05)


def test_historical_cvar_is_at_least_as_extreme_as_var():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.02, 200))
    var = historical_var(returns, 0.95)
    cvar = historical_cvar(returns, 0.95)
    assert cvar <= var


def test_historical_cvar_rejects_bad_confidence():
    with pytest.raises(ValueError):
        historical_cvar([0.01, -0.01], 1.5)


def test_historical_cvar_empty_raises():
    with pytest.raises(ValueError):
        historical_cvar([], 0.95)


def test_parametric_cvar_matches_manual_calculation():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
    mean = returns.mean()
    std = returns.std(ddof=1)
    z = norm.ppf(0.05)
    phi_z = norm.pdf(z)
    expected = mean - std * (phi_z / 0.05)
    assert parametric_cvar(returns, 0.95) == pytest.approx(expected)


def test_parametric_cvar_is_at_least_as_extreme_as_var():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0, 0.02, 500))
    var = parametric_var(returns, 0.95)
    cvar = parametric_cvar(returns, 0.95)
    assert cvar <= var


def test_parametric_cvar_rejects_bad_confidence():
    with pytest.raises(ValueError):
        parametric_cvar([0.01, -0.01], 0.0)


def test_parametric_cvar_single_value_raises():
    with pytest.raises(ValueError):
        parametric_cvar([0.01], 0.95)
