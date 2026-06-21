"""Tests for risk_toolkit.var."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_toolkit import historical_var, parametric_var


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
