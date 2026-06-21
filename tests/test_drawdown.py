"""Tests for risk_toolkit.drawdown."""

import pandas as pd
import pytest

from risk_toolkit import max_drawdown


def test_max_drawdown_known_value():
    # Cumulative wealth: 1.10, 0.99, 1.089, 1.05633
    # Peak at pos 0 (1.10), trough at pos 1 (0.99): (0.99-1.10)/1.10 = -0.10
    returns = pd.Series([0.10, -0.10, 0.10, -0.03])
    result = max_drawdown(returns)
    assert result["max_drawdown"] == pytest.approx(-0.10)
    assert result["peak_index"] == 0
    assert result["trough_index"] == 1


def test_max_drawdown_no_decline_is_zero():
    returns = pd.Series([0.01, 0.02, 0.03])
    result = max_drawdown(returns)
    assert result["max_drawdown"] == pytest.approx(0.0)


def test_max_drawdown_preserves_datetime_index():
    idx = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    returns = pd.Series([0.05, -0.20, 0.10, 0.02], index=idx)
    result = max_drawdown(returns)
    assert result["peak_index"] == pd.Timestamp("2024-01-01")
    assert result["trough_index"] == pd.Timestamp("2024-01-02")


def test_max_drawdown_empty_raises():
    with pytest.raises(ValueError):
        max_drawdown([])
