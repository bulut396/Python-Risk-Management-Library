"""Value at Risk (VaR) estimators.

Both estimators express VaR as a (typically negative) return: the value
represents the loss at the given confidence level over one return period.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

ReturnsInput = Union[Sequence[float], pd.Series]


def _as_series(returns: ReturnsInput) -> pd.Series:
    """Coerce supported inputs into a float ``pandas.Series``."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(list(returns), dtype="float64")
    else:
        returns = returns.astype("float64")
    return returns


def _validate_confidence(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")


def historical_var(returns: ReturnsInput, confidence_level: float = 0.95) -> float:
    """Historical (empirical) Value at Risk.

    Estimates VaR directly from the empirical distribution of returns by taking
    the appropriate lower percentile. At 95% confidence this is the 5th
    percentile of the observed returns.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    confidence_level : float, optional
        Confidence level in the open interval ``(0, 1)``; ``0.95`` by default.

    Returns
    -------
    float
        The VaR as a return, normally negative (a loss). For example, ``-0.03``
        means a 3% loss is the threshold at the given confidence level.

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``confidence_level`` is not in ``(0, 1)``.
    """
    _validate_confidence(confidence_level)
    returns = _as_series(returns)
    if len(returns) == 0:
        raise ValueError("returns must contain at least one observation.")

    percentile = (1.0 - confidence_level) * 100.0
    return float(np.percentile(returns.to_numpy(), percentile))


def parametric_var(returns: ReturnsInput, confidence_level: float = 0.95) -> float:
    """Parametric (Gaussian) Value at Risk.

    Assumes returns are normally distributed and computes VaR from the sample
    mean and standard deviation using the normal-distribution z-score for the
    requested confidence level.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    confidence_level : float, optional
        Confidence level in the open interval ``(0, 1)``; ``0.95`` by default.

    Returns
    -------
    float
        The VaR as a return, normally negative (a loss), computed as
        ``mean + z * std`` where ``z`` is the (negative) lower-tail z-score.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than two observations or ``confidence_level``
        is not in ``(0, 1)``.
    """
    _validate_confidence(confidence_level)
    returns = _as_series(returns)
    if len(returns) < 2:
        raise ValueError(
            "At least two returns are required to estimate parametric VaR."
        )

    mean = float(returns.mean())
    std = float(returns.std(ddof=1))

    # Lower-tail z-score, e.g. ~ -1.645 at 95% confidence.
    z = float(norm.ppf(1.0 - confidence_level))
    return mean + z * std


def historical_cvar(returns: ReturnsInput, confidence_level: float = 0.95) -> float:
    """Historical (empirical) Conditional Value at Risk (Expected Shortfall).

    VaR is a loss threshold; CVaR is the average loss in the tail *beyond*
    that threshold, making it a more conservative measure of tail risk.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    confidence_level : float, optional
        Confidence level in the open interval ``(0, 1)``; ``0.95`` by default.

    Returns
    -------
    float
        The mean of all returns at or below the historical VaR threshold,
        normally negative (a loss).

    Raises
    ------
    ValueError
        If ``returns`` is empty or ``confidence_level`` is not in ``(0, 1)``.
    """
    threshold = historical_var(returns, confidence_level)
    returns = _as_series(returns)
    tail = returns[returns <= threshold]
    return float(tail.mean())


def parametric_cvar(returns: ReturnsInput, confidence_level: float = 0.95) -> float:
    """Parametric (Gaussian) Conditional Value at Risk (Expected Shortfall).

    Assumes returns are normally distributed and computes the closed-form
    expected loss in the tail beyond the parametric VaR threshold.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    confidence_level : float, optional
        Confidence level in the open interval ``(0, 1)``; ``0.95`` by default.

    Returns
    -------
    float
        The CVaR as a return, normally negative (a loss), computed as
        ``mean - std * (phi(z) / (1 - confidence_level))`` where ``z`` is the
        lower-tail z-score and ``phi`` is the standard normal density at ``z``.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than two observations or ``confidence_level``
        is not in ``(0, 1)``.
    """
    _validate_confidence(confidence_level)
    returns = _as_series(returns)
    if len(returns) < 2:
        raise ValueError(
            "At least two returns are required to estimate parametric CVaR."
        )

    mean = float(returns.mean())
    std = float(returns.std(ddof=1))

    z = float(norm.ppf(1.0 - confidence_level))
    phi_z = float(norm.pdf(z))
    return mean - std * (phi_z / (1.0 - confidence_level))
