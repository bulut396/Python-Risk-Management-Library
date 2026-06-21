"""Core risk and performance metrics.

All functions operate on a series of periodic (typically daily) simple returns,
such as the output of :func:`risk_toolkit.utils.prices_to_returns`.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
import pandas as pd

from .drawdown import max_drawdown

ReturnsInput = Union[Sequence[float], pd.Series]


def _as_series(returns: ReturnsInput) -> pd.Series:
    """Coerce supported inputs into a float ``pandas.Series``."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(list(returns), dtype="float64")
    else:
        returns = returns.astype("float64")
    return returns


def annualized_volatility(returns: ReturnsInput, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    periods_per_year : int, optional
        Number of return periods in one year (252 trading days by default).

    Returns
    -------
    float
        The sample standard deviation of ``returns`` scaled by
        ``sqrt(periods_per_year)``.

    Raises
    ------
    ValueError
        If ``returns`` contains fewer than two observations (the sample
        standard deviation is undefined).
    """
    returns = _as_series(returns)
    if len(returns) < 2:
        raise ValueError("At least two returns are required to compute volatility.")

    # ddof=1 -> sample standard deviation.
    period_std = float(returns.std(ddof=1))
    return period_std * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: ReturnsInput,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    risk_free_rate : float, optional
        The **annual** risk-free rate (e.g. ``0.04`` for 4%). It is converted
        to a per-period rate before being subtracted from each return.
    periods_per_year : int, optional
        Number of return periods in one year (252 trading days by default).

    Returns
    -------
    float
        ``mean(excess) / std(excess) * sqrt(periods_per_year)`` where
        ``excess`` is the return in excess of the per-period risk-free rate.

    Raises
    ------
    ValueError
        If fewer than two returns are supplied, or if the standard deviation
        of the excess returns is zero (the ratio would be undefined).
    """
    returns = _as_series(returns)
    if len(returns) < 2:
        raise ValueError("At least two returns are required to compute the Sharpe ratio.")

    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf

    std = float(excess.std(ddof=1))
    if std == 0:
        raise ValueError(
            "Standard deviation of excess returns is zero; Sharpe ratio is undefined."
        )

    return float(excess.mean() / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: ReturnsInput,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio.

    Identical to the Sharpe ratio except that the denominator uses the
    *downside deviation* — the standard deviation computed from the negative
    excess returns only, with positive excess returns contributing zero.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    risk_free_rate : float, optional
        The **annual** risk-free rate (e.g. ``0.04`` for 4%), converted to a
        per-period rate before being subtracted.
    periods_per_year : int, optional
        Number of return periods in one year (252 trading days by default).

    Returns
    -------
    float
        ``mean(excess) / downside_deviation * sqrt(periods_per_year)``.

    Raises
    ------
    ValueError
        If fewer than two returns are supplied, or if the downside deviation
        is zero (e.g. no excess return is ever negative), which would make the
        ratio undefined.
    """
    returns = _as_series(returns)
    if len(returns) < 2:
        raise ValueError("At least two returns are required to compute the Sortino ratio.")

    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf

    # Positive excess returns contribute zero to the downside.
    downside = excess.clip(upper=0.0)
    # Root-mean-square of the (clipped) downside deviations.
    downside_deviation = math.sqrt(float((downside ** 2).mean()))

    if downside_deviation == 0:
        raise ValueError(
            "Downside deviation is zero; Sortino ratio is undefined."
        )

    return float(excess.mean() / downside_deviation) * math.sqrt(periods_per_year)


def beta(asset_returns: ReturnsInput, market_returns: ReturnsInput) -> float:
    """Beta of an asset relative to the market.

    Beta measures the sensitivity of an asset's returns to movements in the
    market and is defined as
    ``covariance(asset, market) / variance(market)``.

    Parameters
    ----------
    asset_returns : sequence of float or pandas.Series
        Periodic simple returns of the asset.
    market_returns : sequence of float or pandas.Series
        Periodic simple returns of the market benchmark, aligned positionally
        with ``asset_returns``.

    Returns
    -------
    float
        The asset's beta.

    Raises
    ------
    ValueError
        If the two series differ in length, if fewer than two observations are
        supplied, or if the market variance is zero (beta is undefined).
    """
    asset = _as_series(asset_returns).reset_index(drop=True)
    market = _as_series(market_returns).reset_index(drop=True)

    if len(asset) != len(market):
        raise ValueError("asset_returns and market_returns must have equal length.")
    if len(asset) < 2:
        raise ValueError("At least two observations are required to compute beta.")

    market_var = float(market.var(ddof=1))
    if market_var == 0:
        raise ValueError("Market variance is zero; beta is undefined.")

    covariance = float(np.cov(asset, market, ddof=1)[0, 1])
    return covariance / market_var


def calmar_ratio(returns: ReturnsInput, periods_per_year: int = 252) -> float:
    """Calmar ratio: annualized return relative to maximum drawdown.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns.
    periods_per_year : int, optional
        Number of return periods in one year (252 trading days by default).

    Returns
    -------
    float
        ``annualized_return / abs(max_drawdown)``, where the annualized
        return is derived from the total compounded return over the sample.

    Raises
    ------
    ValueError
        If fewer than two returns are supplied, or if the maximum drawdown is
        zero (no decline ever occurred, making the ratio undefined).
    """
    returns = _as_series(returns)
    if len(returns) < 2:
        raise ValueError("At least two returns are required to compute the Calmar ratio.")

    n_periods = len(returns)
    total_return = float((1.0 + returns).prod() - 1.0)
    annualized_return = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0

    max_dd = max_drawdown(returns)["max_drawdown"]
    if max_dd == 0:
        raise ValueError(
            "Maximum drawdown is zero; Calmar ratio is undefined."
        )

    return annualized_return / abs(max_dd)
