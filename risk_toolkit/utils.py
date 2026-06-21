"""Utility helpers shared across the :mod:`risk_toolkit` package."""

from __future__ import annotations

from typing import Sequence, Union

import pandas as pd

PriceInput = Union[Sequence[float], pd.Series]


def prices_to_returns(prices: PriceInput) -> pd.Series:
    """Convert a price series into a series of simple daily returns.

    The simple return at time ``t`` is defined as::

        return[t] = (price[t] - price[t-1]) / price[t-1]

    Parameters
    ----------
    prices : sequence of float or pandas.Series
        Closing prices ordered from oldest to newest. A plain list (or any
        sequence) is converted to a :class:`pandas.Series` internally.

    Returns
    -------
    pandas.Series
        Simple daily returns. The result has one fewer element than the input
        because the first price has no prior value to compare against. When a
        :class:`pandas.Series` is passed, the index labels of the input
        (excluding the first) are preserved.

    Raises
    ------
    ValueError
        If fewer than two prices are supplied, or if any price is zero or
        negative (returns are undefined for non-positive prices).
    """
    if not isinstance(prices, pd.Series):
        prices = pd.Series(list(prices), dtype="float64")
    else:
        prices = prices.astype("float64")

    if len(prices) < 2:
        raise ValueError("At least two prices are required to compute returns.")

    if (prices <= 0).any():
        raise ValueError("All prices must be strictly positive.")

    returns = prices.pct_change().iloc[1:]
    return returns
