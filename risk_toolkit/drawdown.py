"""Drawdown analysis utilities."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import pandas as pd

ReturnsInput = Union[Sequence[float], pd.Series]


def _as_series(returns: ReturnsInput) -> pd.Series:
    """Coerce supported inputs into a float ``pandas.Series``."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(list(returns), dtype="float64")
    else:
        returns = returns.astype("float64")
    return returns


def max_drawdown(returns: ReturnsInput) -> Dict[str, Any]:
    """Maximum drawdown of a returns series.

    The maximum drawdown is the largest peak-to-trough decline of the
    cumulative wealth curve ``(1 + returns).cumprod()``.

    Parameters
    ----------
    returns : sequence of float or pandas.Series
        Periodic simple returns. If a :class:`pandas.Series` with a meaningful
        index (e.g. dates) is supplied, the returned peak/trough keys use that
        index's labels; otherwise integer positions are used.

    Returns
    -------
    dict
        A dictionary with keys:

        ``"max_drawdown"``
            The most negative drawdown as a float (e.g. ``-0.23`` for -23%).
            ``0.0`` if the curve never declines.
        ``"peak_index"``
            Index label (or integer position) of the peak preceding the worst
            trough.
        ``"trough_index"``
            Index label (or integer position) of the worst point.

    Raises
    ------
    ValueError
        If ``returns`` is empty.
    """
    returns = _as_series(returns)
    if len(returns) == 0:
        raise ValueError("returns must contain at least one observation.")

    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    # Position of the worst (most negative) drawdown.
    trough_pos = int(drawdown.to_numpy().argmin())
    trough_label = drawdown.index[trough_pos]

    # The peak is the point of the running maximum that the trough fell from,
    # i.e. the highest cumulative value at or before the trough.
    peak_pos = int(cumulative.iloc[: trough_pos + 1].to_numpy().argmax())
    peak_label = cumulative.index[peak_pos]

    return {
        "max_drawdown": float(drawdown.iloc[trough_pos]),
        "peak_index": peak_label,
        "trough_index": trough_label,
    }
