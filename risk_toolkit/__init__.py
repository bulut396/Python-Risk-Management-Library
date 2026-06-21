"""risk_toolkit — financial risk and performance metrics for Python.

Exposes the library's public functions at the package level::

    from risk_toolkit import sharpe_ratio, max_drawdown, historical_var
"""

from __future__ import annotations

from .drawdown import max_drawdown
from .metrics import (
    annualized_volatility,
    beta,
    sharpe_ratio,
    sortino_ratio,
)
from .utils import prices_to_returns
from .var import historical_var, parametric_var

__version__ = "0.1.0"

__all__ = [
    "prices_to_returns",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "beta",
    "max_drawdown",
    "historical_var",
    "parametric_var",
    "__version__",
]
