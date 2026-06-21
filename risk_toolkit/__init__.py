"""risk_toolkit — financial risk and performance metrics for Python.

Exposes the library's public functions at the package level::

    from risk_toolkit import sharpe_ratio, max_drawdown, historical_var
"""

from __future__ import annotations

from .drawdown import max_drawdown
from .metrics import (
    annualized_volatility,
    beta,
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from .utils import prices_to_returns
from .var import historical_cvar, historical_var, parametric_cvar, parametric_var

__version__ = "0.2.0"

__all__ = [
    "prices_to_returns",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "beta",
    "calmar_ratio",
    "max_drawdown",
    "historical_var",
    "parametric_var",
    "historical_cvar",
    "parametric_cvar",
    "__version__",
]
