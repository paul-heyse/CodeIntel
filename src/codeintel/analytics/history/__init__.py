"""History-aware analytics (git deltas and temporal aggregations)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from codeintel.analytics.history.git_history import FileCommitDelta, iter_file_history
from codeintel.analytics.history.history_timeseries import (
    compute_history_timeseries,
    compute_history_timeseries_gateways,
)

if TYPE_CHECKING:
    from codeintel.analytics.functions.function_history import compute_function_history

__all__ = [
    "FileCommitDelta",
    "compute_function_history",
    "compute_history_timeseries",
    "compute_history_timeseries_gateways",
    "iter_file_history",
]


def __getattr__(name: str) -> object:
    """Lazily resolve history helpers to avoid import cycles during module init.

    Returns
    -------
    object
        Requested attribute when exposed by this module.

    Raises
    ------
    AttributeError
        If the requested attribute is not defined.
    """
    if name == "compute_function_history":
        module = importlib.import_module("codeintel.analytics.functions.function_history")
        return module.compute_function_history
    message = f"module {__name__} has no attribute {name}"
    raise AttributeError(message)
