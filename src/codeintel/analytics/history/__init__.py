"""History-aware analytics (git deltas and temporal aggregations)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.history.git_history import FileCommitDelta, iter_file_history
from codeintel.analytics.history.history_timeseries import (
    HistoryTimeseriesOptions,
    compute_history_timeseries,
    compute_history_timeseries_gateways,
)
from codeintel.analytics.utilities.lazy_module import make_lazy_getattr

if TYPE_CHECKING:
    from codeintel.analytics.functions.function_history import compute_function_history

__all__ = [
    "FileCommitDelta",
    "HistoryTimeseriesOptions",
    "compute_function_history",
    "compute_history_timeseries",
    "compute_history_timeseries_gateways",
    "iter_file_history",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "compute_function_history": (
        "codeintel.analytics.functions.function_history",
        "compute_function_history",
    ),
}

__getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__)
