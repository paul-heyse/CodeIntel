"""History-aware analytics (git deltas and temporal aggregations)."""

from __future__ import annotations

from codeintel.analytics.functions.function_history import compute_function_history
from codeintel.analytics.history.git_history import FileCommitDelta, iter_file_history
from codeintel.analytics.history.history_timeseries import (
    compute_history_timeseries,
    compute_history_timeseries_gateways,
)

__all__ = [
    "FileCommitDelta",
    "compute_function_history",
    "compute_history_timeseries",
    "compute_history_timeseries_gateways",
    "iter_file_history",
]
