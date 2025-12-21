"""History-aware analytics (git deltas and temporal aggregations).

For Hamilton native execution, use ``build_history_timeseries_rows`` to get row
tuples, then materialize with ``materialize_rows``.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.history_timeseries``
"""

from __future__ import annotations

from codeintel.analytics.history.git_history import FileCommitDelta, iter_file_history
from codeintel.analytics.history.history_timeseries import build_history_timeseries_rows

__all__ = [
    "FileCommitDelta",
    "build_history_timeseries_rows",
    "iter_file_history",
]
