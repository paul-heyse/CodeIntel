"""History-aware analytics (git deltas and temporal aggregations).

Use the native Hamilton targets under
``codeintel.build.hamilton.native.analytics.metrics_targets`` for execution.
"""

from __future__ import annotations

from codeintel.analytics.history.git_history import FileCommitDelta

__all__ = [
    "FileCommitDelta",
]
