"""Test analytics: behavioral profiles and graph metrics.

This package consolidates test analytics functionality into a coherent structure:

- ``behavioral/``: Behavioral tagging and importance scoring
- ``profiles/``: Test profile building and types
- ``graph_metrics``: Graph metrics over test-function bipartite graphs

For Hamilton native execution, use the targets under
``codeintel.build.hamilton.native.analytics.metrics_targets``.
"""

from __future__ import annotations

from codeintel.build.analytics.testing.compute import TestGraphMetricsResult

__all__ = [
    "TestGraphMetricsResult",
]
