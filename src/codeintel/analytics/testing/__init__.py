"""Test analytics: coverage edges, behavioral profiles, and graph metrics.

This package consolidates test analytics functionality into a coherent structure:

- ``coverage/``: Coverage edge computation and aggregation
- ``behavioral/``: Behavioral tagging and importance scoring
- ``profiles/``: Test profile building and types
- ``graph_metrics``: Graph metrics over test-function bipartite graphs

For Hamilton native execution, use the pure compute function:
- ``compute_test_graph_metrics_pure`` returns ``TestGraphMetricsResult`` without writing

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.test_graph_metrics``
"""

from __future__ import annotations

from codeintel.analytics.testing.compute import (
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.analytics.testing.coverage.edges import (
    TestCoverageOptions,
    build_test_coverage_edges_rows,
)

__all__ = [
    "TestCoverageOptions",
    "TestGraphMetricsResult",
    "build_test_coverage_edges_rows",
    "compute_test_graph_metrics_pure",
]
