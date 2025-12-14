"""Test analytics: coverage edges, behavioral profiles, and graph metrics.

This package consolidates test analytics functionality into a coherent structure:

- ``coverage/``: Coverage edge computation and aggregation
- ``behavioral/``: Behavioral tagging and importance scoring
- ``profiles/``: Test profile building and types
- ``graph_metrics``: Graph metrics over test-function bipartite graphs

For Hamilton native execution, use the pure compute function:
- `compute_test_graph_metrics_pure` returns `TestGraphMetricsResult` without writing

Example
-------
>>> from codeintel.analytics.testing import (
...     build_behavioral_coverage,
...     build_test_profile,
...     compute_test_coverage_edges,
...     compute_test_graph_metrics,
... )
"""

from __future__ import annotations

from codeintel.analytics.testing.compute import (
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.analytics.testing.coverage.edges import (
    TestCoverageOptions,
    compute_test_coverage_edges,
)
from codeintel.analytics.testing.graph_metrics import compute_test_graph_metrics
from codeintel.analytics.testing.profiles.builder import (
    build_behavioral_coverage,
    build_test_profile,
)

__all__ = [
    "TestCoverageOptions",
    "TestGraphMetricsResult",
    "build_behavioral_coverage",
    "build_test_profile",
    "compute_test_coverage_edges",
    "compute_test_graph_metrics",
    "compute_test_graph_metrics_pure",
]
