"""Graph metrics over the test bipartite graph - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.graph_metrics``.
"""

from __future__ import annotations

from codeintel.analytics.testing.graph_metrics import (
    TestMetricsContext,
    compute_test_graph_metrics,
)

__all__ = [
    "TestMetricsContext",
    "compute_test_graph_metrics",
]
