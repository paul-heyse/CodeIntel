"""Coverage edge computation and aggregation.

This subpackage provides:

- ``edges``: Test coverage edge computation
- ``inputs``: Coverage aggregation helpers
"""

from __future__ import annotations

from codeintel.analytics.testing.coverage.edges import build_test_coverage_edges_rows
from codeintel.analytics.testing.coverage.inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
    load_test_graph_metrics,
    load_test_records,
)

__all__ = [
    "FunctionCoverageEntry",
    "SubsystemCoverageEntry",
    "TestGraphMetrics",
    "aggregate_test_coverage_by_function",
    "aggregate_test_coverage_by_subsystem",
    "build_test_coverage_edges_rows",
    "load_test_graph_metrics",
    "load_test_records",
]
