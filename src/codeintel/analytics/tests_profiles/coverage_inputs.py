"""Coverage aggregation - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.coverage.inputs``.
"""

from __future__ import annotations

from codeintel.analytics.testing.coverage.inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
    load_test_graph_metrics,
    load_test_records,
)
from codeintel.analytics.testing.profiles.types import (
    FunctionCoverageEntryProtocol,
    SubsystemCoverageEntryProtocol,
    TestGraphMetricsProtocol,
    TestRecord,
)

__all__ = [
    "FunctionCoverageEntry",
    "FunctionCoverageEntryProtocol",
    "SubsystemCoverageEntry",
    "SubsystemCoverageEntryProtocol",
    "TestGraphMetrics",
    "TestGraphMetricsProtocol",
    "TestRecord",
    "aggregate_test_coverage_by_function",
    "aggregate_test_coverage_by_subsystem",
    "load_test_graph_metrics",
    "load_test_records",
]
