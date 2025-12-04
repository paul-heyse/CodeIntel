"""Build test_coverage_edges - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing.coverage.edges``.
"""

from __future__ import annotations

from codeintel.analytics.testing.coverage.edges import (
    EdgeContext,
    FunctionRow,
    backfill_test_goids_for_catalog,
    build_edges_for_file_for_tests,
    compute_test_coverage_edges,
)

__all__ = [
    "EdgeContext",
    "FunctionRow",
    "backfill_test_goids_for_catalog",
    "build_edges_for_file_for_tests",
    "compute_test_coverage_edges",
]
