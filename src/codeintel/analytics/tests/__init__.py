"""Test analytics helpers - re-exports from testing/.

This module provides backward-compatible imports. New code should import
directly from ``codeintel.analytics.testing``.
"""

from __future__ import annotations

from codeintel.analytics.testing.coverage.edges import compute_test_coverage_edges
from codeintel.analytics.testing.graph_metrics import compute_test_graph_metrics
from codeintel.analytics.testing.profiles.builder import (
    build_behavioral_coverage,
    build_test_profile,
)

__all__ = [
    "build_behavioral_coverage",
    "build_test_profile",
    "compute_test_coverage_edges",
    "compute_test_graph_metrics",
]
