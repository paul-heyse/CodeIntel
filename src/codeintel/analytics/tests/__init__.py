"""
Tests analytics: coverage edges, behavioral profiles, and graph metrics.

This package consolidates the former monolithic test analytics modules.
"""

from __future__ import annotations

from codeintel.analytics.tests.coverage_edges import compute_test_coverage_edges
from codeintel.analytics.tests.graph_metrics import compute_test_graph_metrics
from codeintel.analytics.tests.profiles import build_behavioral_coverage, build_test_profile

__all__ = [
    "build_behavioral_coverage",
    "build_test_profile",
    "compute_test_coverage_edges",
    "compute_test_graph_metrics",
]
