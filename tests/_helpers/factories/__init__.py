"""Factory functions for creating test data structures.

This module provides factory functions for creating test data including:
- Blank profile rows for analytics contract tests
- Pre-built NetworkX graphs for graph algorithm testing
"""

from __future__ import annotations

from tests._helpers.factories.graph_factories import (
    GOLDEN_EXPECTED_COMMUNITIES,
    GOLDEN_EXPECTED_SCC,
    GOLDEN_MIN_EDGES,
    GOLDEN_MIN_NODES,
    build_chain_graph,
    build_cycle_graph,
    build_dag_with_bottleneck,
    build_dense_cluster,
    build_layered_call_graph,
    build_layered_import_graph,
    build_simple_call_graph,
    build_simple_import_graph,
    build_star_graph,
    build_two_communities_graph,
)
from tests._helpers.factories.row_factories import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_function_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
)

__all__ = [
    "GOLDEN_EXPECTED_COMMUNITIES",
    "GOLDEN_EXPECTED_SCC",
    "GOLDEN_MIN_EDGES",
    "GOLDEN_MIN_NODES",
    "blank_behavioral_coverage_row",
    "blank_file_profile_row",
    "blank_function_profile_row",
    "blank_module_profile_row",
    "blank_test_profile_row",
    "build_chain_graph",
    "build_cycle_graph",
    "build_dag_with_bottleneck",
    "build_dense_cluster",
    "build_layered_call_graph",
    "build_layered_import_graph",
    "build_simple_call_graph",
    "build_simple_import_graph",
    "build_star_graph",
    "build_two_communities_graph",
]
