"""Tests for gateway accessor classes (read-only surface).

The gateway accessors provide a typed read interface over DuckDB relations.
All writes are routed through the Warehouse API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.gateway.accessors import (
    AnalyticsTables,
    CoreTables,
    DocsViews,
    GraphTables,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_true,
    require_row,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_accessors_are_frozen_dataclasses(fresh_gateway: StorageGateway) -> None:
    """Verify accessor classes are immutable."""
    replacement = object()
    assert_frozen(CoreTables(fresh_gateway), "gateway", replacement)
    assert_frozen(GraphTables(fresh_gateway), "gateway", replacement)
    assert_frozen(AnalyticsTables(fresh_gateway), "gateway", replacement)
    assert_frozen(DocsViews(fresh_gateway), "gateway", replacement)


def test_core_tables_relations_are_queryable(fresh_gateway: StorageGateway) -> None:
    """Verify core accessor returns relations that can be queried."""
    core = CoreTables(fresh_gateway)
    expect_true(core.con is fresh_gateway.con, message="core uses gateway connection")

    relations = (
        core.goids(),
        core.modules(),
        core.repo_map(),
        core.file_state(),
        core.scip_occurrences(),
    )
    for relation in relations:
        require_row(relation.count("*").fetchone(), message="count row present")


def test_graph_tables_relations_are_queryable(fresh_gateway: StorageGateway) -> None:
    """Verify graph accessor returns relations that can be queried."""
    graph = GraphTables(fresh_gateway)
    for relation in (
        graph.call_graph_nodes(),
        graph.call_graph_edges(),
        graph.import_graph_edges(),
        graph.symbol_use_edges(),
        graph.cfg_blocks(),
        graph.cfg_edges(),
        graph.dfg_edges(),
    ):
        require_row(relation.count("*").fetchone(), message="count row present")


def test_analytics_tables_relations_are_queryable(fresh_gateway: StorageGateway) -> None:
    """Verify analytics accessor returns relations that can be queried."""
    analytics = AnalyticsTables(fresh_gateway)
    for relation in (
        analytics.function_metrics(),
        analytics.function_types(),
        analytics.function_validation(),
        analytics.function_profile(),
        analytics.coverage_functions(),
        analytics.coverage_lines(),
        analytics.test_catalog(),
        analytics.test_coverage_edges(),
        analytics.goid_risk_factors(),
        analytics.config_values(),
        analytics.typedness(),
        analytics.static_diagnostics(),
        analytics.subsystems(),
        analytics.subsystem_modules(),
    ):
        require_row(relation.count("*").fetchone(), message="count row present")


def test_docs_views_relations_are_queryable(fresh_gateway: StorageGateway) -> None:
    """Verify docs view accessor returns relations that can be queried."""
    docs = DocsViews(fresh_gateway)
    relations = (docs.function_summary(), docs.call_graph_enriched(), docs.function_profile())
    for relation in relations:
        require_row(relation.count("*").fetchone(), message="count row present")
