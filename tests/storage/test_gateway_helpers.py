"""Validate StorageGateway insert helpers."""

from __future__ import annotations

import pytest

from tests._helpers.builders import (
    CoverageLineRow,
    insert_rows,
)
from tests._helpers.context import TestContext
from tests._helpers.seeds import (
    CORE_PACK,
    COVERAGE_LINES_PACK,
    COVERAGE_PACK,
    GRAPH_PACK,
    METRICS_PACK,
    SUBSYSTEM_ANALYTICS_PACK,
)


def test_insert_helpers_write_expected_rows(test_ctx: TestContext) -> None:
    """Insert helpers should populate tables without manual SQL."""
    test_ctx.require(
        CORE_PACK,
        GRAPH_PACK,
        METRICS_PACK,
        COVERAGE_PACK,
        COVERAGE_LINES_PACK,
        SUBSYSTEM_ANALYTICS_PACK,
    )
    gateway = test_ctx.gateway
    con = gateway.con
    if con.execute("SELECT COUNT(*) FROM analytics.coverage_lines").fetchone()[0] == 0:
        insert_rows(
            gateway,
            [
                CoverageLineRow(
                    repo=test_ctx.repo,
                    commit=test_ctx.commit,
                    rel_path="core/mod_a.py",
                    line=1,
                    is_executable=True,
                    is_covered=True,
                    hits=1,
                    context_count=1,
                    created_at=test_ctx.snapshot.created_at,
                )
            ],
        )

    def _count(query: str) -> int:
        row = con.execute(query).fetchone()
        if row is None or row[0] is None:
            pytest.fail(f"Missing row count for query: {query}")
        return int(row[0])

    counts = {
        "core.repo_map": _count("SELECT COUNT(*) FROM core.repo_map"),
        "core.modules": _count("SELECT COUNT(*) FROM core.modules"),
        "core.goids": _count("SELECT COUNT(*) FROM core.goids"),
        "graph.call_graph_nodes": _count("SELECT COUNT(*) FROM graph.call_graph_nodes"),
        "graph.call_graph_edges": _count("SELECT COUNT(*) FROM graph.call_graph_edges"),
        "graph.import_graph_edges": _count("SELECT COUNT(*) FROM graph.import_graph_edges"),
        "graph.symbol_use_edges": _count("SELECT COUNT(*) FROM graph.symbol_use_edges"),
        "graph.cfg_blocks": _count("SELECT COUNT(*) FROM graph.cfg_blocks"),
        "graph.cfg_edges": _count("SELECT COUNT(*) FROM graph.cfg_edges"),
        "graph.dfg_edges": _count("SELECT COUNT(*) FROM graph.dfg_edges"),
        "analytics.function_metrics": _count("SELECT COUNT(*) FROM analytics.function_metrics"),
        "analytics.coverage_functions": _count("SELECT COUNT(*) FROM analytics.coverage_functions"),
        "analytics.coverage_lines": _count("SELECT COUNT(*) FROM analytics.coverage_lines"),
        "analytics.test_catalog": _count("SELECT COUNT(*) FROM analytics.test_catalog"),
        "analytics.test_coverage_edges": _count(
            "SELECT COUNT(*) FROM analytics.test_coverage_edges"
        ),
        "analytics.goid_risk_factors": _count("SELECT COUNT(*) FROM analytics.goid_risk_factors"),
        "analytics.config_values": _count("SELECT COUNT(*) FROM analytics.config_values"),
        "analytics.typedness": _count("SELECT COUNT(*) FROM analytics.typedness"),
        "analytics.static_diagnostics": _count("SELECT COUNT(*) FROM analytics.static_diagnostics"),
        "analytics.graph_metrics_functions": _count(
            "SELECT COUNT(*) FROM analytics.graph_metrics_functions"
        ),
        "analytics.graph_metrics_modules": _count(
            "SELECT COUNT(*) FROM analytics.graph_metrics_modules"
        ),
        "analytics.subsystems": _count("SELECT COUNT(*) FROM analytics.subsystems"),
        "analytics.subsystem_modules": _count("SELECT COUNT(*) FROM analytics.subsystem_modules"),
    }
    expected_min = {
        "analytics.goid_risk_factors": 0,
        "analytics.graph_metrics_functions": 0,
        "analytics.graph_metrics_modules": 0,
    }
    for key, value in counts.items():
        minimum = expected_min.get(key, 1)
        if value < minimum:
            pytest.fail(f"Insert helper row counts mismatch for {key}: {value} < {minimum}")
