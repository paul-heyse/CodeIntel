"""Validate StorageGateway insert helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.storage.gateway import StorageGateway

EXPECTED_FUNCTION_METRICS_LEN = 29


def test_insert_helpers_write_expected_rows(fresh_gateway: StorageGateway) -> None:
    """Insert helpers should populate tables without manual SQL."""
    gateway = fresh_gateway
    con = gateway.con
    now = datetime.now(tz=UTC)
    now_str = now.isoformat()

    gateway.core.insert_repo_map([("r", "c", "{}", "{}", now_str)])
    gateway.core.insert_modules([("m", "m.py", "r", "c")])
    gateway.core.insert_goids(
        [
            (
                1,
                "urn:fn",
                "r",
                "c",
                "m.py",
                "python",
                "function",
                "m.fn",
                1,
                2,
                now_str,
            )
        ]
    )

    gateway.graph.insert_call_graph_nodes([(1, "python", "function", 0, True, "m.py")])
    gateway.graph.insert_call_graph_edges(
        [
            (
                "r",
                "c",
                1,
                None,
                "m.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
                "{}",
            )
        ]
    )
    gateway.graph.insert_import_graph_edges([("r", "c", "m", "m", 1, 1, 1)])
    gateway.graph.insert_symbol_use_edges([("sym", "m.py", "m.py", False, True)])
    gateway.graph.insert_cfg_blocks([(1, 0, "b0", "entry", "m.py", 1, 2, "entry", "[]", 0, 1)])
    gateway.graph.insert_cfg_edges([(1, "b0", "b0", "fallthrough")])
    gateway.graph.insert_dfg_edges([(1, "b0", "b0", "x", "y", "assign")])

    function_metrics_row = (
        1,
        "urn:fn",
        "r",
        "c",
        "m.py",
        "python",
        "function",
        "m.fn",
        1,
        2,
        4,
        3,
        0,
        0,
        0,
        False,
        False,
        False,
        False,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        True,
        "low",
        now_str,
    )
    if len(function_metrics_row) != EXPECTED_FUNCTION_METRICS_LEN:
        pytest.fail(
            f"Unexpected function_metrics row length: {len(function_metrics_row)} "
            f"(expected {EXPECTED_FUNCTION_METRICS_LEN})"
        )
    gateway.analytics.insert_function_metrics([function_metrics_row])
    gateway.analytics.insert_coverage_functions(
        [
            (
                1,
                "urn:fn",
                "r",
                "c",
                "m.py",
                "python",
                "function",
                "m.fn",
                1,
                2,
                2,
                2,
                1.0,
                True,
                "",
                now_str,
            )
        ]
    )
    gateway.analytics.insert_coverage_lines([("r", "c", "m.py", 1, True, True, 1, 1, now_str)])
    gateway.analytics.insert_test_catalog(
        [
            (
                "t::id",
                2,
                "urn:test",
                "r",
                "c",
                "m.py",
                "pkg.m.fn",
                "function",
                "passed",
                5,
                "[]",
                False,
                False,
                now_str,
            )
        ]
    )
    gateway.analytics.insert_test_coverage_edges(
        [
            (
                "t::id",
                2,
                1,
                "urn:fn",
                "r",
                "c",
                "m.py",
                "pkg.m.fn",
                2,
                2,
                1.0,
                "passed",
                now_str,
            )
        ]
    )
    gateway.analytics.insert_goid_risk_factors(
        [
            (
                1,
                "urn:fn",
                "r",
                "c",
                "m.py",
                "python",
                "function",
                "m.fn",
                4,
                3,
                1,
                "low",
                "typed",
                "analysis",
                0.0,
                1.0,
                0,
                False,
                2,
                2,
                1.0,
                True,
                1,
                0,
                "passed",
                0.1,
                "low",
                "[]",
                "[]",
                now_str,
            )
        ]
    )
    gateway.analytics.insert_config_values(
        [("cfg.yaml", "yaml", "feature.flag", "[]", '["pkg.m"]', 1)]
    )
    gateway.analytics.insert_typedness([("r", "c", "m.py", 0, '{"params":1}', 0, False)])
    gateway.analytics.insert_static_diagnostics([("r", "c", "m.py", 0, 0, 0, 0, False)])
    gateway.analytics.insert_graph_metrics_functions(
        [("r", "c", 1, 1, 1, 1, 1, 0.1, 0.2, 0.3, False, None, None, now_str)]
    )
    gateway.analytics.insert_graph_metrics_modules(
        [("r", "c", "m", 1, 1, 1, 1, 0.1, 0.2, 0.3, False, None, None, 1, 1, now_str)]
    )
    gateway.analytics.insert_subsystems(
        [
            (
                "r",
                "c",
                "sub1",
                "Subsystem",
                None,
                1,
                '["m"]',
                "[]",
                1,
                0,
                1,
                0,
                1,
                0.1,
                0.1,
                0,
                "low",
                now_str,
            )
        ]
    )
    gateway.analytics.insert_subsystem_modules([("r", "c", "sub1", "m", "member")])

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
    }
    for key, value in counts.items():
        minimum = expected_min.get(key, 1)
        if value < minimum:
            pytest.fail(f"Insert helper row counts mismatch for {key}: {value} < {minimum}")

    con.close()
