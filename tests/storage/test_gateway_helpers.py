"""Validate StorageGateway insert helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.config.datasets import (
    GraphMetricsFunctionsRow,
    GraphMetricsModulesRow,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.rows.analytics import (
    AnalyticsTestCatalogRow,
    AnalyticsTestCoverageEdgesRow,
)
from tests._helpers.builders import insert_symbol_use_edges

EXPECTED_FUNCTION_METRICS_LEN = 29


def test_insert_helpers_write_expected_rows(fresh_gateway: StorageGateway) -> None:
    """Insert helpers should populate tables without manual SQL."""
    gateway = fresh_gateway
    con = gateway.con
    now = datetime.now(tz=UTC)
    now_str = now.isoformat()

    gateway.core.insert_repo_map(
        [
            {
                "repo": "r",
                "commit": "c",
                "modules": "{}",
                "overlays": "{}",
                "generated_at": now_str,
            }
        ]
    )
    gateway.core.insert_modules(
        [
            {
                "module": "m",
                "path": "m.py",
                "repo": "r",
                "commit": "c",
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            }
        ]
    )
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

    gateway.graph.insert_call_graph_nodes(
        [
            {
                "goid_h128": 1.0,
                "language": "python",
                "kind": "function",
                "arity": 0,
                "is_public": True,
                "rel_path": "m.py",
            }
        ]
    )
    gateway.graph.insert_call_graph_edges(
        [
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": 1.0,
                "callee_goid_h128": None,
                "callsite_path": "m.py",
                "callsite_line": 1,
                "callsite_col": 0,
                "language": "python",
                "kind": "direct",
                "resolved_via": "local_name",
                "confidence": 1.0,
                "evidence_json": "{}",
            }
        ]
    )
    gateway.graph.insert_import_graph_edges(
        [
            {
                "repo": "r",
                "commit": "c",
                "src_module": "m",
                "dst_module": "m",
                "src_fan_out": 1,
                "dst_fan_in": 1,
                "cycle_group": 1,
                "module_layer": None,
            }
        ]
    )
    insert_symbol_use_edges(
        gateway,
        [
            {
                "symbol": "sym",
                "def_path": "m.py",
                "use_path": "m.py",
                "same_file": False,
                "same_module": True,
                "def_goid_h128": None,
                "use_goid_h128": None,
            }
        ],
    )
    gateway.graph.insert_cfg_blocks([(1, 0, "b0", "entry", "m.py", 1, 2, "entry", "[]", 0, 1)])
    gateway.graph.insert_cfg_edges([(1, "b0", "b0", "fallthrough")])
    gateway.graph.insert_dfg_edges([(1, "b0", "b0", "x", "y", "assign", False, "read")])

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
    gateway.analytics.insert_coverage_lines(
        [
            {
                "repo": "r",
                "commit": "c",
                "rel_path": "m.py",
                "line": 1,
                "is_executable": True,
                "is_covered": True,
                "hits": 1,
                "context_count": 1,
                "created_at": now_str,
            }
        ]
    )
    catalog_row: AnalyticsTestCatalogRow = {
        "test_id": "t::id",
        "test_goid_h128": 2,
        "urn": "urn:test",
        "repo": "r",
        "commit": "c",
        "rel_path": "m.py",
        "qualname": "pkg.m.fn",
        "kind": "function",
        "status": "passed",
        "duration_ms": 5,
        "markers": "[]",
        "parametrized": False,
        "flaky": False,
        "created_at": now_str,
    }
    gateway.analytics.insert_test_catalog([catalog_row])

    coverage_edge_row: AnalyticsTestCoverageEdgesRow = {
        "test_id": "t::id",
        "test_goid_h128": 2,
        "function_goid_h128": 1,
        "urn": "urn:fn",
        "repo": "r",
        "commit": "c",
        "rel_path": "m.py",
        "qualname": "pkg.m.fn",
        "covered_lines": 2,
        "executable_lines": 2,
        "coverage_ratio": 1.0,
        "last_status": "passed",
        "created_at": now_str,
    }
    gateway.analytics.insert_test_coverage_edges([coverage_edge_row])
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
        [("r", "c", "cfg.yaml", "yaml", "feature.flag", "[]", '["pkg.m"]', 1)]
    )
    gateway.analytics.insert_typedness(
        [
            {
                "repo": "r",
                "commit": "c",
                "path": "m.py",
                "type_error_count": 0,
                "annotation_ratio": '{"params":1}',
                "untyped_defs": 0,
                "overlay_needed": False,
            }
        ]
    )
    gateway.analytics.insert_static_diagnostics([("r", "c", "m.py", 0, 0, 0, 0, False)])
    function_contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_functions")
    module_contract = get_analytics_dataset_contract(gateway, "analytics.graph_metrics_modules")
    insert_analytics_rows(
        gateway,
        function_contract,
        [
            GraphMetricsFunctionsRow(
                repo="r",
                commit="c",
                function_goid_h128=1,
                call_fan_in=1,
                call_fan_out=1,
                call_in_degree=1,
                call_out_degree=1,
                call_pagerank=0.1,
                call_betweenness=0.2,
                call_closeness=0.3,
                call_cycle_member=False,
                call_cycle_id=None,
                call_layer=None,
                created_at=now,
            )
        ],
    )
    insert_analytics_rows(
        gateway,
        module_contract,
        [
            GraphMetricsModulesRow(
                repo="r",
                commit="c",
                module="m",
                import_fan_in=1,
                import_fan_out=1,
                import_in_degree=1,
                import_out_degree=1,
                import_pagerank=0.1,
                import_betweenness=0.2,
                import_closeness=0.3,
                import_cycle_member=False,
                import_cycle_id=None,
                import_layer=None,
                symbol_fan_in=1,
                symbol_fan_out=1,
                created_at=now,
            )
        ],
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
