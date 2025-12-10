"""Validate StorageGateway insert helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.config.datasets import GraphMetricsFunctionsRow, GraphMetricsModulesRow
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.rows.analytics import (
    AnalyticsTestCatalogRow,
    AnalyticsTestCoverageEdgesRow,
)
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    CoverageFunctionRow,
    CoverageLineRow,
    FunctionMetricsRow,
    GoidRow,
    GraphMetricsModulesExtRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    SubsystemModuleRow,
    SubsystemRow,
    TypednessRow,
    insert_symbol_use_edges,
)
from tests._helpers.builders.row_protocol import insert_rows
from tests._helpers.context import TestContext

EXPECTED_FUNCTION_METRICS_LEN = 29


def test_insert_helpers_write_expected_rows(test_ctx: TestContext) -> None:
    """Insert helpers should populate tables without manual SQL."""
    gateway = test_ctx.gateway
    con = gateway.con
    now = datetime.now(tz=UTC)
    now_str = now.isoformat()

    insert_rows(
        gateway,
        [
            RepoMapRow(repo="r", commit="c", modules={}, overlays={}, generated_at=now),
            ModuleRow(module="m", path="m.py", repo="r", commit="c"),
            GoidRow(
                goid_h128=1,
                urn="urn:fn",
                repo="r",
                commit="c",
                rel_path="m.py",
                kind="function",
                qualname="m.fn",
                start_line=1,
                end_line=2,
                created_at=now,
            ),
        ],
    )

    insert_rows(
        gateway,
        [
            CallGraphNodeRow(
                goid_h128=1,
                language="python",
                kind="function",
                arity=0,
                is_public=True,
                rel_path="m.py",
            ),
            CallGraphEdgeRow(
                repo="r",
                commit="c",
                caller_goid_h128=1,
                callee_goid_h128=None,
                callsite_path="m.py",
                callsite_line=1,
                callsite_col=0,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=1.0,
                evidence={},
            ),
            ImportGraphEdgeRow(
                repo="r",
                commit="c",
                src_module="m",
                dst_module="m",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
                module_layer=None,
            ),
        ],
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

    metrics_rows = [
        FunctionMetricsRow(
            function_goid_h128=1,
            urn="urn:fn",
            repo="r",
            commit="c",
            rel_path="m.py",
            language="python",
            kind="function",
            qualname="m.fn",
            start_line=1,
            end_line=2,
            loc=4,
            logical_loc=3,
            param_count=0,
            positional_params=0,
            keyword_only_params=0,
            has_varargs=False,
            has_varkw=False,
            is_async=False,
            is_generator=False,
            return_count=1,
            yield_count=0,
            raise_count=0,
            cyclomatic_complexity=1,
            max_nesting_depth=1,
            stmt_count=1,
            decorator_count=0,
            has_docstring=True,
            complexity_bucket="low",
            created_at=now,
        )
    ]
    if len(metrics_rows[0].to_tuple()) != EXPECTED_FUNCTION_METRICS_LEN:
        pytest.fail(
            f"Unexpected function_metrics row length: {len(metrics_rows[0].to_tuple())} "
            f"(expected {EXPECTED_FUNCTION_METRICS_LEN})"
        )
    insert_rows(gateway, metrics_rows)
    insert_rows(
        gateway,
        [
            CoverageFunctionRow(
                function_goid_h128=1,
                urn="urn:fn",
                repo="r",
                commit="c",
                rel_path="m.py",
                language="python",
                kind="function",
                qualname="m.fn",
                start_line=1,
                end_line=2,
                executable_lines=2,
                covered_lines=2,
                coverage_ratio=1.0,
                tested=True,
                last_status="",
                created_at=now,
            ),
            CoverageLineRow(
                repo="r",
                commit="c",
                rel_path="m.py",
                line=1,
                is_executable=True,
                is_covered=True,
                hits=1,
                context_count=1,
                created_at=now,
            ),
            RiskFactorRow(
                function_goid_h128=1,
                urn="urn:fn",
                repo="r",
                commit="c",
                rel_path="m.py",
                language="python",
                kind="function",
                qualname="m.fn",
                loc=4,
                logical_loc=3,
                cyclomatic_complexity=1,
                risk_score=0.0,
                risk_level="low",
                typedness_bucket="typed",
                hotspot_reason="analysis",
                typedness_score=1.0,
                complexity_score=0.0,
                hotspot_score=0.0,
                has_tests=True,
                coverage_functions=2,
                coverage_lines=2,
                coverage_ratio=1.0,
                tested=True,
                total_tests=1,
                flaky_tests=0,
                last_status="passed",
                risk_weight=0.1,
                risk_component_coverage="low",
                risk_component_static="[]",
                risk_component_hotspot="[]",
                created_at=now,
            ),
            ConfigValueRow(
                repo="r",
                commit="c",
                path="cfg.yaml",
                format="yaml",
                key="feature.flag",
                raw_value="[]",
                parsed_value='["pkg.m"]',
                version=1,
            ),
            TypednessRow(
                repo="r",
                commit="c",
                path="m.py",
                type_error_count=0,
                annotation_ratio='{"params":1}',
                untyped_defs=0,
                overlay_needed=False,
            ),
            StaticDiagnosticsRow(
                repo="r",
                commit="c",
                rel_path="m.py",
                type_error_count=0,
                lint_error_count=0,
                format_error_count=0,
                security_error_count=0,
                has_blocking_errors=False,
            ),
        ],
    )
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
    insert_rows(
        gateway,
        [
            SubsystemRow(
                repo="r",
                commit="c",
                subsystem_id="sub1",
                name="Subsystem",
                description=None,
                module_count=1,
                modules_json='["m"]',
                entrypoints_json="[]",
                internal_edge_count=1,
                external_edge_count=0,
                fan_in=1,
                fan_out=0,
                function_count=1,
                avg_risk_score=0.1,
                max_risk_score=0.1,
                high_risk_function_count=0,
                risk_level="low",
                import_in_degree=None,
                import_out_degree=None,
                import_pagerank=None,
                import_betweenness=None,
                import_closeness=None,
                import_layer=None,
                created_at=now,
            ),
            SubsystemModuleRow(
                repo="r",
                commit="c",
                subsystem_id="sub1",
                module="m",
                member_kind="member",
            ),
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
    }
    for key, value in counts.items():
        minimum = expected_min.get(key, 1)
        if value < minimum:
            pytest.fail(f"Insert helper row counts mismatch for {key}: {value} < {minimum}")

    con.close()
