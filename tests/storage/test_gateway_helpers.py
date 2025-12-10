"""Validate StorageGateway insert helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.utilities.datasets import (
    get_analytics_dataset_contract,
    insert_analytics_rows,
)
from codeintel.config.datasets import GraphMetricsFunctionsRow, GraphMetricsModulesRow
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    CoverageFunctionRow,
    CoverageLineRow,
    FunctionMetricsRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    SubsystemModuleRow,
    SubsystemRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
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

    insert_rows(
        gateway,
        [RepoMapRow(repo="r", commit="c", modules={}, overlays={}, generated_at=now)],
    )
    insert_rows(gateway, [ModuleRow(module="m", path="m.py", repo="r", commit="c")])
    insert_rows(
        gateway,
        [
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
            )
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
            )
        ],
    )
    insert_rows(
        gateway,
        [
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
            )
        ],
    )
    insert_rows(
        gateway,
        [
            ImportGraphEdgeRow(
                repo="r",
                commit="c",
                src_module="m",
                dst_module="m",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
                module_layer=None,
            )
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
                untested_reason=None,
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
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
            )
        ],
    )
    insert_rows(
        gateway,
        [
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
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=2,
                covered_lines=2,
                coverage_ratio=1.0,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="passed",
                risk_score=0.0,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            ConfigValueRow(
                repo="r",
                commit="c",
                config_path="cfg.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=["cfg.yaml"],
                reference_modules=["pkg.m"],
                reference_count=1,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            TypednessRow(
                repo="r",
                commit="c",
                path="m.py",
                type_error_count=0,
                annotation_ratio='{"params":1}',
                untyped_defs=0,
                overlay_needed=False,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            StaticDiagnosticsRow(
                repo="r",
                commit="c",
                rel_path="m.py",
                pyrefly_errors=0,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=0,
                has_errors=False,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            TestCatalogRow(
                test_id="test_case",
                test_goid_h128=10,
                urn="urn:test",
                repo="r",
                commit="c",
                rel_path="m_test.py",
                qualname="test_case",
                kind="unit",
                status="passed",
                duration_ms=5,
                markers="[]",
                parametrized=False,
                flaky=False,
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            TestCoverageEdgeRow(
                test_id="test_case",
                test_goid_h128=10,
                function_goid_h128=1,
                urn="urn:fn",
                repo="r",
                commit="c",
                rel_path="m.py",
                qualname="m.fn",
                covered_lines=2,
                executable_lines=2,
                coverage_ratio=1.0,
                last_status="passed",
                created_at=now,
            )
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
                description="Subsystem",
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
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            SubsystemModuleRow(
                repo="r",
                commit="c",
                subsystem_id="sub1",
                module="m",
                role="member",
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
    }
    for key, value in counts.items():
        minimum = expected_min.get(key, 1)
        if value < minimum:
            pytest.fail(f"Insert helper row counts mismatch for {key}: {value} < {minimum}")

    con.close()
