"""Graph stats and validation coverage for symbol/config graphs and subsystem agreement."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.analytics.graphs.config_graph_metrics import build_config_module_bipartite
from codeintel.build.analytics.graphs.graph_metrics import (
    build_call_graph_from_rows,
    build_import_graph_from_rows,
)
from codeintel.build.analytics.graphs.graph_stats import (
    GraphStatsInputs,
    build_graph_stats_rows,
)
from codeintel.build.analytics.graphs.subsystem_agreement import (
    SubsystemAgreementInputs,
    build_subsystem_agreement_rows,
)
from codeintel.build.analytics.graphs.symbol_graph_metrics import (
    build_symbol_function_graph,
    build_symbol_module_graph,
)
from codeintel.build.graphs.engine import NxGraphEngine
from codeintel.build.graphs.validation import warn_graph_structure
from codeintel.storage.query_results import records_from_relation
from tests._helpers.docs_views import materialize_view_plans
from tests._helpers.fixtures.rows import (
    ConfigValueRow,
    GraphMetricsModulesExtRow,
    ModuleRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolGraphMetricsModulesRow,
    SymbolUseEdgeRow,
    insert_rows,
)

if TYPE_CHECKING:
    from tests._helpers import TestContext


def _seed_test_modules(ctx: TestContext) -> None:
    """Seed basic modules for graph tests.

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    insert_rows(
        ctx.gateway,
        [
            ModuleRow(module="pkg.a", path="pkg/a.py", repo=ctx.repo, commit=ctx.commit),
            ModuleRow(module="pkg.b", path="pkg/b.py", repo=ctx.repo, commit=ctx.commit),
            ModuleRow(module="pkg.c", path="pkg/c.py", repo=ctx.repo, commit=ctx.commit),
        ],
    )


def _module_inputs(ctx: TestContext) -> tuple[dict[str, str], set[str]]:
    module_rows = records_from_relation(
        ctx.gateway.relation_from_table_key("core.modules").select(
            "module",
            "path",
            "repo",
            "commit",
        )
    )
    module_by_path: dict[str, str] = {}
    module_names: set[str] = set()
    for row in module_rows:
        module = row.get("module")
        path = row.get("path")
        if module is None or path is None:
            continue
        if str(row.get("repo")) != ctx.repo or str(row.get("commit")) != ctx.commit:
            continue
        module_name = str(module)
        module_names.add(module_name)
        module_by_path[str(path)] = module_name
    return module_by_path, module_names


def test_graph_stats_include_symbol_and_config_graphs(graph_ctx: TestContext) -> None:
    """Verify graph_stats covers symbol, function, and config projections."""
    _seed_test_modules(graph_ctx)
    module_by_path, module_names = _module_inputs(graph_ctx)

    insert_rows(
        graph_ctx.gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym1",
                def_path="pkg/a.py",
                use_path="pkg/b.py",
                same_file=False,
                same_module=False,
                def_goid_h128=1,
                use_goid_h128=2,
            )
        ],
    )
    insert_rows(
        graph_ctx.gateway,
        [
            ConfigValueRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                config_path="cfg/app.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=[],
                reference_modules=["pkg.a", "pkg.b"],
                reference_count=2,
            )
        ],
    )

    symbol_rows = records_from_relation(
        graph_ctx.gateway.relation_from_table_key("graph.symbol_use_edges").select(
            "def_path",
            "use_path",
            "def_goid_h128",
            "use_goid_h128",
        )
    )
    symbol_module_graph = build_symbol_module_graph(symbol_rows, module_by_path)
    symbol_function_graph = build_symbol_function_graph(symbol_rows)
    config_rows = records_from_relation(
        graph_ctx.gateway.relation_from_table_key("analytics.config_values").select(
            "repo",
            "commit",
            "key",
            "reference_modules",
        )
    )
    config_bipartite = build_config_module_bipartite(
        config_rows,
        allowed_modules=module_names,
        repo=graph_ctx.repo,
        commit=graph_ctx.commit,
    )
    call_graph = build_call_graph_from_rows([], [])
    import_graph = build_import_graph_from_rows([], [])
    stats_rows = build_graph_stats_rows(
        GraphStatsInputs(
            repo=graph_ctx.repo,
            commit=graph_ctx.commit,
            call_graph=call_graph,
            import_graph=import_graph,
            symbol_module_graph=symbol_module_graph,
            symbol_function_graph=symbol_function_graph,
            config_module_bipartite=config_bipartite,
            use_gpu=False,
        )
    )
    if stats_rows:
        graph_ctx.gateway.policy.delete_for_snapshot(
            "analytics.graph_stats",
            repo=graph_ctx.repo,
            commit=graph_ctx.commit,
        )
        graph_ctx.gateway.policy.bulk_insert("analytics.graph_stats", stats_rows)

    rows = graph_ctx.query(
        "SELECT graph_name FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [graph_ctx.repo, graph_ctx.commit],
    )
    names = {str(row.graph_name) for row in rows}
    expected = {
        "symbol_module_graph",
        "symbol_function_graph",
        "config_key_projection",
        "config_module_projection",
    }
    if not expected.issubset(names):
        pytest.fail(f"Missing expected graphs: {expected - names}")


def test_subsystem_agreement_summary_aggregates(graph_ctx: TestContext) -> None:
    """Validate subsystem agreement summary aggregates disagreement counts."""
    now = datetime.now(UTC)

    insert_rows(
        graph_ctx.gateway,
        [
            SubsystemModuleRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                subsystem_id="sub1",
                module="pkg.a",
                role="core",
            )
        ],
    )
    insert_rows(
        graph_ctx.gateway,
        [
            GraphMetricsModulesExtRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                module="pkg.a",
                import_betweenness=0.0,
                import_closeness=0.0,
                import_eigenvector=0.0,
                import_harmonic=0.0,
                import_k_core=1,
                import_constraint=0.0,
                import_effective_size=0.0,
                import_community_id=2,
                import_component_id=0,
                import_component_size=1,
                import_scc_id=0,
                import_scc_size=1,
                created_at=now,
            )
        ],
    )
    insert_rows(
        graph_ctx.gateway,
        [
            SubsystemRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                subsystem_id="sub1",
                name="sub1",
                description="desc",
                module_count=1,
                modules_json=["pkg.a"],
                entrypoints_json=[],
                internal_edge_count=0,
                external_edge_count=0,
                fan_in=0,
                fan_out=0,
                function_count=0,
                avg_risk_score=None,
                max_risk_score=None,
                high_risk_function_count=0,
                risk_level="low",
                created_at=now,
            )
        ],
    )

    subsystem_rows = records_from_relation(
        graph_ctx.gateway.relation_from_table_key("analytics.subsystem_modules").select(
            "repo",
            "commit",
            "module",
            "subsystem_id",
        )
    )
    graph_metrics_rows = records_from_relation(
        graph_ctx.gateway.relation_from_table_key("analytics.graph_metrics_modules_ext").select(
            "repo",
            "commit",
            "module",
            "import_community_id",
        )
    )
    agreement_rows = build_subsystem_agreement_rows(
        SubsystemAgreementInputs(
            repo=graph_ctx.repo,
            commit=graph_ctx.commit,
            subsystem_module_rows=subsystem_rows,
            graph_metrics_module_rows=graph_metrics_rows,
        )
    )
    if agreement_rows:
        graph_ctx.gateway.policy.delete_for_snapshot(
            "analytics.subsystem_agreement",
            repo=graph_ctx.repo,
            commit=graph_ctx.commit,
        )
        graph_ctx.gateway.policy.bulk_insert("analytics.subsystem_agreement", agreement_rows)
    materialize_view_plans(graph_ctx.con)

    disagree_row = graph_ctx.con.execute(
        """
        SELECT subsystem_disagree_count, subsystem_agreement_ratio
        FROM docs.v_subsystem_summary
        WHERE subsystem_id = 'sub1'
        """
    ).fetchone()
    if disagree_row is None:
        pytest.fail("Expected subsystem summary row for sub1")
    disagree_count, ratio = disagree_row
    if disagree_count != 1:
        pytest.fail(f"Expected disagree count 1, got {disagree_count}")
    if ratio != 0.0:
        pytest.fail(f"Expected agreement ratio 0.0, got {ratio}")


def test_validation_flags_large_symbol_community_and_config_hubs(
    graph_ctx: TestContext,
) -> None:
    """Surface validation warnings for oversized symbol communities and config hubs."""
    _seed_test_modules(graph_ctx)
    now = datetime.now(UTC)

    insert_rows(
        graph_ctx.gateway,
        [
            SymbolGraphMetricsModulesRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                module="pkg.a",
                symbol_betweenness=0.0,
                symbol_closeness=0.0,
                symbol_eigenvector=0.0,
                symbol_harmonic=0.0,
                symbol_k_core=1,
                symbol_constraint=0.0,
                symbol_effective_size=0.0,
                symbol_community_id=99,
                symbol_component_id=0,
                symbol_component_size=1,
                created_at=now,
            ),
            SymbolGraphMetricsModulesRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                module="pkg.b",
                symbol_betweenness=0.0,
                symbol_closeness=0.0,
                symbol_eigenvector=0.0,
                symbol_harmonic=0.0,
                symbol_k_core=1,
                symbol_constraint=0.0,
                symbol_effective_size=0.0,
                symbol_community_id=99,
                symbol_component_id=0,
                symbol_component_size=1,
                created_at=now,
            ),
            SymbolGraphMetricsModulesRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                module="pkg.c",
                symbol_betweenness=0.0,
                symbol_closeness=0.0,
                symbol_eigenvector=0.0,
                symbol_harmonic=0.0,
                symbol_k_core=1,
                symbol_constraint=0.0,
                symbol_effective_size=0.0,
                symbol_community_id=99,
                symbol_component_id=0,
                symbol_component_size=1,
                created_at=now,
            ),
        ],
    )
    insert_rows(
        graph_ctx.gateway,
        [
            ConfigValueRow(
                repo=graph_ctx.repo,
                commit=graph_ctx.commit,
                config_path="cfg/app.yaml",
                format="yaml",
                key="wide.key",
                reference_paths=[],
                reference_modules=["pkg.a", "pkg.b", "pkg.c"],
                reference_count=3,
            )
        ],
    )

    engine = NxGraphEngine(
        dataset_root_dir=None,
        snapshot=graph_ctx.to_snapshot_ref(),
    )
    findings = warn_graph_structure(engine, graph_ctx.repo, graph_ctx.commit, log=None)
    check_names = {f["check_name"] for f in findings}
    if "symbol_graph_large_community" not in check_names:
        pytest.fail("Expected symbol_graph_large_community finding")
    if "config_keys_broad_usage" not in check_names:
        pytest.fail("Expected config_keys_broad_usage finding")
