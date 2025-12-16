"""Graph stats and validation coverage for symbol/config graphs and subsystem agreement."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.graphs import (
    compute_graph_stats,
    compute_subsystem_agreement,
)
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.validation import warn_graph_structure
from tests._helpers.builders import (
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


def test_graph_stats_include_symbol_and_config_graphs(graph_ctx: TestContext) -> None:
    """Verify graph_stats covers symbol, function, and config projections."""
    _seed_test_modules(graph_ctx)

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

    compute_graph_stats(graph_ctx.gateway, repo=graph_ctx.repo, commit=graph_ctx.commit)

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
                modules_json='["pkg.a"]',
                entrypoints_json="[]",
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

    compute_subsystem_agreement(
        graph_ctx.gateway,
        repo=graph_ctx.repo,
        commit=graph_ctx.commit,
    )
    graph_ctx.gateway.policy.ensure_all_views(overwrite=True, strict=True)

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
        gateway=graph_ctx.gateway,
        snapshot=graph_ctx.to_snapshot_ref(),
    )
    findings = warn_graph_structure(engine, graph_ctx.repo, graph_ctx.commit, log=None)
    check_names = {f["check_name"] for f in findings}
    if "symbol_graph_large_community" not in check_names:
        pytest.fail("Expected symbol_graph_large_community finding")
    if "config_keys_broad_usage" not in check_names:
        pytest.fail("Expected config_keys_broad_usage finding")
