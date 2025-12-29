"""Smoke tests for symbol/config graph metrics and subsystem agreement."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics_result
from codeintel.analytics.graphs.subsystem_agreement import build_subsystem_agreement_rows
from codeintel.analytics.graphs.symbol_graph_metrics import build_symbol_graph_metrics_module_rows
from tests._helpers.fixtures.rows import (
    ConfigValueRow,
    GraphMetricsModulesExtRow,
    ModuleRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolUseEdgeRow,
    insert_rows,
)

if TYPE_CHECKING:
    from tests._helpers import TestContext


EXPECTED_SYMBOL_ROW_COUNT = 2


def _seed_symbol_config_data(ctx: TestContext) -> None:
    """Seed modules, symbol edges, and config values.

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
        ],
    )
    insert_rows(
        ctx.gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym",
                def_path="pkg/a.py",
                use_path="pkg/b.py",
                same_file=False,
                same_module=False,
            )
        ],
    )
    insert_rows(
        ctx.gateway,
        [
            ConfigValueRow(
                repo=ctx.repo,
                commit=ctx.commit,
                config_path="cfg/app.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=[],
                reference_modules=["pkg.a", "pkg.b"],
                reference_count=2,
            )
        ],
    )


def _seed_subsystem_agreement_data(ctx: TestContext) -> None:
    """Seed data for subsystem agreement testing.

    Creates a subsystem with one module that has a different community ID
    in graph metrics, resulting in disagreement.

    Parameters
    ----------
    ctx
        Test context with gateway.
    """
    now = datetime.now(UTC)
    insert_rows(
        ctx.gateway,
        [
            SubsystemModuleRow(
                repo=ctx.repo,
                commit=ctx.commit,
                subsystem_id="sub1",
                module="pkg.a",
                role="core",
            )
        ],
    )
    insert_rows(
        ctx.gateway,
        [
            GraphMetricsModulesExtRow(
                repo=ctx.repo,
                commit=ctx.commit,
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
        ctx.gateway,
        [
            SubsystemRow(
                repo=ctx.repo,
                commit=ctx.commit,
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


def test_symbol_and_config_metrics_populate_and_views_create(
    test_ctx: TestContext,
) -> None:
    """Verify symbol/config metrics compute and derived views materialize."""
    _seed_symbol_config_data(test_ctx)

    symbol_rows = build_symbol_graph_metrics_module_rows(
        test_ctx.gateway,
        repo=test_ctx.repo,
        commit=test_ctx.commit,
    )
    if symbol_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.symbol_graph_metrics_modules",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert("analytics.symbol_graph_metrics_modules", symbol_rows)

    config_rows = compute_config_graph_metrics_result(
        test_ctx.gateway,
        repo=test_ctx.repo,
        commit=test_ctx.commit,
    )
    if config_rows.key_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.config_graph_metrics_keys",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert(
            "analytics.config_graph_metrics_keys", config_rows.key_rows
        )
    if config_rows.module_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.config_graph_metrics_modules",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert(
            "analytics.config_graph_metrics_modules",
            config_rows.module_rows,
        )
    if config_rows.key_edge_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.config_projection_key_edges",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert(
            "analytics.config_projection_key_edges",
            config_rows.key_edge_rows,
        )
    if config_rows.module_edge_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.config_projection_module_edges",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert(
            "analytics.config_projection_module_edges",
            config_rows.module_edge_rows,
        )
    test_ctx.gateway.policy.ensure_all_views(overwrite=True, strict=True)

    sym_rows = test_ctx.con.execute(
        "SELECT module, symbol_community_id FROM analytics.symbol_graph_metrics_modules"
    ).fetchall()
    cfg_keys = test_ctx.con.execute(
        "SELECT config_key FROM analytics.config_graph_metrics_keys"
    ).fetchall()
    cfg_modules = test_ctx.con.execute(
        "SELECT module FROM analytics.config_graph_metrics_modules"
    ).fetchall()

    if len(sym_rows) != EXPECTED_SYMBOL_ROW_COUNT:
        pytest.fail(f"Expected {EXPECTED_SYMBOL_ROW_COUNT} symbol rows, got {len(sym_rows)}")
    if not any(row[1] is not None for row in sym_rows):
        pytest.fail("Expected at least one non-null symbol_community_id")
    if cfg_keys != [("feature.flag",)]:
        pytest.fail(f"Unexpected config keys: {cfg_keys}")
    modules = {row[0] for row in cfg_modules}
    if modules != {"pkg.a", "pkg.b"}:
        pytest.fail(f"Unexpected config modules: {modules}")

    test_ctx.con.execute("SELECT * FROM docs.v_symbol_module_graph")
    test_ctx.con.execute("SELECT * FROM analytics.config_graph_metrics_keys")
    test_ctx.con.execute("SELECT * FROM analytics.config_projection_module_edges")


def test_subsystem_agreement_exposed_in_views(
    test_ctx: TestContext,
) -> None:
    """Verify subsystem agreement results exposed through docs views."""
    _seed_subsystem_agreement_data(test_ctx)

    agreement_rows = build_subsystem_agreement_rows(
        test_ctx.gateway,
        repo=test_ctx.repo,
        commit=test_ctx.commit,
    )
    if agreement_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.subsystem_agreement",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert("analytics.subsystem_agreement", agreement_rows)
    test_ctx.gateway.policy.ensure_all_views(overwrite=True, strict=True)

    agree_rows = test_ctx.con.execute(
        "SELECT module, agrees FROM docs.v_subsystem_agreement"
    ).fetchall()
    summary = test_ctx.con.execute(
        """
        SELECT subsystem_disagree_count, subsystem_agreement_ratio
        FROM docs.v_subsystem_summary
        WHERE subsystem_id = 'sub1'
        """
    ).fetchone()

    if agree_rows != [("pkg.a", False)]:
        pytest.fail(f"Unexpected subsystem agreement rows: {agree_rows}")
    if summary is None:
        pytest.fail("Expected subsystem summary row for sub1")
    disagree_count, ratio = summary
    if disagree_count != 1:
        pytest.fail(f"Expected disagree count 1, got {disagree_count}")
    if ratio != 0.0:
        pytest.fail(f"Expected agreement ratio 0.0, got {ratio}")
