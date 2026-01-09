"""Smoke tests for symbol/config graph metrics and subsystem agreement."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.analytics.graphs.config_graph_metrics import (
    ConfigGraphMetricsRequest,
    compute_config_graph_metrics_result,
)
from codeintel.build.analytics.graphs.subsystem_agreement import (
    SubsystemAgreementInputs,
    build_subsystem_agreement_rows,
)
from codeintel.build.analytics.graphs.symbol_graph_metrics import (
    build_symbol_graph_metrics_module_rows,
)
from codeintel.build.graphs.builders import build_symbol_module_graph_from_tables
from codeintel.core.columnar.conversion import tabular_to_arrow_table
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.storage.query_results import records_from_arrow_table, records_from_relation
from tests._helpers.columnar_streams import table_for_rows
from tests._helpers.docs_views import materialize_view_plans
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
    from codeintel.storage.gateway import StorageGateway
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
                repo=ctx.repo,
                commit=ctx.commit,
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


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _module_inputs_for_symbol_metrics(test_ctx: TestContext) -> tuple[dict[str, str], set[str]]:
    module_rows = _records_for_table(
        test_ctx,
        "core.modules",
        ["module", "path", "repo", "commit"],
    )
    module_by_path: dict[str, str] = {}
    known_modules: set[str] = set()
    for row in module_rows:
        module = row.get("module")
        if module is None:
            continue
        if not _matches_optional_scope(row.get("repo"), test_ctx.repo):
            continue
        if not _matches_optional_scope(row.get("commit"), test_ctx.commit):
            continue
        module_name = str(module)
        known_modules.add(module_name)
        path = row.get("path")
        if path is not None:
            module_by_path[str(path)] = module_name
    return module_by_path, known_modules


def _records_for_table(
    test_ctx: TestContext,
    table_key: str,
    columns: list[str],
) -> list[dict[str, object]]:
    if test_ctx.gateway.config.dataset_root_dir is None:
        column_clause = ", ".join(columns)
        table = tabular_to_arrow_table(
            test_ctx.gateway.con.sql(f"SELECT {column_clause} FROM {table_key}")
        )
        return records_from_arrow_table(table)
    return records_from_relation(
        test_ctx.gateway.relation_from_table_key(table_key).select(*columns)
    )


def _write_rows_for_snapshot(
    gateway: StorageGateway,
    *,
    table_key: str,
    repo: str,
    commit: str,
    rows: Sequence[tuple[object, ...]] | ColumnarRowBuffer | None,
) -> None:
    if isinstance(rows, ColumnarRowBuffer):
        rows = rows.to_tuples()
    if not rows:
        return
    gateway.policy.delete_for_snapshot(table_key, repo=repo, commit=commit)
    gateway.policy.bulk_insert(table_key, rows)


def _write_symbol_graph_metrics(
    ctx: TestContext,
    *,
    module_by_path: dict[str, str],
    known_modules: set[str],
) -> None:
    symbol_use_rows = _records_for_table(
        ctx,
        "graph.symbol_use_edges",
        ["def_path", "use_path", "def_goid_h128", "use_goid_h128"],
    )
    symbol_use_table = table_for_rows("graph.symbol_use_edges", symbol_use_rows)
    module_map_table = table_for_rows(
        "core.modules",
        [{"path": path, "module": module} for path, module in module_by_path.items()],
    )
    symbol_graph = build_symbol_module_graph_from_tables(symbol_use_table, module_map_table)
    symbol_rows = build_symbol_graph_metrics_module_rows(
        repo=ctx.repo,
        commit=ctx.commit,
        graph=symbol_graph,
        known_modules=known_modules or None,
    )
    _write_rows_for_snapshot(
        ctx.gateway,
        table_key="analytics.symbol_graph_metrics_modules",
        repo=ctx.repo,
        commit=ctx.commit,
        rows=symbol_rows,
    )


def _write_config_graph_metrics(
    ctx: TestContext,
    *,
    config_value_rows: Sequence[dict[str, object]],
    allowed_modules: set[str],
) -> None:
    config_rows = compute_config_graph_metrics_result(
        ConfigGraphMetricsRequest(
            repo=ctx.repo,
            commit=ctx.commit,
            config_value_rows=config_value_rows,
            allowed_modules=allowed_modules,
        )
    )
    _write_rows_for_snapshot(
        ctx.gateway,
        table_key="analytics.config_graph_metrics_keys",
        repo=ctx.repo,
        commit=ctx.commit,
        rows=config_rows.key_rows,
    )
    _write_rows_for_snapshot(
        ctx.gateway,
        table_key="analytics.config_graph_metrics_modules",
        repo=ctx.repo,
        commit=ctx.commit,
        rows=config_rows.module_rows,
    )
    _write_rows_for_snapshot(
        ctx.gateway,
        table_key="analytics.config_projection_key_edges",
        repo=ctx.repo,
        commit=ctx.commit,
        rows=config_rows.key_edge_rows,
    )
    _write_rows_for_snapshot(
        ctx.gateway,
        table_key="analytics.config_projection_module_edges",
        repo=ctx.repo,
        commit=ctx.commit,
        rows=config_rows.module_edge_rows,
    )


def _assert_symbol_config_outputs(ctx: TestContext) -> None:
    sym_rows = ctx.con.execute(
        "SELECT module, symbol_community_id FROM analytics.symbol_graph_metrics_modules"
    ).fetchall()
    cfg_keys = ctx.con.execute(
        "SELECT config_key FROM analytics.config_graph_metrics_keys"
    ).fetchall()
    cfg_modules = ctx.con.execute(
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
    module_by_path, known_modules = _module_inputs_for_symbol_metrics(test_ctx)
    _write_symbol_graph_metrics(
        test_ctx,
        module_by_path=module_by_path,
        known_modules=known_modules,
    )
    config_value_rows = _records_for_table(
        test_ctx,
        "analytics.config_values",
        ["repo", "commit", "key", "extras"],
    )
    _write_config_graph_metrics(
        test_ctx,
        config_value_rows=config_value_rows,
        allowed_modules=known_modules,
    )
    _assert_symbol_config_outputs(test_ctx)
    test_ctx.con.execute("SELECT * FROM analytics.config_graph_metrics_keys")
    test_ctx.con.execute("SELECT * FROM analytics.config_projection_module_edges")
    if test_ctx.gateway.config.dataset_root_dir is None:
        return
    materialize_view_plans(test_ctx.con)
    test_ctx.con.execute("SELECT * FROM docs.v_symbol_module_graph")


def test_subsystem_agreement_exposed_in_views(
    test_ctx: TestContext,
) -> None:
    """Verify subsystem agreement results exposed through docs views."""
    _seed_subsystem_agreement_data(test_ctx)

    subsystem_rows = _records_for_table(
        test_ctx,
        "analytics.subsystem_modules",
        ["repo", "commit", "module", "subsystem_id"],
    )
    graph_metrics_rows = _records_for_table(
        test_ctx,
        "analytics.graph_metrics_modules_ext",
        ["repo", "commit", "module", "import_community_id"],
    )
    agreement_rows = build_subsystem_agreement_rows(
        SubsystemAgreementInputs(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            subsystem_module_rows=subsystem_rows,
            graph_metrics_module_rows=graph_metrics_rows,
        )
    )
    if agreement_rows:
        test_ctx.gateway.policy.delete_for_snapshot(
            "analytics.subsystem_agreement",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        )
        test_ctx.gateway.policy.bulk_insert("analytics.subsystem_agreement", agreement_rows)
    if test_ctx.gateway.config.dataset_root_dir is None:
        agree_rows = test_ctx.con.execute(
            """
            SELECT module, agrees
            FROM analytics.subsystem_agreement
            WHERE subsystem_id = 'sub1'
            """
        ).fetchall()
        summary = test_ctx.con.execute(
            """
            SELECT
                SUM(CASE WHEN sa.agrees = FALSE THEN 1 ELSE 0 END) AS disagree_count,
                CASE
                    WHEN COUNT(sm.module) = 0 THEN NULL
                    ELSE CAST(SUM(CASE WHEN sa.agrees = TRUE THEN 1 ELSE 0 END) AS DOUBLE)
                        / COUNT(sm.module)
                END AS agreement_ratio
            FROM analytics.subsystem_modules AS sm
            LEFT JOIN analytics.subsystem_agreement AS sa
              ON sm.repo = sa.repo
             AND sm.commit = sa.commit
             AND sm.subsystem_id = sa.subsystem_id
             AND sm.module = sa.module
            WHERE sm.subsystem_id = 'sub1'
            """
        ).fetchone()
    else:
        materialize_view_plans(test_ctx.con)
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
