"""Smoke tests for symbol/config graph metrics and subsystem agreement."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.graphs import (
    compute_config_graph_metrics,
    compute_subsystem_agreement,
    compute_symbol_graph_metrics_modules,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.views import create_all_views
from tests._helpers.builders import (
    ConfigValueRow,
    GraphMetricsModulesExtRow,
    ModuleRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolUseEdgeRow,
    insert_config_values,
    insert_graph_metrics_modules_ext,
    insert_modules,
    insert_subsystem_modules,
    insert_subsystems,
    insert_symbol_use_edges,
)

REPO = "demo/repo"
COMMIT = "abc123"
EXPECTED_SYMBOL_ROW_COUNT = 2


def test_symbol_and_config_metrics_populate_and_views_create(
    fresh_gateway: StorageGateway,
) -> None:
    """Compute symbol/config metrics and verify derived views materialize."""
    gateway = fresh_gateway
    con = gateway.con
    insert_modules(
        gateway,
        [
            ModuleRow(module="pkg.a", path="pkg/a.py", repo=REPO, commit=COMMIT),
            ModuleRow(module="pkg.b", path="pkg/b.py", repo=REPO, commit=COMMIT),
        ],
    )
    insert_symbol_use_edges(
        gateway,
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
    insert_config_values(
        gateway,
        [
            ConfigValueRow(
                repo=REPO,
                commit=COMMIT,
                config_path="cfg/app.yaml",
                format="yaml",
                key="feature.flag",
                reference_paths=[],
                reference_modules=["pkg.a", "pkg.b"],
                reference_count=2,
            )
        ],
    )

    compute_symbol_graph_metrics_modules(gateway, repo=REPO, commit=COMMIT)
    compute_config_graph_metrics(gateway, repo=REPO, commit=COMMIT)
    create_all_views(con)

    sym_rows = con.execute(
        "SELECT module, symbol_community_id FROM analytics.symbol_graph_metrics_modules"
    ).fetchall()
    cfg_keys = con.execute("SELECT config_key FROM analytics.config_graph_metrics_keys").fetchall()
    cfg_modules = con.execute(
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
    # Views created
    con.execute("SELECT * FROM docs.v_symbol_module_graph")
    con.execute("SELECT * FROM analytics.config_graph_metrics_keys")
    con.execute("SELECT * FROM analytics.config_projection_module_edges")


def test_subsystem_agreement_exposed_in_views(
    fresh_gateway: StorageGateway,
) -> None:
    """Expose subsystem agreement results through docs views."""
    gateway = fresh_gateway
    con = gateway.con
    now = datetime.now(UTC)
    insert_subsystem_modules(
        gateway,
        [
            SubsystemModuleRow(
                repo=REPO,
                commit=COMMIT,
                subsystem_id="sub1",
                module="pkg.a",
                role="core",
            )
        ],
    )
    insert_graph_metrics_modules_ext(
        gateway,
        [
            GraphMetricsModulesExtRow(
                repo=REPO,
                commit=COMMIT,
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
    insert_subsystems(
        gateway,
        [
            SubsystemRow(
                repo=REPO,
                commit=COMMIT,
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

    compute_subsystem_agreement(gateway, repo=REPO, commit=COMMIT)
    create_all_views(con)

    agree_rows = con.execute("SELECT module, agrees FROM docs.v_subsystem_agreement").fetchall()
    summary = con.execute(
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
