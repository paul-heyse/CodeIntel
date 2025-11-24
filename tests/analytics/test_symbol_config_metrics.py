"""Smoke tests for symbol/config graph metrics and subsystem agreement."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.symbol_graph_metrics import compute_symbol_graph_metrics_modules
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.views import create_all_views

REPO = "demo/repo"
COMMIT = "abc123"
EXPECTED_SYMBOL_ROW_COUNT = 2


def _db() -> StorageGateway:
    return open_memory_gateway(apply_schema=True)


def test_symbol_and_config_metrics_populate_and_views_create() -> None:
    """Compute symbol/config metrics and verify derived views materialize."""
    gateway = _db()
    con = gateway.con
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            ('pkg.a', 'pkg/a.py', ?, ?, 'python', '[]', '[]'),
            ('pkg.b', 'pkg/b.py', ?, ?, 'python', '[]', '[]')
        """,
        [REPO, COMMIT, REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO graph.symbol_use_edges (symbol, def_path, use_path, same_file, same_module)
        VALUES ('sym', 'pkg/a.py', 'pkg/b.py', FALSE, FALSE)
        """
    )
    con.execute(
        """
        INSERT INTO analytics.config_values (
            config_path, format, key, reference_paths, reference_modules, reference_count
        ) VALUES ('cfg/app.yaml', 'yaml', 'feature.flag', '[]', '["pkg.a","pkg.b"]', 2)
        """
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
    con.execute("SELECT * FROM docs.v_config_graph_metrics_keys")
    con.execute("SELECT * FROM docs.v_config_projection_module_edges")


def test_subsystem_agreement_exposed_in_views() -> None:
    """Expose subsystem agreement results through docs views."""
    gateway = _db()
    con = gateway.con
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO analytics.subsystem_modules (repo, commit, subsystem_id, module, role)
        VALUES (?, ?, 'sub1', 'pkg.a', 'core')
        """,
        [REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO analytics.graph_metrics_modules_ext (
            repo, commit, module, import_betweenness, import_closeness, import_eigenvector,
            import_harmonic, import_k_core, import_constraint, import_effective_size,
            import_community_id, import_component_id, import_component_size,
            import_scc_id, import_scc_size, created_at
        ) VALUES (
            ?, ?, 'pkg.a', 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0,
            2, 0, 1, 0, 1, ?
        )
        """,
        [REPO, COMMIT, now],
    )
    con.execute(
        """
        INSERT INTO analytics.subsystems (
            repo, commit, subsystem_id, name, description, module_count, modules_json,
            entrypoints_json, internal_edge_count, external_edge_count, fan_in, fan_out,
            function_count, avg_risk_score, max_risk_score, high_risk_function_count,
            risk_level, created_at
        ) VALUES (
            ?, ?, 'sub1', 'sub1', 'desc', 1, '["pkg.a"]', '[]', 0, 0, 0, 0,
            0, NULL, NULL, 0, 'low', ?
        )
        """,
        [REPO, COMMIT, now],
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
