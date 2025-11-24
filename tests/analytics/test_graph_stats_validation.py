"""Graph stats and validation coverage for symbol/config graphs and subsystem agreement."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.analytics.graph_stats import compute_graph_stats
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement
from codeintel.graphs.validation import warn_graph_structure
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.views import create_all_views

REPO = "demo/repo"
COMMIT = "abc123"


def _gateway() -> StorageGateway:
    return open_memory_gateway(apply_schema=True)


def _seed_modules(gateway: StorageGateway) -> None:
    gateway.con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            ('pkg.a', 'pkg/a.py', ?, ?, 'python', '[]', '[]'),
            ('pkg.b', 'pkg/b.py', ?, ?, 'python', '[]', '[]'),
            ('pkg.c', 'pkg/c.py', ?, ?, 'python', '[]', '[]')
        """,
        [REPO, COMMIT, REPO, COMMIT, REPO, COMMIT],
    )


def test_graph_stats_include_symbol_and_config_graphs() -> None:
    """Ensure graph_stats covers symbol, function, and config projections."""
    gateway = _gateway()
    con = gateway.con
    _seed_modules(gateway)
    # Symbol edges with GOIDs for function graph
    con.execute(
        """
        INSERT INTO graph.symbol_use_edges (
            symbol, def_path, use_path, same_file, same_module,
            def_goid_h128, use_goid_h128
        ) VALUES
            ('sym1', 'pkg/a.py', 'pkg/b.py', FALSE, FALSE, 1, 2)
        """
    )
    # Config values referencing modules
    con.execute(
        """
        INSERT INTO analytics.config_values (
            config_path, format, key, reference_paths, reference_modules, reference_count
        ) VALUES
            ('cfg/app.yaml', 'yaml', 'feature.flag', '[]', '["pkg.a","pkg.b"]', 2)
        """
    )

    compute_graph_stats(gateway, repo=REPO, commit=COMMIT)
    rows = con.execute(
        "SELECT graph_name FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [REPO, COMMIT],
    ).fetchall()
    names = {row[0] for row in rows}
    expected = {
        "symbol_module_graph",
        "symbol_function_graph",
        "config_key_projection",
        "config_module_projection",
    }
    if not expected.issubset(names):
        pytest.fail(f"Missing expected graphs: {expected - names}")


def test_subsystem_agreement_summary_aggregates() -> None:
    """Validate subsystem agreement summary aggregates disagreement counts."""
    gateway = _gateway()
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
    disagree_row = con.execute(
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


def test_validation_flags_large_symbol_community_and_config_hubs() -> None:
    """Surface validation warnings for oversized symbol communities and config hubs."""
    gateway = _gateway()
    con = gateway.con
    _seed_modules(gateway)
    # Seed symbol metrics table with a large community id
    con.executemany(
        """
        INSERT INTO analytics.symbol_graph_metrics_modules (
            repo, commit, module, symbol_betweenness, symbol_closeness,
            symbol_eigenvector, symbol_harmonic, symbol_k_core, symbol_constraint,
            symbol_effective_size, symbol_community_id, symbol_component_id,
            symbol_component_size, created_at
        ) VALUES (?, ?, ?, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 99, 0, 1, ?)
        """,
        [
            (REPO, COMMIT, "pkg.a", datetime.now(UTC)),
            (REPO, COMMIT, "pkg.b", datetime.now(UTC)),
            (REPO, COMMIT, "pkg.c", datetime.now(UTC)),
        ],
    )
    # Config hubs: key referenced by many modules
    con.execute(
        """
        INSERT INTO analytics.config_values (
            config_path, format, key, reference_paths, reference_modules, reference_count
        ) VALUES ('cfg/app.yaml', 'yaml', 'wide.key', '[]', '["pkg.a","pkg.b","pkg.c"]', 3)
        """
    )

    findings = warn_graph_structure(gateway, REPO, COMMIT, log=None)
    check_names = {f["check_name"] for f in findings}
    if "symbol_graph_large_community" not in check_names:
        pytest.fail("Expected symbol_graph_large_community finding")
    if "config_keys_broad_usage" not in check_names:
        pytest.fail("Expected config_keys_broad_usage finding")
