"""Subsystem inference tests covering clustering and risk aggregation."""

from __future__ import annotations

import json

import duckdb
import pytest

from codeintel.analytics.subsystems import build_subsystems
from codeintel.config.models import SubsystemsConfig, SubsystemsOverrides
from codeintel.storage.gateway import StorageGateway, open_memory_gateway

REPO = "demo/repo"
COMMIT = "abc123"
EXPECTED_SUBSYSTEMS = 2
EXPECTED_MEMBERSHIPS = 3
TARGET_CLUSTER_SIZE = 2
EXPECTED_HIGH_RISK_COUNT = 1


def _setup_db() -> StorageGateway:
    """
    Create an in-memory DuckDB with schemas applied.

    Returns
    -------
    StorageGateway
        Connected in-memory database ready for writes.
    """
    return open_memory_gateway(apply_schema=True)


def _seed_modules(con: duckdb.DuckDBPyConnection) -> None:
    """Insert sample modules, edges, and risk data to drive clustering."""
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            ('pkg.api', 'pkg/api.py', ?, ?, 'python', '["api"]', '[]'),
            ('pkg.core', 'pkg/core.py', ?, ?, 'python', '["api"]', '[]'),
            ('pkg.misc', 'pkg/misc.py', ?, ?, 'python', '[]', '[]')
        """,
        [REPO, COMMIT, REPO, COMMIT, REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO graph.import_graph_edges (
            repo, commit, src_module, dst_module, src_fan_out, dst_fan_in, cycle_group
        )
        VALUES
            (?, ?, 'pkg.api', 'pkg.core', 1, 1, 0),
            (?, ?, 'pkg.core', 'pkg.api', 1, 1, 0)
        """,
        [REPO, COMMIT, REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO graph.symbol_use_edges (symbol, def_path, use_path, same_file, same_module)
        VALUES
            ('sym_core', 'pkg/core.py', 'pkg/api.py', FALSE, FALSE)
        """
    )
    con.execute(
        """
        INSERT INTO analytics.config_values (
            config_path, format, key, reference_paths, reference_modules, reference_count
        )
        VALUES (
            'cfg/app.yaml', 'yaml', 'feature.flag', '[]', '["pkg.api", "pkg.core"]', 2
        )
        """
    )
    con.execute(
        """
        INSERT INTO analytics.function_metrics (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, loc, logical_loc, param_count, positional_params,
            keyword_only_params, has_varargs, has_varkw, is_async, is_generator,
            return_count, yield_count, raise_count, cyclomatic_complexity, max_nesting_depth,
            stmt_count, decorator_count, has_docstring, complexity_bucket, created_at
        ) VALUES
            (10, 'goid:demo/repo#python:function:pkg.api.handler', ?, ?, 'pkg/api.py', 'python',
             'function', 'pkg.api.handler', 1, 2, 4, 3, 1, 1, 0, FALSE, FALSE, FALSE, FALSE, 1, 0,
             0, 1, 1, 2, 0, TRUE, 'low', NOW()),
            (11, 'goid:demo/repo#python:function:pkg.core.service', ?, ?, 'pkg/core.py', 'python',
             'function', 'pkg.core.service', 1, 2, 4, 3, 1, 1, 0, FALSE, FALSE, FALSE, FALSE, 1,
             0, 0, 1, 1, 2, 0, TRUE, 'low', NOW())
        """,
        [REPO, COMMIT, REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO analytics.goid_risk_factors (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname, loc,
            logical_loc, cyclomatic_complexity, complexity_bucket, typedness_bucket,
            typedness_source, hotspot_score, file_typed_ratio, static_error_count,
            has_static_errors, executable_lines, covered_lines, coverage_ratio, tested,
            test_count, failing_test_count, last_test_status, risk_score, risk_level, tags,
            owners, created_at
        ) VALUES
            (10, 'goid:demo/repo#python:function:pkg.api.handler', ?, ?, 'pkg/api.py', 'python',
             'function', 'pkg.api.handler', 4, 3, 1, 'low', 'typed', 'analysis', 0.0, 1.0, 0,
             FALSE, 4, 2, 0.5, TRUE, 1, 0, 'all_passing', 0.2, 'low', '[]', '[]', NOW()),
            (11, 'goid:demo/repo#python:function:pkg.core.service', ?, ?, 'pkg/core.py', 'python',
             'function', 'pkg.core.service', 4, 3, 1, 'low', 'typed', 'analysis', 0.0, 1.0, 0,
             FALSE, 4, 2, 0.5, TRUE, 1, 0, 'all_passing', 0.8, 'high', '[]', '[]', NOW())
        """,
        [REPO, COMMIT, REPO, COMMIT],
    )


def test_subsystems_cluster_and_risk_aggregation() -> None:
    """Clusters modules and aggregates risk across subsystems."""
    gateway = _setup_db()
    con = gateway.con
    _seed_modules(con)

    cfg = SubsystemsConfig.from_paths(
        repo=REPO,
        commit=COMMIT,
        overrides=SubsystemsOverrides(max_subsystems=2, min_modules=1),
    )
    build_subsystems(gateway, cfg)

    subsystems = con.execute(
        """
        SELECT subsystem_id, modules_json, risk_level, high_risk_function_count
        FROM analytics.subsystems
        """
    ).fetchall()
    if len(subsystems) != EXPECTED_SUBSYSTEMS:
        pytest.fail(f"Expected {EXPECTED_SUBSYSTEMS} subsystems, found {len(subsystems)}")

    by_size = {len(json.loads(mods)): (mods, risk, high) for _, mods, risk, high in subsystems}
    large_modules, large_risk, high_count = by_size[TARGET_CLUSTER_SIZE]
    if "pkg.api" not in large_modules or "pkg.core" not in large_modules:
        pytest.fail(f"Subsystem missing expected modules: {large_modules}")
    if large_risk != "high":
        pytest.fail(f"Expected high risk for core cluster, got {large_risk}")
    if high_count != EXPECTED_HIGH_RISK_COUNT:
        pytest.fail(f"Expected one high-risk function, got {high_count}")

    memberships = con.execute(
        "SELECT subsystem_id, module FROM analytics.subsystem_modules"
    ).fetchall()
    if len(memberships) != EXPECTED_MEMBERSHIPS:
        pytest.fail(f"Expected {EXPECTED_MEMBERSHIPS} memberships, got {len(memberships)}")
    members = {row[1] for row in memberships}
    if members != {"pkg.api", "pkg.core", "pkg.misc"}:
        pytest.fail(f"Unexpected subsystem membership: {members}")
