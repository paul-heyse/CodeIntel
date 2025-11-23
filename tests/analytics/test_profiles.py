"""Unit tests for analytics.profiles aggregations."""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest

from codeintel.analytics.profiles import (
    SLOW_TEST_THRESHOLD_MS,
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.config.models import ProfilesAnalyticsConfig
from codeintel.storage.schemas import apply_all_schemas

EPSILON = 1e-6
REPO = "demo/repo"
COMMIT = "abc123"
REL_PATH = "pkg/mod.py"
MODULE = "pkg.mod"


def _seed_profile_fixtures(con: duckdb.DuckDBPyConnection) -> None:
    now = datetime.now(tz=UTC)
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '["server"]', '["team@example.com"]')
        """,
        [MODULE, REL_PATH, REPO, COMMIT],
    )
    con.execute(
        """
        INSERT INTO core.ast_metrics (rel_path, node_count, function_count, class_count,
                                      avg_depth, max_depth, complexity, generated_at)
        VALUES (?, 10, 1, 0, 1.0, 1, 2.0, ?)
        """,
        [REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO analytics.hotspots (rel_path, commit_count, author_count, lines_added,
                                        lines_deleted, complexity, score)
        VALUES (?, 1, 1, 5, 1, 2.0, 0.5)
        """,
        [REL_PATH],
    )
    con.execute(
        """
        INSERT INTO analytics.typedness (path, type_error_count, annotation_ratio,
                                         untyped_defs, overlay_needed)
        VALUES (?, 1, '{"params": 0.5}', 0, FALSE)
        """,
        [REL_PATH],
    )
    con.execute(
        """
        INSERT INTO analytics.static_diagnostics (rel_path, pyrefly_errors, pyright_errors,
                                                  ruff_errors, total_errors, has_errors)
        VALUES (?, 1, 0, 0, 1, TRUE)
        """,
        [REL_PATH],
    )
    con.execute(
        """
        INSERT INTO core.docstrings (
            repo, commit, rel_path, module, qualname, kind, lineno, end_lineno, raw_docstring,
            style, short_desc, long_desc, params, returns, raises, examples, created_at
        )
        VALUES (?, ?, ?, ?, 'pkg.mod.func', 'function', 1, 2, 'Doc', 'auto',
                'Short doc', 'Longer doc', '[]', '{"return": "int"}', '[]', '[]', ?)
        """,
        [REPO, COMMIT, REL_PATH, MODULE, now],
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
        )
        VALUES (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, ?, 'python',
                'function', 'pkg.mod.func', 4, 3, 2, 'medium', 'typed', 'analysis',
                0.5, 0.5, 1, TRUE, 4, 2, 0.5, TRUE, 1, 1, 'some_failing', 0.9, 'high',
                '["server"]', '["team@example.com"]', ?)
        """,
        [REPO, COMMIT, REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO analytics.function_metrics (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, loc, logical_loc, param_count, positional_params,
            keyword_only_params, has_varargs, has_varkw, is_async, is_generator,
            return_count, yield_count, raise_count, cyclomatic_complexity, max_nesting_depth,
            stmt_count, decorator_count, has_docstring, complexity_bucket, created_at
        )
        VALUES (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, ?, 'python', 'function',
                'pkg.mod.func', 1, 2, 4, 3, 2, 1, 1, TRUE, FALSE, FALSE, FALSE, 1, 0, 0, 2, 1,
                2, 0, TRUE, 'medium', ?)
        """,
        [REPO, COMMIT, REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO analytics.function_types (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, total_params, annotated_params, unannotated_params,
            param_typed_ratio, has_return_annotation, return_type, return_type_source,
            type_comment, param_types, fully_typed, partial_typed, untyped, typedness_bucket,
            typedness_source, created_at
        )
        VALUES (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, ?, 'python', 'function',
                'pkg.mod.func', 1, 2, 2, 2, 0, 1.0, TRUE, 'int', 'annotation', NULL, '[]', TRUE,
                FALSE, FALSE, 'typed', 'analysis', ?)
        """,
        [REPO, COMMIT, REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO analytics.coverage_functions (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, executable_lines, covered_lines, coverage_ratio, tested,
            untested_reason, created_at
        )
        VALUES (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, ?, 'python', 'function',
                'pkg.mod.func', 1, 2, 4, 2, 0.5, TRUE, '', ?)
        """,
        [REPO, COMMIT, REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog (
            test_id, test_goid_h128, urn, repo, commit, rel_path, qualname, kind, status,
            duration_ms, markers, parametrized, flaky, created_at
        )
        VALUES ('pkg/mod.py::test_func', 2, 'goid:demo/repo#python:function:pkg.mod.test_func',
                ?, ?, ?, 'pkg.mod.test_func', 'function', 'failed', 1500, '[]', FALSE, TRUE, ?)
        """,
        [REPO, COMMIT, REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO analytics.test_coverage_edges (
            test_id, test_goid_h128, function_goid_h128, urn, repo, commit, rel_path, qualname,
            covered_lines, executable_lines, coverage_ratio, last_status, created_at
        )
        VALUES ('pkg/mod.py::test_func', 2, 1, 'goid:demo/repo#python:function:pkg.mod.func',
                ?, ?, ?, 'pkg.mod.func', 2, 4, 0.5, 'failed', ?)
        """,
        [REPO, COMMIT, REL_PATH, now],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            caller_goid_h128, callee_goid_h128, callsite_path, callsite_line, callsite_col,
            language, kind, resolved_via, confidence, evidence_json
        )
        VALUES
            (1, 2, ?, 1, 1, 'python', 'direct', 'local_name', 1.0, '{}'),
            (3, 1, ?, 2, 2, 'python', 'direct', 'global_name', 1.0, '{}')
        """,
        [REL_PATH, REL_PATH],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_nodes (goid_h128, language, kind, arity, is_public, rel_path)
        VALUES
            (1, 'python', 'function', 0, TRUE, ?),
            (2, 'python', 'function', 0, FALSE, ?),
            (3, 'python', 'function', 0, FALSE, ?)
        """,
        [REL_PATH, REL_PATH, REL_PATH],
    )
    con.execute(
        """
        INSERT INTO graph.import_graph_edges (src_module, dst_module, src_fan_out, dst_fan_in, cycle_group)
        VALUES (?, ?, 1, 1, 1)
        """,
        [MODULE, MODULE],
    )


def _assert_function_profile(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        """
        SELECT tests_touching, failing_tests, slow_tests, call_fan_in, call_fan_out,
               risk_component_coverage, doc_short, slow_test_threshold_ms
        FROM analytics.function_profile
        WHERE function_goid_h128 = 1
        """
    ).fetchone()
    if row is None:
        pytest.fail("function_profile row missing")
    if row[0] != 1:
        pytest.fail(f"Expected tests_touching=1, got {row[0]}")
    if row[1] != 1:
        pytest.fail(f"Expected failing_tests=1, got {row[1]}")
    if row[2] != 1:
        pytest.fail(f"Expected slow_tests=1, got {row[2]}")
    if row[3] != 1 or row[4] != 1:
        pytest.fail(f"Unexpected call fan-in/out: {row[3]}, {row[4]}")
    if abs(row[5] - 0.2) > EPSILON:
        pytest.fail(f"Unexpected coverage component {row[5]}")
    if row[6] != "Short doc":
        pytest.fail(f"Doc summary mismatch: {row[6]}")
    if row[7] != SLOW_TEST_THRESHOLD_MS:
        pytest.fail(f"Slow threshold mismatch: {row[7]}")


def _assert_file_profile(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        """
        SELECT file_coverage_ratio, high_risk_function_count, module
        FROM analytics.file_profile
        WHERE rel_path = ?
        """,
        [REL_PATH],
    ).fetchone()
    if row is None:
        pytest.fail("file_profile row missing")
    if abs(row[0] - 0.5) > EPSILON:
        pytest.fail(f"Unexpected file coverage ratio {row[0]}")
    if row[1] != 1:
        pytest.fail(f"Expected 1 high-risk function, got {row[1]}")
    if row[2] != MODULE:
        pytest.fail(f"Module mismatch: {row[2]}")


def _assert_module_profile(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        """
        SELECT module_coverage_ratio, import_fan_in, import_fan_out, in_cycle
        FROM analytics.module_profile
        WHERE module = ?
        """,
        [MODULE],
    ).fetchone()
    if row is None:
        pytest.fail("module_profile row missing")
    if abs(row[0] - 1.0) > EPSILON:
        pytest.fail(f"Unexpected module coverage ratio {row[0]}")
    if row[1] != 1 or row[2] != 1:
        pytest.fail(f"Import fan-in/out mismatch: {row[1]}, {row[2]}")
    if row[3] is not True:
        pytest.fail("Expected module to be marked as in_cycle")


def test_profile_builders_aggregate_expected_fields() -> None:
    """Ensure profile builders compose metrics, tests, coverage, and graph data."""
    con = duckdb.connect(":memory:")
    apply_all_schemas(con)
    _seed_profile_fixtures(con)
    cfg = ProfilesAnalyticsConfig(repo=REPO, commit=COMMIT)
    build_function_profile(con, cfg)
    build_file_profile(con, cfg)
    build_module_profile(con, cfg)
    _assert_function_profile(con)
    _assert_file_profile(con)
    _assert_module_profile(con)
