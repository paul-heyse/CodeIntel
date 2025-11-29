"""Validate subsystem docs view schemas align with contracts."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.views import create_all_views


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _fresh_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    create_all_views(con)
    return con


def _seed_subsystem(
    con: duckdb.DuckDBPyConnection,
    *,
    overrides: dict[str, object] | None = None,
) -> None:
    base = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "subsystem_id": "subsysdemo",
        "name": "Subsystem Demo",
        "description": "demo subsystem",
        "module_count": 1,
        "function_count": 1,
    }
    if overrides:
        base.update(overrides)
    con.execute(
        """
        INSERT INTO analytics.subsystems (
            repo, commit, subsystem_id, name, description,
            module_count, modules_json, entrypoints_json,
            internal_edge_count, external_edge_count, fan_in, fan_out,
            function_count, avg_risk_score, max_risk_score,
            high_risk_function_count, risk_level, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            base["repo"],
            base["commit"],
            base["subsystem_id"],
            base["name"],
            base["description"],
            base["module_count"],
            0,
            0,
            0,
            0,
            base["function_count"],
            0.1,
            0.2,
            0,
            "low",
        ],
    )


def test_subsystem_profile_columns() -> None:
    """Subsystem profile view exposes expected columns for typed contracts."""
    con = _fresh_connection()
    rel_df = con.execute("SELECT * FROM docs.v_subsystem_profile LIMIT 0").fetchdf()
    cols = [c.lower() for c in rel_df.columns]
    expected = {
        "repo",
        "commit",
        "subsystem_id",
        "name",
        "description",
        "module_count",
        "modules_json",
        "entrypoints_json",
        "internal_edge_count",
        "external_edge_count",
        "fan_in",
        "fan_out",
        "function_count",
        "avg_risk_score",
        "max_risk_score",
        "high_risk_function_count",
        "risk_level",
        "import_in_degree",
        "import_out_degree",
        "import_pagerank",
        "import_betweenness",
        "import_closeness",
        "import_layer",
        "created_at",
    }
    missing = expected - set(cols)
    _require(
        condition=not missing,
        message=f"Missing columns in v_subsystem_profile: {sorted(missing)}",
    )


def test_subsystem_coverage_columns() -> None:
    """Subsystem coverage view exposes expected columns for typed contracts."""
    con = _fresh_connection()
    rel_df = con.execute("SELECT * FROM docs.v_subsystem_coverage LIMIT 0").fetchdf()
    cols = [c.lower() for c in rel_df.columns]
    expected = {
        "repo",
        "commit",
        "subsystem_id",
        "name",
        "description",
        "module_count",
        "function_count",
        "risk_level",
        "avg_risk_score",
        "max_risk_score",
        "test_count",
        "passed_test_count",
        "failed_test_count",
        "skipped_test_count",
        "xfail_test_count",
        "flaky_test_count",
        "total_functions_covered",
        "avg_functions_covered",
        "max_functions_covered",
        "min_functions_covered",
        "function_coverage_ratio",
        "created_at",
    }
    missing = expected - set(cols)
    _require(
        condition=not missing,
        message=f"Missing columns in v_subsystem_coverage: {sorted(missing)}",
    )


def test_subsystem_profile_view_prefers_cache() -> None:
    """Cached subsystem profile rows should override computed values."""
    con = _fresh_connection()
    _seed_subsystem(con, overrides={"module_count": 1, "function_count": 2})
    expected_module_count = 42
    expected_function_count = 4
    con.execute(
        """
        INSERT INTO analytics.subsystem_profile_cache (
            repo, commit, subsystem_id, name, description, module_count,
            modules_json, entrypoints_json, internal_edge_count,
            external_edge_count, fan_in, fan_out, function_count,
            avg_risk_score, max_risk_score, high_risk_function_count,
            risk_level, import_in_degree, import_out_degree, import_pagerank,
            import_betweenness, import_closeness, import_layer, created_at
        )
        VALUES (
            'demo/repo', 'deadbeef', 'subsysdemo', 'Cached Name', 'cached',
            ?, '[]', '[]', 1, 1, 2, 3, ?, 0.5, 0.9, 7, 'medium',
            0.1, 0.2, 0.3, 0.4, 0.5, 2, CURRENT_TIMESTAMP
        )
        """,
        [expected_module_count, expected_function_count],
    )
    row = con.execute(
        """
        SELECT name, module_count, function_count, risk_level
        FROM docs.v_subsystem_profile
        WHERE subsystem_id = 'subsysdemo'
        """
    ).fetchone()
    if row is None:
        pytest.fail("No subsystem profile row returned")
        return
    name, module_count, function_count, risk_level = row
    _require(condition=name == "Cached Name", message="Expected cached name to be used")
    _require(
        condition=module_count == expected_module_count,
        message="Expected cached module_count to be used",
    )
    _require(
        condition=function_count == expected_function_count,
        message="Expected cached function_count to be used",
    )
    _require(condition=risk_level == "medium", message="Expected cached risk_level to be used")


def test_subsystem_coverage_view_prefers_cache() -> None:
    """Cached subsystem coverage rows should override computed values."""
    con = _fresh_connection()
    _seed_subsystem(con, overrides={"module_count": 1, "function_count": 2})
    expected_test_count = 99
    expected_functions_covered = 50
    con.execute(
        """
        INSERT INTO analytics.subsystem_coverage_cache (
            repo, commit, subsystem_id, name, description, module_count,
            function_count, risk_level, avg_risk_score, max_risk_score,
            test_count, passed_test_count, failed_test_count,
            skipped_test_count, xfail_test_count, flaky_test_count,
            total_functions_covered, avg_functions_covered,
            max_functions_covered, min_functions_covered,
            function_coverage_ratio, created_at
        )
        VALUES (
            'demo/repo', 'deadbeef', 'subsysdemo', 'Cached Name', 'cached',
            3, 10, 'high', 0.7, 0.9, ?, 90, 9, 0, 0, 5,
            ?, 5.0, 10.0, 1.0, 0.5, CURRENT_TIMESTAMP
        )
        """,
        [expected_test_count, expected_functions_covered],
    )
    row = con.execute(
        """
        SELECT test_count, total_functions_covered, risk_level
        FROM docs.v_subsystem_coverage
        WHERE subsystem_id = 'subsysdemo'
        """
    ).fetchone()
    if row is None:
        pytest.fail("No subsystem coverage row returned")
        return
    test_count, total_functions_covered, risk_level = row
    _require(
        condition=test_count == expected_test_count,
        message="Expected cached test_count to be used",
    )
    _require(
        condition=total_functions_covered == expected_functions_covered,
        message="Expected cached total_functions_covered to be used",
    )
    _require(condition=risk_level == "high", message="Expected cached risk_level to be used")
