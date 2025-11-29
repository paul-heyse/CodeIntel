"""Ensure performance indexes exist for docs view sources."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.storage.schemas import apply_all_schemas


def _list_indexes(con: duckdb.DuckDBPyConnection, *, schema: str, table: str) -> set[str]:
    rows = con.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE schema_name = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def test_test_profile_has_primary_subsystem_index() -> None:
    """analytics.test_profile should be indexed for primary_subsystem_id scans."""
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    index_names = _list_indexes(con, schema="analytics", table="test_profile")
    expected = "idx_analytics_test_profile_primary_subsystem"
    if expected not in index_names:
        pytest.fail(f"Missing index {expected} on analytics.test_profile")


def test_subsystems_has_repo_commit_index() -> None:
    """analytics.subsystems should be indexed for repo/commit/subsystem lookups."""
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    index_names = _list_indexes(con, schema="analytics", table="subsystems")
    expected = "idx_analytics_subsystems_repo_commit_id"
    if expected not in index_names:
        pytest.fail(f"Missing index {expected} on analytics.subsystems")
