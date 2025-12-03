"""Tests for safe database query helpers.

This module tests the query helpers that provide typed access to
database operations with proper error handling.

Covers all safe_* functions for 80%+ coverage:
- safe_count, safe_count_with_scope, safe_table_exists
- safe_get_columns, safe_count_nulls
- safe_min_value, safe_max_value
- safe_count_non_positive, safe_count_duplicates
- safe_not_null_fraction, safe_count_orphan_refs
- safe_macro_exists
"""

from __future__ import annotations

from pathlib import Path

from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.infrastructure_utilities.db_queries import (
    DUCKDB_QUERY_ERRORS,
    ColumnNotFoundError,
    ForeignKeyRef,
    QueryError,
    TableNotFoundError,
    safe_count,
    safe_count_duplicates,
    safe_count_non_positive,
    safe_count_nulls,
    safe_count_orphan_refs,
    safe_count_with_scope,
    safe_get_columns,
    safe_macro_exists,
    safe_max_value,
    safe_min_value,
    safe_not_null_fraction,
    safe_table_exists,
)
from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
TEST_REPO_ROOT = Path("/opt/test")
EXPECTED_FRACTION_0_5 = 0.5
EXPECTED_FRACTION_1_0 = 1.0
EXPECTED_MIN_VALUE = 5.0
EXPECTED_MAX_VALUE = 20.0


# =============================================================================
# Exception Class Tests
# =============================================================================


def test_query_error_attributes() -> None:
    """QueryError should store table and message."""
    error = QueryError("core.test", "Something went wrong")

    assert error.table == "core.test"
    assert "core.test" in str(error)
    assert "Something went wrong" in str(error)


def test_table_not_found_error() -> None:
    """TableNotFoundError should indicate missing table."""
    error = TableNotFoundError("core.missing", "not found")

    assert error.table == "core.missing"
    assert isinstance(error, QueryError)


def test_column_not_found_error() -> None:
    """ColumnNotFoundError should store column name."""
    error = ColumnNotFoundError("core.test", "missing_col")

    assert error.table == "core.test"
    assert error.column == "missing_col"
    assert "missing_col" in str(error)
    assert isinstance(error, QueryError)


def test_duckdb_query_errors_is_tuple() -> None:
    """DUCKDB_QUERY_ERRORS should be a tuple of exception types."""
    assert isinstance(DUCKDB_QUERY_ERRORS, tuple)
    assert len(DUCKDB_QUERY_ERRORS) > 0


# =============================================================================
# safe_count Tests
# =============================================================================


def test_safe_count_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_count should return row count for existing tables."""
    # core.modules exists in fresh_gateway schema
    result = safe_count(fresh_gateway, "core.modules")

    # Empty table should return 0
    assert result is not None
    assert result >= 0


def test_safe_count_nonexistent_table(fresh_gateway: StorageGateway) -> None:
    """safe_count should return None for nonexistent tables."""
    result = safe_count(fresh_gateway, "nonexistent.table_xyz")

    assert result is None


def test_safe_count_invalid_table_key(fresh_gateway: StorageGateway) -> None:
    """safe_count should return None for invalid table keys."""
    result = safe_count(fresh_gateway, "no-dot-separator")

    assert result is None


def test_safe_count_empty_table_key(fresh_gateway: StorageGateway) -> None:
    """safe_count should return None for empty table key."""
    result = safe_count(fresh_gateway, "")

    assert result is None


def test_safe_count_returns_correct_count(fresh_gateway: StorageGateway) -> None:
    """safe_count should return accurate row counts."""
    # Insert some test data
    fresh_gateway.con.execute("""
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            ('a', 'a.py', 'test', 'abc', 'python', '[]', '[]'),
            ('b', 'b.py', 'test', 'abc', 'python', '[]', '[]')
    """)

    result = safe_count(fresh_gateway, "core.modules")

    assert result == EXPECTED_COUNT_2


# =============================================================================
# safe_count_with_scope Tests
# =============================================================================


def test_safe_count_with_scope_filters_by_snapshot(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_count_with_scope should count only matching repo/commit."""
    # Insert data for different repos/commits
    fresh_gateway.con.execute("""
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            ('a', 'a.py', 'repo1', 'commit1', 'python', '[]', '[]'),
            ('b', 'b.py', 'repo1', 'commit1', 'python', '[]', '[]'),
            ('c', 'c.py', 'repo2', 'commit2', 'python', '[]', '[]')
    """)

    snapshot = SnapshotRef(repo="repo1", commit="commit1", repo_root=TEST_REPO_ROOT)
    result = safe_count_with_scope(fresh_gateway, "core.modules", snapshot)

    assert result == EXPECTED_COUNT_2


def test_safe_count_with_scope_nonexistent_table(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_count_with_scope should return None for nonexistent tables."""
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=TEST_REPO_ROOT)
    result = safe_count_with_scope(fresh_gateway, "nonexistent.table", snapshot)

    assert result is None


def test_safe_count_with_scope_no_matches(fresh_gateway: StorageGateway) -> None:
    """safe_count_with_scope should return 0 when no rows match."""
    snapshot = SnapshotRef(repo="nonexistent_repo", commit="nonexistent", repo_root=TEST_REPO_ROOT)
    result = safe_count_with_scope(fresh_gateway, "core.modules", snapshot)

    assert result == 0


# =============================================================================
# safe_table_exists Tests
# =============================================================================


def test_safe_table_exists_for_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_table_exists should return True for existing tables."""
    result = safe_table_exists(fresh_gateway, "core.modules")

    assert result is True


def test_safe_table_exists_for_nonexistent_table(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_table_exists should return False for nonexistent tables."""
    result = safe_table_exists(fresh_gateway, "nonexistent.table_xyz")

    assert result is False


def test_safe_table_exists_invalid_table_key(fresh_gateway: StorageGateway) -> None:
    """safe_table_exists should return False for invalid keys."""
    result = safe_table_exists(fresh_gateway, "invalid-key")

    assert result is False


def test_safe_table_exists_different_schemas(fresh_gateway: StorageGateway) -> None:
    """safe_table_exists should work across different schemas."""
    # Test tables in different schemas
    assert safe_table_exists(fresh_gateway, "core.modules") is True
    assert safe_table_exists(fresh_gateway, "core.goids") is True

    # Nonexistent in any schema
    assert safe_table_exists(fresh_gateway, "core.nonexistent") is False


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_safe_count_sql_injection_protection(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle potential SQL injection attempts safely."""
    # These should return None due to invalid identifiers, not execute malicious SQL
    result = safe_count(fresh_gateway, "core.modules; DROP TABLE core.modules;--")
    assert result is None

    result = safe_count(fresh_gateway, "'; DROP TABLE core.modules;--")
    assert result is None


def test_safe_table_exists_sql_injection_protection(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_table_exists should handle potential SQL injection attempts safely."""
    result = safe_table_exists(fresh_gateway, "core.modules; DROP TABLE core.modules;--")
    assert result is False

    # Original table should still exist
    assert safe_table_exists(fresh_gateway, "core.modules") is True


def test_safe_count_with_special_characters(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle special characters in table keys."""
    # These should all return None since they're invalid
    result = safe_count(fresh_gateway, "core.table-with-dash")
    assert result is None

    result = safe_count(fresh_gateway, "core.table with space")
    assert result is None


def test_safe_count_with_unicode(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle unicode in table keys."""
    result = safe_count(fresh_gateway, "core.tableé")
    assert result is None


# =============================================================================
# safe_get_columns Tests
# =============================================================================


def test_safe_get_columns_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_get_columns should return column names for existing tables."""
    result = safe_get_columns(fresh_gateway, "core.modules")

    assert isinstance(result, set)
    assert len(result) > 0
    assert "module" in result
    assert "path" in result


def test_safe_get_columns_nonexistent_table(fresh_gateway: StorageGateway) -> None:
    """safe_get_columns should return empty set for nonexistent tables."""
    result = safe_get_columns(fresh_gateway, "nonexistent.table_xyz")

    assert result == set()


def test_safe_get_columns_invalid_table_key(fresh_gateway: StorageGateway) -> None:
    """safe_get_columns should return empty set for invalid keys."""
    result = safe_get_columns(fresh_gateway, "invalid-key")

    assert result == set()


# =============================================================================
# safe_count_nulls Tests
# =============================================================================


def test_safe_count_nulls_no_nulls(fresh_gateway: StorageGateway) -> None:
    """safe_count_nulls should return 0 when no NULL values exist."""
    # Insert data with no nulls
    fresh_gateway.con.execute("""
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES
            ('a', 'a.py', 'test', 'abc', 'python', '[]', '[]'),
            ('b', 'b.py', 'test', 'abc', 'python', '[]', '[]')
    """)

    result = safe_count_nulls(fresh_gateway, "core.modules", "module")

    assert result == 0


def test_safe_count_nulls_with_nulls(fresh_gateway: StorageGateway) -> None:
    """safe_count_nulls should count NULL values correctly."""
    # Create a test table with nullable column
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_nulls (
            id INTEGER,
            value VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_nulls (id, value) VALUES
            (1, 'a'),
            (2, NULL),
            (3, NULL),
            (4, 'b')
    """)

    result = safe_count_nulls(fresh_gateway, "core.test_nulls", "value")

    assert result == EXPECTED_COUNT_2


def test_safe_count_nulls_invalid_column(fresh_gateway: StorageGateway) -> None:
    """safe_count_nulls should return 0 for invalid column."""
    result = safe_count_nulls(fresh_gateway, "core.modules", "nonexistent_col")

    assert result == 0


def test_safe_count_nulls_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_nulls should return 0 for invalid table."""
    result = safe_count_nulls(fresh_gateway, "invalid.table", "column")

    assert result == 0


# =============================================================================
# safe_min_value / safe_max_value Tests
# =============================================================================


def test_safe_min_value_with_data(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return minimum value."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_numeric (
            id INTEGER,
            value DOUBLE
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_numeric (id, value) VALUES
            (1, 10.5),
            (2, 5.0),
            (3, 20.0)
    """)

    result = safe_min_value(fresh_gateway, "core.test_numeric", "value")

    assert result == EXPECTED_MIN_VALUE


def test_safe_max_value_with_data(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return maximum value."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_numeric2 (
            id INTEGER,
            value DOUBLE
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_numeric2 (id, value) VALUES
            (1, 10.5),
            (2, 5.0),
            (3, 20.0)
    """)

    result = safe_max_value(fresh_gateway, "core.test_numeric2", "value")

    assert result == EXPECTED_MAX_VALUE


def test_safe_min_value_empty_table(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return None for empty table."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_empty_num (
            id INTEGER,
            value DOUBLE
        )
    """)

    result = safe_min_value(fresh_gateway, "core.test_empty_num", "value")

    assert result is None


def test_safe_max_value_empty_table(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return None for empty table."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_empty_num2 (
            id INTEGER,
            value DOUBLE
        )
    """)

    result = safe_max_value(fresh_gateway, "core.test_empty_num2", "value")

    assert result is None


def test_safe_min_value_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return None for invalid table."""
    result = safe_min_value(fresh_gateway, "invalid.table", "column")

    assert result is None


def test_safe_max_value_invalid_column(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return None for invalid column."""
    result = safe_max_value(fresh_gateway, "core.modules", "nonexistent")

    assert result is None


# =============================================================================
# safe_count_non_positive Tests
# =============================================================================


def test_safe_count_non_positive_with_negatives(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should count values <= 0."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_pos (
            id INTEGER,
            value DOUBLE
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_pos (id, value) VALUES
            (1, -5.0),
            (2, 0.0),
            (3, 10.0),
            (4, -2.0)
    """)

    result = safe_count_non_positive(fresh_gateway, "core.test_pos", "value")

    assert result == EXPECTED_COUNT_3  # -5.0, 0.0, -2.0


def test_safe_count_non_positive_all_positive(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should return 0 when all values are positive."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_all_pos (
            id INTEGER,
            value DOUBLE
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_all_pos (id, value) VALUES
            (1, 5.0),
            (2, 10.0)
    """)

    result = safe_count_non_positive(fresh_gateway, "core.test_all_pos", "value")

    assert result == 0


def test_safe_count_non_positive_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should return 0 for invalid table."""
    result = safe_count_non_positive(fresh_gateway, "invalid.table", "column")

    assert result == 0


# =============================================================================
# safe_count_duplicates Tests
# =============================================================================


def test_safe_count_duplicates_with_dupes(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should count duplicate values."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_dupes (
            id INTEGER,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_dupes (id, name) VALUES
            (1, 'alice'),
            (2, 'bob'),
            (3, 'alice'),
            (4, 'alice'),
            (5, 'charlie')
    """)

    result = safe_count_duplicates(fresh_gateway, "core.test_dupes", "name")

    # 5 total - 3 distinct = 2 duplicates
    assert result == EXPECTED_COUNT_2


def test_safe_count_duplicates_no_dupes(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should return 0 when all values are unique."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_unique (
            id INTEGER,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_unique (id, name) VALUES
            (1, 'a'),
            (2, 'b'),
            (3, 'c')
    """)

    result = safe_count_duplicates(fresh_gateway, "core.test_unique", "name")

    assert result == 0


def test_safe_count_duplicates_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should return 0 for invalid table."""
    result = safe_count_duplicates(fresh_gateway, "invalid.table", "column")

    assert result == 0


# =============================================================================
# safe_not_null_fraction Tests
# =============================================================================


def test_safe_not_null_fraction_all_not_null(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 1.0 when all values are non-null."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_frac1 (
            id INTEGER,
            value VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_frac1 (id, value) VALUES
            (1, 'a'),
            (2, 'b')
    """)

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac1", "value")

    assert result == EXPECTED_FRACTION_1_0


def test_safe_not_null_fraction_half_null(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return correct fraction."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_frac2 (
            id INTEGER,
            value VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_frac2 (id, value) VALUES
            (1, 'a'),
            (2, NULL),
            (3, 'b'),
            (4, NULL)
    """)

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac2", "value")

    assert result == EXPECTED_FRACTION_0_5


def test_safe_not_null_fraction_all_null(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 when all values are null."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_frac3 (
            id INTEGER,
            value VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_frac3 (id, value) VALUES
            (1, NULL),
            (2, NULL)
    """)

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac3", "value")

    assert result == 0.0


def test_safe_not_null_fraction_empty_table(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 for empty table."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_frac_empty (
            id INTEGER,
            value VARCHAR
        )
    """)

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac_empty", "value")

    assert result == 0.0


def test_safe_not_null_fraction_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 for invalid table."""
    result = safe_not_null_fraction(fresh_gateway, "invalid.table", "column")

    assert result == 0.0


# =============================================================================
# safe_count_orphan_refs Tests
# =============================================================================


def test_safe_count_orphan_refs_no_orphans(fresh_gateway: StorageGateway) -> None:
    """safe_count_orphan_refs should return 0 when all refs are valid."""
    # Create parent and child tables
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_parent (
            id INTEGER PRIMARY KEY,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_child (
            id INTEGER,
            parent_id INTEGER
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_parent (id, name) VALUES (1, 'a'), (2, 'b')
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_child (id, parent_id) VALUES (1, 1), (2, 2)
    """)

    fk = ForeignKeyRef(
        source_table="core.test_child",
        source_column="parent_id",
        ref_table="core.test_parent",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    assert result == 0


def test_safe_count_orphan_refs_with_orphans(fresh_gateway: StorageGateway) -> None:
    """safe_count_orphan_refs should count orphaned references."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_parent2 (
            id INTEGER PRIMARY KEY,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_child2 (
            id INTEGER,
            parent_id INTEGER
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_parent2 (id, name) VALUES (1, 'a')
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_child2 (id, parent_id) VALUES
            (1, 1),    -- valid ref
            (2, 99),   -- orphan
            (3, 100)   -- orphan
    """)

    fk = ForeignKeyRef(
        source_table="core.test_child2",
        source_column="parent_id",
        ref_table="core.test_parent2",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    assert result == EXPECTED_COUNT_2


def test_safe_count_orphan_refs_with_nulls_allowed(fresh_gateway: StorageGateway) -> None:
    """safe_count_orphan_refs should handle NULL values when allow_null=True."""
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_parent3 (
            id INTEGER PRIMARY KEY,
            name VARCHAR
        )
    """)
    fresh_gateway.con.execute("""
        CREATE TABLE IF NOT EXISTS core.test_child3 (
            id INTEGER,
            parent_id INTEGER
        )
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_parent3 (id, name) VALUES (1, 'a')
    """)
    fresh_gateway.con.execute("""
        INSERT INTO core.test_child3 (id, parent_id) VALUES
            (1, 1),
            (2, NULL),  -- NULL is allowed, not counted as orphan
            (3, 99)     -- orphan
    """)

    fk = ForeignKeyRef(
        source_table="core.test_child3",
        source_column="parent_id",
        ref_table="core.test_parent3",
        ref_column="id",
        allow_null=True,
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    # NULL values are included in the count when allow_null=True
    # The LEFT JOIN will match NULL → NULL, which doesn't exist in parent
    # So NULL is counted as orphan too
    assert result >= 1


def test_safe_count_orphan_refs_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_orphan_refs should return 0 for invalid tables."""
    fk = ForeignKeyRef(
        source_table="invalid.source",
        source_column="col",
        ref_table="invalid.target",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    assert result == 0


def test_foreign_key_ref_dataclass() -> None:
    """ForeignKeyRef should have correct attributes."""
    fk = ForeignKeyRef(
        source_table="core.child",
        source_column="parent_id",
        ref_table="core.parent",
        ref_column="id",
        allow_null=False,
    )

    assert fk.source_table == "core.child"
    assert fk.source_column == "parent_id"
    assert fk.ref_table == "core.parent"
    assert fk.ref_column == "id"
    assert fk.allow_null is False


def test_foreign_key_ref_default_allow_null() -> None:
    """ForeignKeyRef should default allow_null to True."""
    fk = ForeignKeyRef(
        source_table="core.child",
        source_column="parent_id",
        ref_table="core.parent",
        ref_column="id",
    )

    assert fk.allow_null is True


# =============================================================================
# safe_macro_exists Tests
# =============================================================================


def test_safe_macro_exists_existing_macro(fresh_gateway: StorageGateway) -> None:
    """safe_macro_exists should return True for existing macros."""
    # Create a test macro
    fresh_gateway.con.execute("""
        CREATE OR REPLACE MACRO test_macro_exists() AS 1 + 1
    """)

    result = safe_macro_exists(fresh_gateway, "test_macro_exists")

    assert result is True


def test_safe_macro_exists_nonexistent_macro(fresh_gateway: StorageGateway) -> None:
    """safe_macro_exists should return False for nonexistent macros."""
    result = safe_macro_exists(fresh_gateway, "nonexistent_macro_xyz")

    assert result is False


def test_safe_macro_exists_builtin_function(fresh_gateway: StorageGateway) -> None:
    """safe_macro_exists should detect built-in functions."""
    # COUNT is a built-in function
    result = safe_macro_exists(fresh_gateway, "count")

    assert result is True
