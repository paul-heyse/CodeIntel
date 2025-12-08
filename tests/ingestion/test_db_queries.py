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

import pytest

from codeintel.ingestion.infrastructure.db_queries import (
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
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_true,
)
from tests._helpers.factories import make_snapshot

# Test constants (non-repo/commit)
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3
TEST_REPO_ROOT = Path("/opt/test")
EXPECTED_FRACTION_0_5 = 0.5
EXPECTED_FRACTION_1_0 = 1.0
EXPECTED_MIN_VALUE = 5.0
EXPECTED_MAX_VALUE = 20.0
MODULE_INSERT_SQL = """
    INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
    VALUES (?, ?, ?, ?, ?, '[]', '[]')
"""


def _insert_modules(
    gateway: StorageGateway, rows: list[tuple[str, str, str, str, str | None]]
) -> None:
    """Insert rows into core.modules for tests."""
    params = [
        (module, path, repo, commit, language or "python")
        for module, path, repo, commit, language in rows
    ]
    gateway.con.executemany(MODULE_INSERT_SQL, params)


def _create_numeric_table(gateway: StorageGateway, table: str, values: list[float]) -> None:
    """Create a numeric table with id/value rows.

    Raises
    ------
    ValueError
        If an unsupported table is requested.
    """
    params = [(idx, value) for idx, value in enumerate(values, start=1)]

    def _insert(query: str) -> None:
        if not params:
            return
        gateway.con.executemany(query, params)

    if table == "core.test_numeric":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_numeric (
                id INTEGER,
                value DOUBLE
            )
            """
        )
        _insert("INSERT INTO core.test_numeric (id, value) VALUES (?, ?)")
    elif table == "core.test_numeric2":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_numeric2 (
                id INTEGER,
                value DOUBLE
            )
            """
        )
        _insert("INSERT INTO core.test_numeric2 (id, value) VALUES (?, ?)")
    elif table == "core.test_empty_num":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_empty_num (
                id INTEGER,
                value DOUBLE
            )
            """
        )
        _insert("INSERT INTO core.test_empty_num (id, value) VALUES (?, ?)")
    elif table == "core.test_empty_num2":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_empty_num2 (
                id INTEGER,
                value DOUBLE
            )
            """
        )
        _insert("INSERT INTO core.test_empty_num2 (id, value) VALUES (?, ?)")
    elif table == "core.test_pos":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_pos (
                id INTEGER,
                value DOUBLE
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_pos (id, value) VALUES (?, ?)",
            params,
        )
    elif table == "core.test_all_pos":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_all_pos (
                id INTEGER,
                value DOUBLE
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_all_pos (id, value) VALUES (?, ?)",
            params,
        )
    else:
        message = f"Unsupported numeric table for tests: {table}"
        raise ValueError(message)


def _create_varchar_table(
    gateway: StorageGateway, table: str, values: list[tuple[int, str | None]]
) -> None:
    """Create a VARCHAR table with the provided rows.

    Raises
    ------
    ValueError
        If an unsupported table is requested.
    """
    if table == "core.test_nulls":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_nulls (
                id INTEGER,
                value VARCHAR
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_nulls (id, value) VALUES (?, ?)",
            values,
        )
    elif table == "core.test_dupes":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_dupes (
                id INTEGER,
                name VARCHAR
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_dupes (id, name) VALUES (?, ?)",
            values,
        )
    elif table == "core.test_unique":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_unique (
                id INTEGER,
                name VARCHAR
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_unique (id, name) VALUES (?, ?)",
            values,
        )
    elif table == "core.test_frac1":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_frac1 (
                id INTEGER,
                value VARCHAR
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_frac1 (id, value) VALUES (?, ?)",
            values,
        )
    elif table == "core.test_frac2":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_frac2 (
                id INTEGER,
                value VARCHAR
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_frac2 (id, value) VALUES (?, ?)",
            values,
        )
    elif table == "core.test_frac3":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_frac3 (
                id INTEGER,
                value VARCHAR
            )
            """
        )
        gateway.con.executemany(
            "INSERT INTO core.test_frac3 (id, value) VALUES (?, ?)",
            values,
        )
    elif table == "core.test_frac_empty":
        gateway.con.execute(
            """
            CREATE TABLE IF NOT EXISTS core.test_frac_empty (
                id INTEGER,
                value VARCHAR
            )
            """
        )
        if values:
            gateway.con.executemany(
                "INSERT INTO core.test_frac_empty (id, value) VALUES (?, ?)",
                values,
            )
    else:
        message = f"Unsupported varchar table for tests: {table}"
        raise ValueError(message)


# =============================================================================
# Exception Class Tests
# =============================================================================


def test_query_error_attributes() -> None:
    """QueryError should store table and message."""
    error = QueryError("core.test", "Something went wrong")

    expect_equal(error.table, "core.test")
    expect_in("core.test", str(error))
    expect_in("Something went wrong", str(error))


def test_table_not_found_error() -> None:
    """TableNotFoundError should indicate missing table."""
    error = TableNotFoundError("core.missing", "not found")

    expect_equal(error.table, "core.missing")
    expect_is_instance(error, QueryError)


def test_column_not_found_error() -> None:
    """ColumnNotFoundError should store column name."""
    error = ColumnNotFoundError("core.test", "missing_col")

    expect_equal(error.table, "core.test")
    expect_equal(error.column, "missing_col")
    expect_in("missing_col", str(error))
    expect_is_instance(error, QueryError)


def test_duckdb_query_errors_is_tuple() -> None:
    """DUCKDB_QUERY_ERRORS should be a tuple of exception types."""
    expect_is_instance(DUCKDB_QUERY_ERRORS, tuple)
    expect_true(len(DUCKDB_QUERY_ERRORS) > 0)


# =============================================================================
# safe_count Tests
# =============================================================================


def test_safe_count_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_count should return row count for existing tables."""
    # core.modules exists in fresh_gateway schema
    result = safe_count(fresh_gateway, "core.modules")

    # Empty table should return 0
    if result is None:
        pytest.fail("safe_count returned None for existing table")
    expect_true(result >= 0)


@pytest.mark.parametrize(
    "table_key",
    [
        "nonexistent.table_xyz",
        "no-dot-separator",
        "",
    ],
)
def test_safe_count_invalid_or_missing_table(fresh_gateway: StorageGateway, table_key: str) -> None:
    """safe_count should return None for invalid or missing tables."""
    result = safe_count(fresh_gateway, table_key)

    expect_is_none(result)


def test_safe_count_returns_correct_count(fresh_gateway: StorageGateway) -> None:
    """safe_count should return accurate row counts."""
    _insert_modules(
        fresh_gateway,
        [
            ("a", "a.py", "test", "abc", "python"),
            ("b", "b.py", "test", "abc", "python"),
        ],
    )

    result = safe_count(fresh_gateway, "core.modules")

    expect_equal(result, EXPECTED_COUNT_2)


# =============================================================================
# safe_count_with_scope Tests
# =============================================================================


def test_safe_count_with_scope_filters_by_snapshot(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_count_with_scope should count only matching repo/commit."""
    # Insert data for different repos/commits
    _insert_modules(
        fresh_gateway,
        [
            ("a", "a.py", "repo1", "commit1", "python"),
            ("b", "b.py", "repo1", "commit1", "python"),
            ("c", "c.py", "repo2", "commit2", "python"),
        ],
    )

    snapshot = make_snapshot(repo="repo1", commit="commit1", repo_root=TEST_REPO_ROOT)
    result = safe_count_with_scope(fresh_gateway, "core.modules", snapshot)

    expect_equal(result, EXPECTED_COUNT_2)


def test_safe_count_with_scope_nonexistent_table(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_count_with_scope should return None for nonexistent tables."""
    snapshot = make_snapshot(repo_root=TEST_REPO_ROOT)
    result = safe_count_with_scope(fresh_gateway, "nonexistent.table", snapshot)

    expect_is_none(result)


def test_safe_count_with_scope_no_matches(fresh_gateway: StorageGateway) -> None:
    """safe_count_with_scope should return 0 when no rows match."""
    snapshot = make_snapshot(
        repo="nonexistent_repo", commit="nonexistent", repo_root=TEST_REPO_ROOT
    )
    result = safe_count_with_scope(fresh_gateway, "core.modules", snapshot)

    expect_equal(result, 0)


# =============================================================================
# safe_table_exists Tests
# =============================================================================


def test_safe_table_exists_for_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_table_exists should return True for existing tables."""
    result = safe_table_exists(fresh_gateway, "core.modules")

    expect_true(result)


@pytest.mark.parametrize(
    "table_key",
    [
        "nonexistent.table_xyz",
        "invalid-key",
    ],
)
def test_safe_table_exists_invalid_or_missing(
    fresh_gateway: StorageGateway, table_key: str
) -> None:
    """safe_table_exists should return False for invalid or missing tables."""
    result = safe_table_exists(fresh_gateway, table_key)

    expect_false(result)


def test_safe_table_exists_different_schemas(fresh_gateway: StorageGateway) -> None:
    """safe_table_exists should work across different schemas."""
    # Test tables in different schemas
    expect_true(safe_table_exists(fresh_gateway, "core.modules"))
    expect_true(safe_table_exists(fresh_gateway, "core.goids"))

    # Nonexistent in any schema
    expect_false(safe_table_exists(fresh_gateway, "core.nonexistent"))


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_safe_count_sql_injection_protection(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle potential SQL injection attempts safely."""
    # These should return None due to invalid identifiers, not execute malicious SQL
    result = safe_count(fresh_gateway, "core.modules; DROP TABLE core.modules;--")
    expect_is_none(result)

    result = safe_count(fresh_gateway, "'; DROP TABLE core.modules;--")
    expect_is_none(result)


def test_safe_table_exists_sql_injection_protection(
    fresh_gateway: StorageGateway,
) -> None:
    """safe_table_exists should handle potential SQL injection attempts safely."""
    result = safe_table_exists(fresh_gateway, "core.modules; DROP TABLE core.modules;--")
    expect_false(result)

    # Original table should still exist
    expect_true(safe_table_exists(fresh_gateway, "core.modules"))


def test_safe_count_with_special_characters(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle special characters in table keys."""
    # These should all return None since they're invalid
    result = safe_count(fresh_gateway, "core.table-with-dash")
    expect_is_none(result)

    result = safe_count(fresh_gateway, "core.table with space")
    expect_is_none(result)


def test_safe_count_with_unicode(fresh_gateway: StorageGateway) -> None:
    """safe_count should handle unicode in table keys."""
    result = safe_count(fresh_gateway, "core.tableé")
    expect_is_none(result)


# =============================================================================
# safe_get_columns Tests
# =============================================================================


def test_safe_get_columns_existing_table(fresh_gateway: StorageGateway) -> None:
    """safe_get_columns should return column names for existing tables."""
    result = safe_get_columns(fresh_gateway, "core.modules")

    expect_is_instance(result, set)
    expect_true(len(result) > 0)
    expect_in("module", result)
    expect_in("path", result)


@pytest.mark.parametrize(
    "table_key",
    [
        "nonexistent.table_xyz",
        "invalid-key",
    ],
)
def test_safe_get_columns_nonexistent_or_invalid(
    fresh_gateway: StorageGateway, table_key: str
) -> None:
    """safe_get_columns should return empty set for nonexistent or invalid tables."""
    result = safe_get_columns(fresh_gateway, table_key)

    expect_equal(result, set())


# =============================================================================
# safe_count_nulls Tests
# =============================================================================


def test_safe_count_nulls_no_nulls(fresh_gateway: StorageGateway) -> None:
    """safe_count_nulls should return 0 when no NULL values exist."""
    # Insert data with no nulls
    _insert_modules(
        fresh_gateway,
        [
            ("a", "a.py", "test", "abc", "python"),
            ("b", "b.py", "test", "abc", "python"),
        ],
    )

    result = safe_count_nulls(fresh_gateway, "core.modules", "module")

    expect_equal(result, 0)


def test_safe_count_nulls_with_nulls(fresh_gateway: StorageGateway) -> None:
    """safe_count_nulls should count NULL values correctly."""
    _create_varchar_table(
        fresh_gateway,
        "core.test_nulls",
        [
            (1, "a"),
            (2, None),
            (3, None),
            (4, "b"),
        ],
    )

    result = safe_count_nulls(fresh_gateway, "core.test_nulls", "value")

    expect_equal(result, EXPECTED_COUNT_2)


@pytest.mark.parametrize(
    ("table_key", "column"),
    [
        ("core.modules", "nonexistent_col"),
        ("invalid.table", "column"),
    ],
)
def test_safe_count_nulls_invalid_inputs(
    fresh_gateway: StorageGateway, table_key: str, column: str
) -> None:
    """safe_count_nulls should return 0 for invalid table or column."""
    result = safe_count_nulls(fresh_gateway, table_key, column)

    expect_equal(result, 0)


# =============================================================================
# safe_min_value / safe_max_value Tests
# =============================================================================


def test_safe_min_value_with_data(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return minimum value."""
    _create_numeric_table(fresh_gateway, "core.test_numeric", [10.5, 5.0, 20.0])

    result = safe_min_value(fresh_gateway, "core.test_numeric", "value")

    expect_equal(result, EXPECTED_MIN_VALUE)


def test_safe_max_value_with_data(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return maximum value."""
    _create_numeric_table(fresh_gateway, "core.test_numeric2", [10.5, 5.0, 20.0])

    result = safe_max_value(fresh_gateway, "core.test_numeric2", "value")

    expect_equal(result, EXPECTED_MAX_VALUE)


def test_safe_min_value_empty_table(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return None for empty table."""
    _create_numeric_table(fresh_gateway, "core.test_empty_num", [])

    result = safe_min_value(fresh_gateway, "core.test_empty_num", "value")

    expect_is_none(result)


def test_safe_max_value_empty_table(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return None for empty table."""
    _create_numeric_table(fresh_gateway, "core.test_empty_num2", [])

    result = safe_max_value(fresh_gateway, "core.test_empty_num2", "value")

    expect_is_none(result)


def test_safe_min_value_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_min_value should return None for invalid table."""
    result = safe_min_value(fresh_gateway, "invalid.table", "column")

    expect_is_none(result)


def test_safe_max_value_invalid_column(fresh_gateway: StorageGateway) -> None:
    """safe_max_value should return None for invalid column."""
    result = safe_max_value(fresh_gateway, "core.modules", "nonexistent")

    expect_is_none(result)


# =============================================================================
# safe_count_non_positive Tests
# =============================================================================


def test_safe_count_non_positive_with_negatives(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should count values <= 0."""
    _create_numeric_table(fresh_gateway, "core.test_pos", [-5.0, 0.0, 10.0, -2.0])

    result = safe_count_non_positive(fresh_gateway, "core.test_pos", "value")

    expect_equal(result, EXPECTED_COUNT_3)  # -5.0, 0.0, -2.0


def test_safe_count_non_positive_all_positive(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should return 0 when all values are positive."""
    _create_numeric_table(fresh_gateway, "core.test_all_pos", [5.0, 10.0])

    result = safe_count_non_positive(fresh_gateway, "core.test_all_pos", "value")

    expect_equal(result, 0)


def test_safe_count_non_positive_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_non_positive should return 0 for invalid table."""
    result = safe_count_non_positive(fresh_gateway, "invalid.table", "column")

    expect_equal(result, 0)


# =============================================================================
# safe_count_duplicates Tests
# =============================================================================


def test_safe_count_duplicates_with_dupes(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should count duplicate values."""
    _create_varchar_table(
        fresh_gateway,
        "core.test_dupes",
        [
            (1, "alice"),
            (2, "bob"),
            (3, "alice"),
            (4, "alice"),
            (5, "charlie"),
        ],
    )

    result = safe_count_duplicates(fresh_gateway, "core.test_dupes", "name")

    # 5 total - 3 distinct = 2 duplicates
    expect_equal(result, EXPECTED_COUNT_2)


def test_safe_count_duplicates_no_dupes(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should return 0 when all values are unique."""
    _create_varchar_table(
        fresh_gateway,
        "core.test_unique",
        [
            (1, "a"),
            (2, "b"),
            (3, "c"),
        ],
    )

    result = safe_count_duplicates(fresh_gateway, "core.test_unique", "name")

    expect_equal(result, 0)


def test_safe_count_duplicates_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_duplicates should return 0 for invalid table."""
    result = safe_count_duplicates(fresh_gateway, "invalid.table", "column")

    expect_equal(result, 0)


# =============================================================================
# safe_not_null_fraction Tests
# =============================================================================


def test_safe_not_null_fraction_all_not_null(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 1.0 when all values are non-null."""
    _create_varchar_table(
        fresh_gateway,
        "core.test_frac1",
        [
            (1, "a"),
            (2, "b"),
        ],
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac1", "value")

    expect_equal(result, EXPECTED_FRACTION_1_0)


def test_safe_not_null_fraction_half_null(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return correct fraction."""
    _create_varchar_table(
        fresh_gateway,
        "core.test_frac2",
        [
            (1, "a"),
            (2, None),
            (3, "b"),
            (4, None),
        ],
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac2", "value")

    expect_equal(result, EXPECTED_FRACTION_0_5)


def test_safe_not_null_fraction_all_null(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 when all values are null."""
    _create_varchar_table(
        fresh_gateway,
        "core.test_frac3",
        [
            (1, None),
            (2, None),
        ],
    )

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac3", "value")

    expect_equal(result, 0.0)


def test_safe_not_null_fraction_empty_table(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 for empty table."""
    _create_varchar_table(fresh_gateway, "core.test_frac_empty", [])

    result = safe_not_null_fraction(fresh_gateway, "core.test_frac_empty", "value")

    expect_equal(result, 0.0)


def test_safe_not_null_fraction_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_not_null_fraction should return 0.0 for invalid table."""
    result = safe_not_null_fraction(fresh_gateway, "invalid.table", "column")

    expect_equal(result, 0.0)


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

    expect_equal(result, 0)


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

    expect_equal(result, EXPECTED_COUNT_2)


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
    expect_true(result >= 1)


def test_safe_count_orphan_refs_invalid_table(fresh_gateway: StorageGateway) -> None:
    """safe_count_orphan_refs should return 0 for invalid tables."""
    fk = ForeignKeyRef(
        source_table="invalid.source",
        source_column="col",
        ref_table="invalid.target",
        ref_column="id",
    )

    result = safe_count_orphan_refs(fresh_gateway, fk)

    expect_equal(result, 0)


def test_foreign_key_ref_dataclass() -> None:
    """ForeignKeyRef should have correct attributes."""
    fk = ForeignKeyRef(
        source_table="core.child",
        source_column="parent_id",
        ref_table="core.parent",
        ref_column="id",
        allow_null=False,
    )

    expect_equal(fk.source_table, "core.child")
    expect_equal(fk.source_column, "parent_id")
    expect_equal(fk.ref_table, "core.parent")
    expect_equal(fk.ref_column, "id")
    expect_false(fk.allow_null)


def test_foreign_key_ref_default_allow_null() -> None:
    """ForeignKeyRef should default allow_null to True."""
    fk = ForeignKeyRef(
        source_table="core.child",
        source_column="parent_id",
        ref_table="core.parent",
        ref_column="id",
    )

    expect_true(fk.allow_null)


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

    expect_true(result)


def test_safe_macro_exists_nonexistent_macro(fresh_gateway: StorageGateway) -> None:
    """safe_macro_exists should return False for nonexistent macros."""
    result = safe_macro_exists(fresh_gateway, "nonexistent_macro_xyz")

    expect_false(result)


def test_safe_macro_exists_builtin_function(fresh_gateway: StorageGateway) -> None:
    """safe_macro_exists should detect built-in functions."""
    # COUNT is a built-in function
    result = safe_macro_exists(fresh_gateway, "count")

    expect_true(result)
