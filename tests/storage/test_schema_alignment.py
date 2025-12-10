"""Ensure DuckDB schemas are applied and aligned with TABLE_SCHEMAS."""

from __future__ import annotations

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schema import (
    assert_schema_alignment,
    create_schemas,
    ensure_schemas_preserve,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_not_empty,
    expect_true,
)


def test_apply_and_validate_schema_alignment(fresh_gateway: StorageGateway) -> None:
    """
    Schema application should create expected columns (incl. decorator spans).

    Raises
    ------
    AssertionError
        If schema drift is detected or decorator columns are missing.
    """
    con = fresh_gateway.con
    issues = assert_schema_alignment(con, strict=False)
    if issues:
        message = f"Schema drift detected: {issues}"
        raise AssertionError(message)

    cols = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'core' AND table_name = 'ast_nodes'
        ORDER BY ordinal_position
        """
    ).fetchall()
    col_names = [row[0] for row in cols]
    expected_columns = {"decorator_start_line", "decorator_end_line"}
    if not expected_columns.issubset(set(col_names)):
        message = f"Missing decorator columns in core.ast_nodes: {col_names}"
        raise AssertionError(message)


def test_ensure_schemas_preserve_creates_tables_idempotently(
    schema_gateway: StorageGateway,
) -> None:
    """Verify ensure_schemas_preserve can be called multiple times safely."""
    con = schema_gateway.con

    ensure_schemas_preserve(con)

    first_count = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema IN ('core', 'graph', 'analytics', 'docs')
        """
    ).fetchone()

    ensure_schemas_preserve(con)

    second_count = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema IN ('core', 'graph', 'analytics', 'docs')
        """
    ).fetchone()

    expect_is_not_none(first_count)
    expect_is_not_none(second_count)
    if first_count is not None and second_count is not None:
        expect_equal(first_count[0], second_count[0])


def test_ensure_schemas_preserve_with_extra_ddl(
    schema_gateway: StorageGateway,
) -> None:
    """Verify ensure_schemas_preserve applies extra DDL statements."""
    con = schema_gateway.con

    extra_ddl = [
        "CREATE TABLE IF NOT EXISTS core.test_extra_table (id INTEGER PRIMARY KEY);",
    ]

    ensure_schemas_preserve(con, extra_ddl=extra_ddl)

    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'core' AND table_name = 'test_extra_table'
        """
    ).fetchone()

    expect_is_not_none(row)
    if row is not None:
        expect_equal(row[0], 1)


def test_assert_schema_alignment_detects_drift_nonstrict(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify assert_schema_alignment returns issues without raising when not strict."""
    con = fresh_gateway.con

    con.execute("ALTER TABLE core.goids ADD COLUMN test_drift_column VARCHAR;")

    issues = assert_schema_alignment(con, strict=False)

    expect_not_empty(issues)
    expect_true(any("core.goids" in issue for issue in issues))


def test_create_schemas_creates_all_namespaces(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify create_schemas creates core, graph, analytics, docs schemas."""
    con = fresh_gateway.con

    con.execute("DROP SCHEMA IF EXISTS core CASCADE;")
    con.execute("DROP SCHEMA IF EXISTS graph CASCADE;")
    con.execute("DROP SCHEMA IF EXISTS analytics CASCADE;")
    con.execute("DROP SCHEMA IF EXISTS docs CASCADE;")

    create_schemas(con)

    rows = con.execute(
        """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name IN ('core', 'graph', 'analytics', 'docs')
        ORDER BY schema_name
        """
    ).fetchall()

    schema_names = [row[0] for row in rows]
    expect_in("core", schema_names)
    expect_in("graph", schema_names)
    expect_in("analytics", schema_names)
    expect_in("docs", schema_names)
