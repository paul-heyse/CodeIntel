"""DuckDB schema definitions for the CodeIntel metadata warehouse.

These DDLs are derived from README_METADATA.md ("CodeIntel Metadata Outputs")
and cover all exported datasets (goids, call graph, CFG/DFG, coverage, tests,
risk factors, etc.).

This module now delegates to `DuckDBPolicyBackend` for DDL generation while
maintaining backward-compatible function signatures. The string-based DDL
constants (TABLE_DDL, INDEX_DDL) are maintained for compatibility but new
code should use the policy backend directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection

from codeintel.config.datasets import TableSchema, get_dataset_contracts_by_table_key
from codeintel.storage.sql.primitives import quote_identifier

if TYPE_CHECKING:
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

SCHEMAS = ("build", "core", "graph", "analytics", "docs")
log = logging.getLogger(__name__)

__all__ = [
    "INDEX_DDL",
    "SCHEMAS",
    "TABLE_DDL",
    "TABLE_DDL_IF_NOT_EXISTS",
    "apply_all_schemas",
    "assert_schema_alignment",
    "create_schemas",
    "ensure_schemas_preserve",
]


def _build_table_ddl(table: TableSchema) -> str:
    """Generate CREATE TABLE DDL from a TableSchema.

    Returns
    -------
    str
        CREATE TABLE statement for the provided schema.

    .. deprecated::
        Use `DuckDBPolicyBackend.create_table_from_schema()` instead.
    """
    col_lines: list[str] = []
    for col in table.columns:
        nullable_sql = "" if col.nullable else " NOT NULL"
        col_lines.append(f"    {quote_identifier(col.name)} {col.type}{nullable_sql}")
    if table.primary_key:
        pk_cols = ", ".join(quote_identifier(col) for col in table.primary_key)
        col_lines.append(f"    PRIMARY KEY ({pk_cols})")
    cols_sql = ",\n".join(col_lines)
    return (
        f"DROP TABLE IF EXISTS {quote_identifier(table.schema)}.{quote_identifier(table.name)};\n"
        f"CREATE TABLE {quote_identifier(table.schema)}.{quote_identifier(table.name)} (\n"
        f"{cols_sql}\n"
        ");"
    )


def _build_table_ddl_if_not_exists(table: TableSchema) -> str:
    """Generate non-destructive CREATE TABLE DDL from a TableSchema.

    Returns
    -------
    str
        CREATE TABLE IF NOT EXISTS statement for the provided schema.

    .. deprecated::
        Use `DuckDBPolicyBackend.create_table_from_schema(if_not_exists=True)` instead.
    """
    col_lines: list[str] = []
    for col in table.columns:
        nullable_sql = "" if col.nullable else " NOT NULL"
        col_lines.append(f"    {quote_identifier(col.name)} {col.type}{nullable_sql}")
    if table.primary_key:
        pk_cols = ", ".join(quote_identifier(col) for col in table.primary_key)
        col_lines.append(f"    PRIMARY KEY ({pk_cols})")
    cols_sql = ",\n".join(col_lines)
    return f"CREATE TABLE IF NOT EXISTS {quote_identifier(table.schema)}.{quote_identifier(table.name)} (\n{cols_sql}\n);"


_TABLE_CREATION_DENYLIST = {"docs.v_validation_summary"}


# Legacy DDL dictionaries - maintained for backward compatibility
# New code should use DuckDBPolicyBackend directly
TABLE_DDL: dict[str, str] = {
    key: _build_table_ddl(contract.schema)
    for key, contract in get_dataset_contracts_by_table_key().items()
    if contract.schema is not None and key not in _TABLE_CREATION_DENYLIST
}
TABLE_DDL_IF_NOT_EXISTS: dict[str, str] = {
    key: _build_table_ddl_if_not_exists(contract.schema)
    for key, contract in get_dataset_contracts_by_table_key().items()
    if contract.schema is not None and key not in _TABLE_CREATION_DENYLIST
}


def _build_index_ddl(table: TableSchema) -> list[str]:
    """Generate CREATE INDEX statements from a TableSchema.

    Parameters
    ----------
    table
        Table schema definition with indexes.

    Returns
    -------
    list[str]
        List of CREATE INDEX statements.

    .. deprecated::
        Use `DuckDBPolicyBackend.create_indexes_from_schema()` instead.
    """
    statements: list[str] = []
    for index in table.indexes:
        columns = ", ".join(quote_identifier(col) for col in index.columns)
        uniqueness = "UNIQUE " if index.unique else ""
        statements.append(
            f"CREATE {uniqueness}INDEX IF NOT EXISTS {quote_identifier(index.name)} "
            f"ON {quote_identifier(table.schema)}.{quote_identifier(table.name)}({columns});"
        )
    return statements


INDEX_DDL: tuple[str, ...] = tuple(
    ddl
    for contract in get_dataset_contracts_by_table_key().values()
    if contract.schema is not None
    for ddl in _build_index_ddl(contract.schema)
)


def _get_policy_backend(con: DuckDBPyConnection) -> DuckDBPolicyBackend:
    """Create a minimal policy backend wrapper for a raw connection.

    This is a compatibility shim to allow existing code using raw connections
    to use the policy backend. New code should use the StorageGateway directly.

    Parameters
    ----------
    con
        DuckDB connection.

    Returns
    -------
    DuckDBPolicyBackend
        Policy backend instance wrapping the connection.
    """
    # Import here to avoid circular imports
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend  # noqa: PLC0415

    # Create a minimal gateway-like wrapper
    class _MinimalGateway:
        def __init__(self, connection: DuckDBPyConnection) -> None:
            self._con = connection

        @property
        def con(self) -> DuckDBPyConnection:
            return self._con

    return DuckDBPolicyBackend(gateway=_MinimalGateway(con))  # type: ignore[arg-type]


def create_schemas(con: DuckDBPyConnection) -> None:
    """Ensure logical schemas (core, graph, analytics, docs) exist."""
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")


def apply_all_schemas(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Create all known tables in the current DuckDB database.

    Call this once at startup before running any pipeline steps that
    insert into these tables.

    This function now uses the DuckDBPolicyBackend internally but maintains
    the same external interface for backward compatibility.

    Parameters
    ----------
    con
        DuckDB connection.
    extra_ddl
        Optional additional DDL statements to execute.
    """
    backend = _get_policy_backend(con)
    backend.ensure_all_schemas(drop_existing=True, extra_ddl=extra_ddl)


def ensure_schemas_preserve(
    con: DuckDBPyConnection,
    extra_ddl: Iterable[str] | None = None,
) -> None:
    """Ensure schemas and tables exist without dropping existing data.

    Creates missing tables and indexes using IF NOT EXISTS; existing tables are left
    untouched. Use assert_schema_alignment separately to detect drift.

    Parameters
    ----------
    con
        DuckDB connection.
    extra_ddl
        Optional additional DDL statements to execute.
    """
    backend = _get_policy_backend(con)
    backend.ensure_schemas_preserve(extra_ddl=extra_ddl)


def assert_schema_alignment(
    con: DuckDBPyConnection,
    *,
    strict: bool = True,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Validate that the live DuckDB schema matches the DatasetContract definitions.

    Returns
    -------
    list[str]
        Human-readable drift messages; empty when aligned.

    Raises
    ------
    RuntimeError
        If strict is True and schema drift is detected.
    """
    issues: list[str] = []
    for contract in get_dataset_contracts_by_table_key().values():
        if contract.schema is None:
            continue
        table = contract.schema
        rows = con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [table.schema, table.name],
        ).fetchall()
        actual = [row[0] for row in rows]
        expected = table.column_names()
        if actual != expected:
            issues.append(f"{table.fq_name}: expected {expected} got {actual}")

    if issues:
        message = "; ".join(issues)
        logref = logger or log
        logref.error("Schema drift detected: %s", message)
        if strict:
            error_message = f"Schema drift detected: {message}"
            raise RuntimeError(error_message)
    return issues
