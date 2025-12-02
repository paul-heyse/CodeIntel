"""Safe database query helpers for ingestion plugins.

This module provides typed query helpers that properly handle database errors
without resorting to blind exception catching. Each function handles specific
exception types and returns None or default values on failure.

The helpers use SafeTableRef and SafeColumnRef for SQL injection prevention
and provide consistent logging for debugging.

Examples
--------
>>> from codeintel.ingestion.infrastructure_utilities.db_queries import safe_count
>>> count = safe_count(gateway, "core.ast_nodes")
>>> count  # Returns int or None on error
42

NOTE: All f-string SQL construction uses validated identifiers (SafeTableRef/SafeColumnRef),
making them safe from SQL injection.
"""
# ruff: noqa: S608

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.ingestion.infrastructure_utilities.safe_sql import (
    InvalidIdentifierError,
    SafeColumnRef,
    SafeTableRef,
)
from codeintel.storage.gateway import (
    DuckDBBinderException,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBError,
    DuckDBInvalidInputException,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


# Exception types from DuckDB that indicate query/connection issues
DUCKDB_QUERY_ERRORS: tuple[type[BaseException], ...] = (
    DuckDBError,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBInvalidInputException,
    DuckDBBinderException,
)


class QueryError(Exception):
    """Base class for query execution errors.

    Attributes
    ----------
    table
        The table involved in the failed query.
    message
        Description of the error.
    """

    def __init__(self, table: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        table
            The table involved in the failed query.
        message
            Description of the error.
        """
        self.table = table
        super().__init__(f"Query error on {table}: {message}")


class TableNotFoundError(QueryError):
    """Table does not exist in the database."""


class ColumnNotFoundError(QueryError):
    """Column does not exist in the table.

    Attributes
    ----------
    column
        The missing column name.
    """

    def __init__(self, table: str, column: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        table
            The table being queried.
        column
            The missing column name.
        """
        self.column = column
        super().__init__(table, f"column '{column}' not found")


def safe_count(gateway: StorageGateway, table_key: str) -> int | None:
    """Safely count rows in a table.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.

    Returns
    -------
    int | None
        Row count or None if query failed.

    Examples
    --------
    >>> count = safe_count(gateway, "core.ast_nodes")
    >>> count
    42
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        result = gateway.con.execute(
            f"SELECT COUNT(*) FROM {ref.full_name}"  # Safe: validated identifier
        ).fetchone()
        return int(result[0]) if result else None
    except InvalidIdentifierError:
        log.debug("Invalid table key: %s", table_key)
        return None
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count query failed for %s: %s", table_key, exc)
        return None


def safe_count_with_scope(
    gateway: StorageGateway,
    table_key: str,
    snapshot: SnapshotRef,
) -> int | None:
    """Safely count rows in a table scoped to a snapshot.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    snapshot
        Snapshot reference for scoping.

    Returns
    -------
    int | None
        Row count or None if query failed.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        result = gateway.con.execute(
            f"SELECT COUNT(*) FROM {ref.full_name} WHERE repo = ? AND commit = ?",
            [snapshot.repo, snapshot.commit],
        ).fetchone()
        return int(result[0]) if result else None
    except InvalidIdentifierError:
        log.debug("Invalid table key: %s", table_key)
        return None
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Scoped count query failed for %s: %s", table_key, exc)
        return None


def safe_table_exists(gateway: StorageGateway, table_key: str) -> bool:
    """Check if a table exists in the database.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.

    Returns
    -------
    bool
        True if table exists, False otherwise.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        result = gateway.con.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            """,
            [ref.schema, ref.table],
        ).fetchone()
    except InvalidIdentifierError:
        log.debug("Invalid table key: %s", table_key)
        return False
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Table exists check failed for %s: %s", table_key, exc)
        return False
    return result is not None


def safe_get_columns(gateway: StorageGateway, table_key: str) -> set[str]:
    """Get column names for a table.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.

    Returns
    -------
    set[str]
        Column names, or empty set if query failed.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        rows = gateway.con.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            """,
            [ref.schema, ref.table],
        ).fetchall()
        return {str(row[0]) for row in rows}
    except InvalidIdentifierError:
        log.debug("Invalid table key: %s", table_key)
        return set()
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Get columns failed for %s: %s", table_key, exc)
        return set()


def safe_count_nulls(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count NULL values in a column.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    column
        Column name to check.

    Returns
    -------
    int
        Count of NULL values, or 0 if query failed.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        col = SafeColumnRef(column)
        result = gateway.con.execute(
            f"SELECT COUNT(*) FROM {ref.full_name} WHERE {col.name} IS NULL"
        ).fetchone()
        return int(result[0]) if result else 0
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return 0
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count nulls failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_min_value(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float | None:
    """Get minimum value in a numeric column.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    column
        Column name to query.

    Returns
    -------
    float | None
        Minimum value or None if query failed or no data.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        col = SafeColumnRef(column)
        result = gateway.con.execute(f"SELECT MIN({col.name}) FROM {ref.full_name}").fetchone()
        return float(result[0]) if result and result[0] is not None else None
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return None
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Min value failed for %s.%s: %s", table_key, column, exc)
        return None


def safe_max_value(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float | None:
    """Get maximum value in a numeric column.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    column
        Column name to query.

    Returns
    -------
    float | None
        Maximum value or None if query failed or no data.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        col = SafeColumnRef(column)
        result = gateway.con.execute(f"SELECT MAX({col.name}) FROM {ref.full_name}").fetchone()
        return float(result[0]) if result and result[0] is not None else None
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return None
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Max value failed for %s.%s: %s", table_key, column, exc)
        return None


def safe_count_non_positive(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count non-positive values in a numeric column.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    column
        Column name to query.

    Returns
    -------
    int
        Count of non-positive values, or 0 if query failed.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        col = SafeColumnRef(column)
        result = gateway.con.execute(
            f"SELECT COUNT(*) FROM {ref.full_name} WHERE {col.name} <= 0"
        ).fetchone()
        return int(result[0]) if result else 0
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return 0
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count non-positive failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_count_duplicates(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count duplicate values in a column.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    column
        Column name to query.

    Returns
    -------
    int
        Count of duplicate values (total rows - distinct), or 0 if query failed.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        col = SafeColumnRef(column)
        result = gateway.con.execute(
            f"""
            SELECT COUNT(*) - COUNT(DISTINCT {col.name}) FROM {ref.full_name}
            WHERE {col.name} IS NOT NULL
            """
        ).fetchone()
        return int(result[0]) if result else 0
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return 0
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count duplicates failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_not_null_fraction(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float:
    """Get fraction of non-null values in a column.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    table_key
        Table key in 'schema.table' format.
    column
        Column name to query.

    Returns
    -------
    float
        Fraction of non-null values (0.0 to 1.0), or 0.0 if query failed.
    """
    try:
        ref = SafeTableRef.from_key(table_key)
        col = SafeColumnRef(column)
        result = gateway.con.execute(
            f"""
            SELECT
                CAST(COUNT({col.name}) AS DOUBLE) / NULLIF(COUNT(*), 0)
            FROM {ref.full_name}
            """
        ).fetchone()
        return float(result[0]) if result and result[0] is not None else 0.0
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return 0.0
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Not null fraction failed for %s.%s: %s", table_key, column, exc)
        return 0.0


@dataclass(frozen=True)
class ForeignKeyRef:
    """Foreign key reference specification for orphan counting.

    Attributes
    ----------
    source_table
        Source table key in 'schema.table' format.
    source_column
        Column in source table holding the reference.
    ref_table
        Referenced table key in 'schema.table' format.
    ref_column
        Referenced column in target table.
    allow_null
        Whether to exclude NULL values from orphan count.
    """

    source_table: str
    source_column: str
    ref_table: str
    ref_column: str
    allow_null: bool = True


def safe_count_orphan_refs(gateway: StorageGateway, fk: ForeignKeyRef) -> int:
    """Count orphaned foreign key references.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    fk
        Foreign key reference specification.

    Returns
    -------
    int
        Count of orphaned references, or 0 if query failed.
    """
    try:
        src_ref = SafeTableRef.from_key(fk.source_table)
        src_col = SafeColumnRef(fk.source_column)
        tgt_ref = SafeTableRef.from_key(fk.ref_table)
        tgt_col = SafeColumnRef(fk.ref_column)

        null_clause = f"AND t.{src_col.name} IS NOT NULL" if not fk.allow_null else ""
        query = f"""
            SELECT COUNT(*) FROM {src_ref.full_name} t
            LEFT JOIN {tgt_ref.full_name} r
                ON t.{src_col.name} = r.{tgt_col.name}
            WHERE r.{tgt_col.name} IS NULL {null_clause}
        """
        result = gateway.con.execute(query).fetchone()
        return int(result[0]) if result else 0
    except InvalidIdentifierError as exc:
        log.debug("Invalid identifier: %s", exc)
        return 0
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug(
            "Count orphan refs failed for %s.%s -> %s.%s: %s",
            fk.source_table,
            fk.source_column,
            fk.ref_table,
            fk.ref_column,
            exc,
        )
        return 0


def safe_macro_exists(gateway: StorageGateway, macro_name: str) -> bool:
    """Check if a DuckDB macro exists.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    macro_name
        Name of the macro to check.

    Returns
    -------
    bool
        True if macro exists, False otherwise.
    """
    try:
        result = gateway.con.execute(
            "SELECT * FROM duckdb_functions() WHERE function_name = ?",
            [macro_name],
        ).fetchone()
    except DUCKDB_QUERY_ERRORS:
        return False
    return result is not None


__all__ = [
    "DUCKDB_QUERY_ERRORS",
    "ColumnNotFoundError",
    "ForeignKeyRef",
    "QueryError",
    "TableNotFoundError",
    "safe_count",
    "safe_count_duplicates",
    "safe_count_non_positive",
    "safe_count_nulls",
    "safe_count_orphan_refs",
    "safe_count_with_scope",
    "safe_get_columns",
    "safe_macro_exists",
    "safe_max_value",
    "safe_min_value",
    "safe_not_null_fraction",
    "safe_table_exists",
]
