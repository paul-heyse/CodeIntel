"""Safe database query helpers for ingestion plugins using Ibis.

This module provides typed query helpers that use Ibis for database queries,
properly handling errors without resorting to blind exception catching.
Each function handles specific exception types and returns None or default
values on failure.

Examples
--------
>>> from codeintel.ingestion.infrastructure.db_queries import safe_count
>>> count = safe_count(gateway, "core.ast_nodes")
>>> count  # Returns int or None on error
42
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

# Import Ibis exception types - catch a broad set of Ibis errors
from ibis.common.exceptions import (
    ExpressionError,
    IbisError,
    IbisInputError,
    IbisTypeError,
    IntegrityError,
    RelationError,
    TableNotFound,
)

from codeintel.storage.gateway.protocol import (
    DuckDBBinderException,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBDatabaseError,
    DuckDBError,
    DuckDBInvalidInputException,
    DuckDBProgrammingError,
)

# Alias for compatibility
IbisBaseError = IbisError

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

# All exceptions we catch for safe database operations
DUCKDB_QUERY_ERRORS: tuple[type[BaseException], ...] = (
    # DuckDB exceptions
    DuckDBError,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBInvalidInputException,
    DuckDBBinderException,
    DuckDBDatabaseError,
    DuckDBProgrammingError,
    # Ibis exceptions
    IbisError,
    IbisBaseError,
    IbisInputError,
    IbisTypeError,
    TableNotFound,
    ExpressionError,
    IntegrityError,
    RelationError,
    # Catch-all for edge cases with invalid identifiers
    KeyError,
)

log = logging.getLogger(__name__)


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
    """Safely count rows in a table using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        return cast("int", tbl.count().execute())
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count query failed for %s: %s", table_key, exc)
        return None


def safe_count_with_scope(
    gateway: StorageGateway,
    table_key: str,
    snapshot: SnapshotRef,
) -> int | None:
    """Safely count rows in a table scoped to a snapshot using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        filtered = tbl.filter(
            cast("Any", (tbl.repo == snapshot.repo) & (tbl.commit == snapshot.commit))
        )
        return cast("int", filtered.count().execute())
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Scoped count query failed for %s: %s", table_key, exc)
        return None


def safe_table_exists(gateway: StorageGateway, table_key: str) -> bool:
    """Check if a table exists in the database using Ibis.

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
        gateway.ibis.table(table_key)
    except DUCKDB_QUERY_ERRORS:
        return False
    else:
        return True


def safe_get_columns(gateway: StorageGateway, table_key: str) -> set[str]:
    """Get column names for a table using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        return set(tbl.columns)
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Get columns failed for %s: %s", table_key, exc)
        return set()


def safe_count_nulls(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count NULL values in a column using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        col = tbl[column]
        return cast("int", tbl.filter(cast("Any", col.isnull())).count().execute())
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count nulls failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_min_value(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float | None:
    """Get minimum value in a numeric column using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        col = tbl[column]
        result = col.min().execute()
        if result is None:
            return None
        # Convert to float and handle NaN (returned for empty tables)
        value = float(cast("Any", result))
        return None if math.isnan(value) else value
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Min value failed for %s.%s: %s", table_key, column, exc)
        return None


def safe_max_value(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float | None:
    """Get maximum value in a numeric column using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        col = tbl[column]
        result = col.max().execute()
        if result is None:
            return None
        # Convert to float and handle NaN (returned for empty tables)
        value = float(cast("Any", result))
        return None if math.isnan(value) else value
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Max value failed for %s.%s: %s", table_key, column, exc)
        return None


def safe_count_non_positive(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count non-positive values in a numeric column using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        col = cast("Any", tbl[column])
        return cast("int", tbl.filter(col <= 0).count().execute())
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count non-positive failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_count_duplicates(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count duplicate values in a column using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        col = tbl[column]
        # Filter to non-null values only
        non_null = tbl.filter(cast("Any", col.notnull()))
        total = cast("int", non_null.count().execute())
        distinct = cast("int", col.nunique().execute())
        return total - distinct
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count duplicates failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_not_null_fraction(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float:
    """Get fraction of non-null values in a column using Ibis.

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
        tbl = gateway.ibis.table(table_key)
        col = tbl[column]
        total = cast("int", tbl.count().execute())
        if total == 0:
            return 0.0
        non_null_count = cast("int", tbl.filter(cast("Any", col.notnull())).count().execute())
        return float(non_null_count) / float(total)
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
    """Count orphaned foreign key references using Ibis.

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
        src_tbl = gateway.ibis.table(fk.source_table)
        tgt_tbl = gateway.ibis.table(fk.ref_table)

        # Left join source to target
        joined = src_tbl.left_join(
            tgt_tbl,
            cast("Any", src_tbl[fk.source_column] == tgt_tbl[fk.ref_column]),
        )

        # Filter to orphans: target key is null
        orphans = joined.filter(cast("Any", tgt_tbl[fk.ref_column].isnull()))

        # Optionally exclude null source values
        if not fk.allow_null:
            orphans = orphans.filter(cast("Any", src_tbl[fk.source_column].notnull()))

        return cast("int", orphans.count().execute())
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

    Note: This function uses raw SQL as it queries DuckDB system functions
    which are not accessible via Ibis table interface.

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
        # Use Ibis SQL interface for system query
        result = gateway.ibis.con.raw_sql(
            "SELECT * FROM duckdb_functions() WHERE function_name = ?",
            parameters=[macro_name],
        ).fetchone()
    except Exception:  # noqa: BLE001
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
