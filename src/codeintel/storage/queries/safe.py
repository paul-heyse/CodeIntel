"""Safe database query helpers using Ibis.

This module provides typed query helpers that use Ibis for database queries,
properly handling errors without resorting to blind exception catching.
Each function handles specific exception types and returns None or default
values on failure.

This is the canonical location for safe query utilities. The ingestion
infrastructure module re-exports these for backward compatibility.

Examples
--------
>>> from codeintel.storage.queries import safe_count
>>> count = safe_count(gateway, "core.ast_nodes")
>>> count
42
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from duckdb import ColumnExpression, ConstantExpression
from ibis.common.exceptions import (
    ExpressionError,
    IbisError,
    IbisInputError,
    IbisTypeError,
    IntegrityError,
    RelationError,
    TableNotFound,
)
from sqlglot import exp, parse
from sqlglot.errors import ParseError

from codeintel.core.errors.storage import (
    ColumnNotFoundError,
    QueryError,
    TableNotFoundError,
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

IbisBaseError = IbisError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBConnection


DUCKDB_QUERY_ERRORS: tuple[type[BaseException], ...] = (
    DuckDBError,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBInvalidInputException,
    DuckDBBinderException,
    DuckDBDatabaseError,
    DuckDBProgrammingError,
    IbisError,
    IbisBaseError,
    IbisInputError,
    IbisTypeError,
    TableNotFound,
    ExpressionError,
    IntegrityError,
    RelationError,
    KeyError,
)

log = logging.getLogger(__name__)


class UnsafeSqlError(ValueError):
    """Raised when a SQL string violates the select-only perimeter."""

    def __init__(self, reason: str, *, detail: str | None = None) -> None:
        messages = {
            "empty_sql": "Empty SQL string",
            "parse_failed": "Failed to parse SQL",
            "multiple_statements": "Only single-statement SQL is allowed",
            "disallowed_operation": "SQL contains disallowed non-select operations",
            "not_select": "SQL must be a SELECT query",
        }
        msg = messages.get(reason, "Unsafe SQL")
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)
        self.reason = reason
        self.detail = detail


_DISALLOWED_SQL_NODES: tuple[type[exp.Expression], ...] = (
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Insert,
    exp.Update,
)


def assert_single_select_statement(sql: str) -> exp.Expression:
    """Validate a SQL string contains exactly one select-like statement.

    Parameters
    ----------
    sql
        DuckDB SQL string to validate.

    Returns
    -------
    sqlglot.expressions.Expression
        Parsed SQLGlot AST root.

    Raises
    ------
    UnsafeSqlError
        If the SQL contains multiple statements or disallowed operations.
    """
    normalized = sql.strip()
    if not normalized:
        reason = "empty_sql"
        raise UnsafeSqlError(reason)

    try:
        statements = parse(normalized, read="duckdb")
    except ParseError as exc:
        reason = "parse_failed"
        raise UnsafeSqlError(reason, detail=str(exc)) from exc

    if len(statements) != 1:
        reason = "multiple_statements"
        raise UnsafeSqlError(reason)

    root = statements[0]
    if root is None:
        reason = "parse_failed"
        raise UnsafeSqlError(reason)
    for node_type in _DISALLOWED_SQL_NODES:
        if root.find(node_type) is not None:
            reason = "disallowed_operation"
            raise UnsafeSqlError(reason)

    if not root.find(exp.Select) and not isinstance(root, (exp.Select, exp.Union, exp.With)):
        reason = "not_select"
        raise UnsafeSqlError(reason)

    return root


def table_has_rows_for_snapshot(
    con: DuckDBConnection,
    contract: DatasetContract,
    *,
    repo: str,
    commit: str,
) -> bool:
    """Check if a dataset table has rows for the given repo/commit.

    Parameters
    ----------
    con
        DuckDB connection.
    contract
        Dataset contract with schema information.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    bool
        True if at least one row exists, False otherwise.

    Notes
    -----
    When the contract schema includes both ``repo`` and ``commit`` columns, the
    query filters by those values. Otherwise, it checks for any row existence.
    """
    table_key = contract.table_key
    schema = contract.schema
    has_repo_col = schema is not None and any(c.name == "repo" for c in schema.columns)
    has_commit_col = schema is not None and any(c.name == "commit" for c in schema.columns)

    try:
        relation = con.table(table_key)
        if has_repo_col and has_commit_col:
            relation = relation.filter(
                (ColumnExpression("repo") == ConstantExpression(repo))
                & (ColumnExpression("commit") == ConstantExpression(commit))
            )
        return relation.limit(1).fetchone() is not None
    except (DuckDBError, RuntimeError, ValueError, OSError) as exc:
        log.debug("table_has_rows_for_snapshot: error checking %s: %s", table_key, exc)
        return False


def count_rows_for_snapshot(
    con: DuckDBConnection,
    table_key: str,
    *,
    repo: str,
    commit: str,
) -> int:
    """Count rows in a table filtered by repo/commit.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table name (schema.table).
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows matching the repo/commit filter.
    """
    relation = con.table(table_key).filter(
        (ColumnExpression("repo") == ConstantExpression(repo))
        & (ColumnExpression("commit") == ConstantExpression(commit))
    )
    result = relation.count("*").fetchone()
    if result is None:
        return 0
    return int(result[0])


def count_rows_for_tables(
    con: DuckDBConnection,
    tables: Sequence[str],
    *,
    repo: str,
    commit: str,
) -> dict[str, int] | None:
    """Compute row counts for multiple tables filtered by repo/commit.

    Parameters
    ----------
    con
        DuckDB connection.
    tables
        Sequence of fully qualified table names.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, int] | None
        Mapping of table name to row counts, or None if any table fails.
    """
    counts: dict[str, int] = {}
    for table in tables:
        try:
            counts[table] = count_rows_for_snapshot(con, table, repo=repo, commit=commit)
        except DuckDBError:
            return None
    return counts


def safe_count_rows(
    con: DuckDBConnection | None,
    tables: Iterable[str],
    *,
    repo: str,
    commit: str,
) -> dict[str, int] | None:
    """Tolerant variant of count_rows_for_tables that handles None connection.

    Parameters
    ----------
    con
        DuckDB connection, or None.
    tables
        Iterable of fully qualified table names.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, int] | None
        Row counts, or None when connection is unavailable or query fails.
    """
    if con is None:
        return None
    return count_rows_for_tables(con, tuple(tables), repo=repo, commit=commit)


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

        joined = src_tbl.left_join(
            tgt_tbl,
            cast("Any", src_tbl[fk.source_column] == tgt_tbl[fk.ref_column]),
        )

        orphans = joined.filter(cast("Any", tgt_tbl[fk.ref_column].isnull()))

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
    "safe_max_value",
    "safe_min_value",
    "safe_not_null_fraction",
    "safe_table_exists",
]
