"""Safe database query helpers using DuckDB relations.

This module provides typed query helpers that use DuckDB relations or
parameterized SQL, properly handling errors without resorting to blind
exception catching. Each function handles specific exception types and
returns None or default values on failure.

This is the canonical location for safe query utilities.

Examples
--------
>>> from codeintel.storage.queries import safe_count
>>> count = safe_count(gateway, "core.ast_nodes")
>>> count
42
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from duckdb import ColumnExpression, ConstantExpression, FunctionExpression
from sqlglot import exp, parse
from sqlglot.errors import ParseError, SqlglotError

from codeintel.core.errors.storage import ColumnNotFoundError, QueryError, TableNotFoundError
from codeintel.core.filters import FilterSpecInput
from codeintel.core.queries.filter_compiler import (
    FilterCompilerError,
    compile_filter_predicates,
    duckdb_filter_expression,
)
from codeintel.core.sqlglot_tools import (
    SELECT_ONLY_DISALLOWED_NODES,
    AstCapabilityConfig,
    AstCapabilityError,
    canonicalize_expression_duckdb,
    ensure_ast_capability,
    extract_table_refs,
)
from codeintel.storage.duckdb_types import (
    DuckDBBinderException,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBDatabaseError,
    DuckDBError,
    DuckDBInvalidInputException,
    DuckDBProgrammingError,
)
from codeintel.storage.helpers.table_key import is_valid_table_key
from codeintel.storage.query_results import coerce_int, coerce_optional_float

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBRelation, Expression
    from codeintel.storage.gateway import StorageGateway


DUCKDB_QUERY_ERRORS: tuple[type[BaseException], ...] = (
    DuckDBError,
    DuckDBCatalogException,
    DuckDBConnectionException,
    DuckDBInvalidInputException,
    DuckDBBinderException,
    DuckDBDatabaseError,
    DuckDBProgrammingError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)

log = logging.getLogger(__name__)
_SNAPSHOT_FILTER_COLUMNS = frozenset({"commit", "repo"})


def _ensure_valid_table_key(table_key: str) -> bool:
    if not is_valid_table_key(table_key):
        log.debug("Invalid table key provided: %s", table_key)
        return False
    return True


def _relation_for_table_key(
    gateway: StorageGateway,
    table_key: str,
) -> DuckDBRelation | None:
    if not _ensure_valid_table_key(table_key):
        return None
    try:
        return gateway.relation_from_table_key(table_key)
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Failed to resolve relation for %s: %s", table_key, exc)
        return None


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


@dataclass(frozen=True, slots=True)
class SqlIngressPolicy:
    """Additional perimeter policy for SELECT-only SQL ingress."""

    allowed_schemas: frozenset[str] | None = None
    allowed_tables: frozenset[str] | None = None
    allowed_functions: frozenset[str] | None = None
    deny_functions: frozenset[str] = frozenset()
    allow_unqualified_tables: bool = True
    allow_cross_database_references: bool = False


def assert_select_perimeter(
    sql: str,
    *,
    policy: SqlIngressPolicy,
    enforce_safe_sql: bool = True,
) -> exp.Expression:
    """Validate a SQL string against a policy perimeter.

    Parameters
    ----------
    sql
        DuckDB SQL string to validate.
    policy
        Additional ingress policy constraints applied after SELECT-only validation.
    enforce_safe_sql
        Whether to enforce the DuckDBSafe SQL capability envelope.

    Returns
    -------
    sqlglot.expressions.Expression
        Parsed SQLGlot AST root.
    """
    root = assert_single_select_statement(sql, enforce_safe_sql=enforce_safe_sql)
    _validate_ingress_tables(root, policy=policy)
    _validate_ingress_functions(root, policy=policy)
    try:
        return canonicalize_expression_duckdb(root)
    except (SqlglotError, TypeError, ValueError):
        return root


def _validate_ingress_tables(root: exp.Expression, *, policy: SqlIngressPolicy) -> None:
    reason = "policy_violation"
    tables = extract_table_refs(root)
    allowed_schemas = (
        {s.lower() for s in policy.allowed_schemas} if policy.allowed_schemas else None
    )
    allowed_tables = {t.lower() for t in policy.allowed_tables} if policy.allowed_tables else None

    for table in tables:
        schema = table.db
        catalog = getattr(table, "catalog", None)
        if not policy.allow_cross_database_references and isinstance(catalog, str) and catalog:
            detail = "Cross-database references are not allowed"
            raise UnsafeSqlError(reason, detail=detail)

        if schema:
            if allowed_schemas is not None and schema.lower() not in allowed_schemas:
                detail = f"Schema {schema!r} is not allowed"
                raise UnsafeSqlError(
                    reason,
                    detail=detail,
                )
            key = f"{schema}.{table.name}".lower()
        else:
            if not policy.allow_unqualified_tables:
                detail = f"Unqualified table {table.name!r} is not allowed"
                raise UnsafeSqlError(
                    reason,
                    detail=detail,
                )
            key = table.name.lower()

        if allowed_tables is not None and key not in allowed_tables:
            detail = f"Table {key!r} is not allowed"
            raise UnsafeSqlError(reason, detail=detail)


def _validate_ingress_functions(root: exp.Expression, *, policy: SqlIngressPolicy) -> None:
    reason = "policy_violation"
    deny = {name.lower() for name in policy.deny_functions}
    allow = (
        {name.lower() for name in policy.allowed_functions} if policy.allowed_functions else None
    )
    if not deny and allow is None:
        return
    for func_name in _extract_function_names(root):
        if allow is not None and func_name.lower() not in allow:
            detail = f"Function {func_name!r} is not allowed"
            raise UnsafeSqlError(reason, detail=detail)
        if func_name.lower() in deny:
            detail = f"Function {func_name!r} is not allowed"
            raise UnsafeSqlError(reason, detail=detail)


def _extract_function_names(root: exp.Expression) -> tuple[str, ...]:
    names: set[str] = set()
    for node in root.find_all(exp.Func):
        if isinstance(node, exp.Anonymous):
            if node.name:
                names.add(node.name)
            continue
        names.add(node.sql_name())
    return tuple(sorted(names))


def assert_single_select_statement(sql: str, *, enforce_safe_sql: bool = True) -> exp.Expression:
    """Validate a SQL string contains exactly one select-like statement.

    Parameters
    ----------
    sql
        DuckDB SQL string to validate.
    enforce_safe_sql
        Whether to enforce the DuckDBSafe SQL capability envelope.

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

    if not root.find(exp.Select) and not isinstance(root, (exp.Select, exp.Union, exp.With)):
        reason = "not_select"
        raise UnsafeSqlError(reason)

    try:
        ensure_ast_capability(
            root,
            AstCapabilityConfig(
                disallowed_nodes=SELECT_ONLY_DISALLOWED_NODES,
                allow_aggregates=True,
                enforce_safe_sql=enforce_safe_sql,
                log_context="storage_sql_ingress",
            ),
        )
    except AstCapabilityError as exc:
        reason = "disallowed_operation"
        raise UnsafeSqlError(reason, detail=str(exc)) from exc

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
            predicate = _snapshot_filter_expression(repo=repo, commit=commit)
            relation = relation.filter(predicate)
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
    predicate = _snapshot_filter_expression(repo=repo, commit=commit)
    relation = con.table(table_key).filter(predicate)
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
    table_list = tuple(tables)
    if any(not _ensure_valid_table_key(table_key) for table_key in table_list):
        return None
    return count_rows_for_tables(con, table_list, repo=repo, commit=commit)


def safe_count(gateway: StorageGateway, table_key: str) -> int | None:
    """Safely count rows in a table using DuckDB relations.

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
    if not _ensure_valid_table_key(table_key):
        return None
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return None
    try:
        result = relation.count("*").fetchone()
        if result is None:
            return 0
        return coerce_int(result[0], ctx=f"{table_key}.count()")
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count query failed for %s: %s", table_key, exc)
        return None


def safe_count_with_scope(
    gateway: StorageGateway,
    table_key: str,
    snapshot: SnapshotRef,
) -> int | None:
    """Safely count rows in a table scoped to a snapshot using relations.

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
    if not _ensure_valid_table_key(table_key):
        return None
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return None
    columns = set(relation.columns)
    if "repo" in columns and "commit" in columns:
        predicate = _snapshot_filter_expression(repo=snapshot.repo, commit=snapshot.commit)
        relation = relation.filter(predicate)
    try:
        result = relation.count("*").fetchone()
        if result is None:
            return 0
        return coerce_int(result[0], ctx=f"{table_key}.count(scope)")
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Scoped count query failed for %s: %s", table_key, exc)
        return None


def safe_table_exists(gateway: StorageGateway, table_key: str) -> bool:
    """Check if a table exists in the database using relations.

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
    if not _ensure_valid_table_key(table_key):
        return False
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return False
    try:
        _ = relation.columns
    except DUCKDB_QUERY_ERRORS:
        return False
    return True


def safe_get_columns(gateway: StorageGateway, table_key: str) -> set[str]:
    """Get column names for a table using relations.

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
    if not _ensure_valid_table_key(table_key):
        return set()
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return set()
    try:
        return set(relation.columns)
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Get columns failed for %s: %s", table_key, exc)
        return set()


def safe_count_nulls(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count NULL values in a column using relations.

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
    if not _ensure_valid_table_key(table_key):
        return 0
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return 0
    if column not in relation.columns:
        return 0
    try:
        predicate = ColumnExpression(column).isnull()
        result = relation.filter(predicate).count("*").fetchone()
        if result is None:
            return 0
        return coerce_int(result[0], ctx=f"{table_key}.{column}.null_count")
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count nulls failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_min_value(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float | None:
    """Get minimum value in a numeric column using relations.

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
    if not _ensure_valid_table_key(table_key):
        return None
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return None
    if column not in relation.columns:
        return None
    try:
        expr = FunctionExpression("min", ColumnExpression(column)).alias("min_value")
        result = relation.aggregate(expr).fetchone()
        if result is None:
            return None
        return coerce_optional_float(result[0], ctx=f"{table_key}.{column}.min")
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Min value failed for %s.%s: %s", table_key, column, exc)
        return None


def safe_max_value(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float | None:
    """Get maximum value in a numeric column using relations.

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
    if not _ensure_valid_table_key(table_key):
        return None
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return None
    if column not in relation.columns:
        return None
    try:
        expr = FunctionExpression("max", ColumnExpression(column)).alias("max_value")
        result = relation.aggregate(expr).fetchone()
        if result is None:
            return None
        return coerce_optional_float(result[0], ctx=f"{table_key}.{column}.max")
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Max value failed for %s.%s: %s", table_key, column, exc)
        return None


def safe_count_non_positive(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count non-positive values in a numeric column using relations.

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
    if not _ensure_valid_table_key(table_key):
        return 0
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return 0
    if column not in relation.columns:
        return 0
    try:
        predicate = ColumnExpression(column) <= ConstantExpression("0")
        result = relation.filter(predicate).count("*").fetchone()
        if result is None:
            return 0
        return coerce_int(result[0], ctx=f"{table_key}.{column}.non_positive_count")
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count non-positive failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_count_duplicates(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> int:
    """Count duplicate values in a column using relations.

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
    if not _ensure_valid_table_key(table_key):
        return 0
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None:
        return 0
    if column not in relation.columns:
        return 0
    try:
        non_null = relation.filter(~ColumnExpression(column).isnull())
        total_result = non_null.count("*").fetchone()
        distinct = non_null.distinct().count("*").fetchone()
        if total_result is None or distinct is None:
            return 0
        total = coerce_int(total_result[0], ctx=f"{table_key}.{column}.non_null_count")
        distinct_count = coerce_int(distinct[0], ctx=f"{table_key}.{column}.distinct_count")
        return total - distinct_count
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Count duplicates failed for %s.%s: %s", table_key, column, exc)
        return 0


def safe_not_null_fraction(
    gateway: StorageGateway,
    table_key: str,
    column: str,
) -> float:
    """Get fraction of non-null values in a column using relations.

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
    if not _ensure_valid_table_key(table_key):
        return 0.0
    relation = _relation_for_table_key(gateway, table_key)
    if relation is None or column not in relation.columns:
        return 0.0
    fraction = 0.0
    try:
        total_result = relation.count("*").fetchone()
        if total_result is not None:
            total = coerce_int(total_result[0], ctx=f"{table_key}.count()")
            if total:
                non_null = relation.filter(~ColumnExpression(column).isnull())
                non_null_result = non_null.count("*").fetchone()
                if non_null_result is not None:
                    non_null_count = coerce_int(
                        non_null_result[0], ctx=f"{table_key}.{column}.non_null_count"
                    )
                    fraction = float(non_null_count) / float(total)
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug("Not null fraction failed for %s.%s: %s", table_key, column, exc)
    return fraction


@dataclass(frozen=True, slots=True)
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
    """Count orphaned foreign key references using relations.

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
    if not _ensure_valid_table_key(fk.source_table):
        return 0
    if not _ensure_valid_table_key(fk.ref_table):
        return 0
    src = _relation_for_table_key(gateway, fk.source_table)
    tgt = _relation_for_table_key(gateway, fk.ref_table)
    if src is None or tgt is None:
        return 0
    if fk.source_column not in src.columns or fk.ref_column not in tgt.columns:
        return 0
    src_col = fk.source_column
    tgt_col = fk.ref_column
    orphan_count = 0
    try:
        src_relation = src.set_alias("src")
        tgt_relation = tgt.set_alias("tgt")
        join_condition = ColumnExpression(f"src.{src_col}") == ColumnExpression(f"tgt.{tgt_col}")
        joined = src_relation.join(tgt_relation, join_condition, how="left")
        predicate = ColumnExpression(f"tgt.{tgt_col}").isnull()
        if not fk.allow_null:
            predicate &= ColumnExpression(f"src.{src_col}").isnotnull()
        result = joined.filter(predicate).count("*").fetchone()
        if result is not None:
            orphan_count = coerce_int(
                result[0],
                ctx=f"{fk.source_table}.{fk.source_column}.orphan_count",
            )
    except DUCKDB_QUERY_ERRORS as exc:
        log.debug(
            "Count orphan refs failed for %s.%s -> %s.%s: %s",
            fk.source_table,
            fk.source_column,
            fk.ref_table,
            fk.ref_column,
            exc,
        )
    return orphan_count


def _snapshot_filter_expression(*, repo: str, commit: str) -> Expression:
    filters = (
        FilterSpecInput(column="repo", op="eq", value=repo),
        FilterSpecInput(column="commit", op="eq", value=commit),
    )
    predicates = compile_filter_predicates(
        filters,
        allowed_columns=_SNAPSHOT_FILTER_COLUMNS,
    )
    expression = duckdb_filter_expression(predicates)
    if expression is None:
        msg = "Snapshot filter compilation returned empty predicate"
        raise FilterCompilerError(msg)
    return expression


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
