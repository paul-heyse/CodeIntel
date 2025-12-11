"""Ibis adapter for DuckDB-backed storage gateways.

This module provides a unified data access layer combining:
- Ibis expressions for type-safe, composable query building
- SQLGlot for SQL generation (INSERT, UPSERT, DDL)
- DuckDB as the execution backend

The key abstraction is `IbisGateway.write()` which accepts:
- Ibis table expressions → INSERT...SELECT via SQLGlot
- pandas DataFrames → INSERT...VALUES via SQLGlot
- Sequences of tuples → INSERT...VALUES via SQLGlot

All SQL is generated programmatically - no raw SQL strings in application code.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import ibis
import ibis.expr.types as it
import pandas as pd
from sqlglot import exp, parse_one

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["IbisGateway", "OnConflict", "WriteResult"]

log = logging.getLogger(__name__)

# DuckDB dialect for SQLGlot
DUCKDB_DIALECT = "duckdb"


def _convert_numpy_types(value: object) -> object:
    """Convert numpy types to native Python types for DuckDB compatibility.

    Returns
    -------
    object
        Native Python type or original value if not a numpy type.
    """
    # Handle numpy integer types
    item = getattr(value, "item", None)
    dtype = getattr(value, "dtype", None)
    if callable(item) and dtype is not None:
        # This is a numpy scalar - convert to native Python type
        return item()
    return value


def _normalize_row(row: tuple[object, ...]) -> tuple[object, ...]:
    """Normalize a row by converting numpy types to native Python types.

    Returns
    -------
    tuple[object, ...]
        Row with all numpy types converted to native Python types.
    """
    return tuple(_convert_numpy_types(v) for v in row)


def _extract_scalar(value: object) -> object:
    """
    Extract a single scalar value from common ibis execute() results.

    Returns
    -------
    object
        Native Python scalar extracted from the result.

    Raises
    ------
    ValueError
        If the result cannot be reduced to a single scalar.
    """
    if isinstance(value, pd.DataFrame):
        if value.empty or value.shape[1] != 1:
            msg = "Expected single-column DataFrame result for scalar execution"
            raise ValueError(msg)
        return _convert_numpy_types(value.iloc[0, 0])

    if isinstance(value, pd.Series):
        if value.empty:
            msg = "Expected non-empty Series result for scalar execution"
            raise ValueError(msg)
        return _convert_numpy_types(value.iloc[0])

    return _convert_numpy_types(value)


@dataclass(frozen=True)
class OnConflict:
    """Specification for UPSERT behavior on conflict.

    Parameters
    ----------
    conflict_columns
        Column(s) that define the uniqueness constraint.
    update_columns
        Column(s) to update on conflict. If None, updates all non-conflict columns.
    """

    conflict_columns: Sequence[str]
    update_columns: Sequence[str] | None = None


@dataclass(frozen=True)
class WriteResult:
    """Result of a write operation.

    Parameters
    ----------
    table_key
        Target table (schema.table format).
    rows_affected
        Number of rows written or affected.
    method
        Method used: 'insert_select', 'insert_values', 'upsert'.
    """

    table_key: str
    rows_affected: int
    method: str


@dataclass(frozen=True)
class _WriteContext:
    """Internal context for write operations."""

    schema: str
    table: str
    table_key: str
    columns: Sequence[str]
    on_conflict: OnConflict | None


def _split_table_key(table_key: str) -> tuple[str, str]:
    """Split 'schema.table' into (schema, table) components.

    Returns
    -------
    tuple[str, str]
        Schema and table name.

    Raises
    ------
    ValueError
        If table_key is not schema-qualified.
    """
    if "." not in table_key:
        msg = f"Table key must be schema-qualified: {table_key}"
        raise ValueError(msg)
    schema, table = table_key.split(".", 1)
    return schema, table


def _build_insert_select(
    schema: str,
    table: str,
    columns: Sequence[str],
    select_sql: str,
) -> str:
    """Build INSERT...SELECT statement using SQLGlot.

    Parameters
    ----------
    schema
        Target schema name.
    table
        Target table name.
    columns
        Column names for the INSERT.
    select_sql
        SQL SELECT statement to insert from.

    Returns
    -------
    str
        Generated INSERT...SELECT SQL.
    """
    # Parse the SELECT statement
    select_ast = parse_one(select_sql, dialect=DUCKDB_DIALECT)

    # Build INSERT...SELECT
    insert_stmt = exp.Insert(
        this=exp.Schema(
            this=exp.Table(
                this=exp.to_identifier(table),
                db=exp.to_identifier(schema),
            ),
            expressions=[exp.to_identifier(c) for c in columns],
        ),
        expression=select_ast,
    )

    return insert_stmt.sql(dialect=DUCKDB_DIALECT)


def _build_insert_values(
    schema: str,
    table: str,
    columns: Sequence[str],
) -> str:
    """Build INSERT...VALUES statement with placeholders using SQLGlot.

    Parameters
    ----------
    schema
        Target schema name.
    table
        Target table name.
    columns
        Column names for the INSERT.

    Returns
    -------
    str
        Generated INSERT...VALUES SQL with ? placeholders.
    """
    insert_stmt = exp.Insert(
        this=exp.Schema(
            this=exp.Table(
                this=exp.to_identifier(table),
                db=exp.to_identifier(schema),
            ),
            expressions=[exp.to_identifier(c) for c in columns],
        ),
        expression=exp.Values(
            expressions=[exp.Tuple(expressions=[exp.Placeholder() for _ in columns])]
        ),
    )

    return insert_stmt.sql(dialect=DUCKDB_DIALECT)


def _build_upsert(
    schema: str,
    table: str,
    columns: Sequence[str],
    conflict_columns: Sequence[str],
    update_columns: Sequence[str],
) -> str:
    """Build INSERT...ON CONFLICT DO UPDATE statement.

    Parameters
    ----------
    schema
        Target schema name.
    table
        Target table name.
    columns
        All column names for the INSERT.
    conflict_columns
        Columns that define uniqueness for conflict detection.
    update_columns
        Columns to update on conflict.

    Returns
    -------
    str
        Generated UPSERT SQL with ? placeholders.
    """
    # Build the base INSERT
    quoted_schema = f'"{schema}"'
    quoted_table = f'"{table}"'
    quoted_cols = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join("?" for _ in columns)

    # Build ON CONFLICT clause
    conflict_cols = ", ".join(f'"{c}"' for c in conflict_columns)
    update_parts = [f'"{c}" = EXCLUDED."{c}"' for c in update_columns]
    update_clause = ", ".join(update_parts)

    # Construct full UPSERT - DuckDB uses INSERT...ON CONFLICT syntax
    return (
        f"INSERT INTO {quoted_schema}.{quoted_table} ({quoted_cols}) "  # noqa: S608
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}"
    )


class IbisGateway:
    """Expose an Ibis backend bound to a `StorageGateway`.

    This class provides a unified data access API:
    - `table()` / `view()`: Start Ibis read expressions
    - `write()`: Write data using Ibis + SQLGlot (INSERT, UPSERT)
    - `con`: Access underlying Ibis connection for advanced operations
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway

    @cached_property
    def con(self) -> DuckDBBackend:
        """
        Return an Ibis backend that reuses the gateway DuckDB connection.

        Returns
        -------
        DuckDBBackend
            Ibis backend bound to the DuckDB connection.
        """
        return ibis.duckdb.from_connection(self._gateway.con)

    def table(self, table_name: str) -> it.Table:
        """
        Return an Ibis table expression for a fully qualified table.

        Parameters
        ----------
        table_name
            Fully qualified table or view name (e.g., "analytics.function_metrics").

        Returns
        -------
        it.Table
            Ibis table expression for the requested object.

        Note
        ----
        Ibis 11+ requires the `database` parameter for schema-qualified names.
        This method automatically splits "schema.table" into the correct form.
        """
        if "." in table_name:
            database, name = table_name.split(".", 1)
            return self.con.table(name, database=database)
        return self.con.table(table_name)

    def read(self, table_name: str) -> it.Table:
        """
        Return a table expression (alias for `table`) to standardize reads.

        Parameters
        ----------
        table_name
            Fully qualified table or view name.

        Returns
        -------
        it.Table
            Ibis table expression for the requested object.
        """
        return self.table(table_name)

    def view(self, view_name: str) -> it.Table:
        """
        Alias for `table` for semantic clarity when accessing views.

        Parameters
        ----------
        view_name
            Fully qualified view name.

        Returns
        -------
        it.Table
            Ibis table expression for the view.
        """
        return self.table(view_name)

    def sql(self, raw_sql: str) -> it.Table:
        """
        Execute raw SQL through Ibis and return the resulting table expression.

        Parameters
        ----------
        raw_sql
            SQL string to execute via Ibis.

        Returns
        -------
        it.Table
            Table expression backed by the SQL statement.

        Note
        ----
        Prefer using `table()` and Ibis expressions over raw SQL.
        This method exists for compatibility and edge cases.
        """
        return self.con.sql(raw_sql)

    @staticmethod
    def execute_scalar(expr: it.Scalar | it.Table) -> object:
        """
        Execute an Ibis scalar or single-value table and return a Python scalar.

        Parameters
        ----------
        expr
            Ibis scalar expression or table that yields a single value (e.g., count()).

        Returns
        -------
        Any
            Native Python scalar extracted from the execution result.

        Raises
        ------
        ValueError
            If the execution result cannot be reduced to a single scalar.
        """
        result = expr.execute()
        try:
            return _extract_scalar(result)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    def write(
        self,
        table_key: str,
        data: it.Table | pd.DataFrame | Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None = None,
        on_conflict: OnConflict | None = None,
    ) -> WriteResult:
        """Write data to a table using Ibis + SQLGlot.

        This is the unified write API that handles:
        - Ibis expressions → INSERT...SELECT
        - DataFrames → INSERT...VALUES (batch)
        - Tuples → INSERT...VALUES (batch)
        - UPSERT semantics via on_conflict

        Parameters
        ----------
        table_key
            Target table in 'schema.table' format.
        data
            Data to write. Can be:
            - Ibis Table expression (generates INSERT...SELECT)
            - pandas DataFrame (generates INSERT...VALUES)
            - Sequence of tuples (generates INSERT...VALUES)
        columns
            Column names. Required for tuples, optional for DataFrame/Ibis
            (will be inferred from data if not provided).
        on_conflict
            UPSERT specification. If provided, generates INSERT...ON CONFLICT.

        Returns
        -------
        WriteResult
            Result containing rows affected and method used.

        Raises
        ------
        TypeError
            If data type is not supported.
        """
        schema, table = _split_table_key(table_key)  # Raises ValueError if invalid
        write_ctx = _WriteContext(schema=schema, table=table, table_key=table_key)

        # Dispatch based on data type
        if isinstance(data, it.Table):
            return self._write_ibis_expression(
                write_ctx, expr=data, columns=columns, on_conflict=on_conflict
            )
        if isinstance(data, pd.DataFrame):
            return self._write_dataframe(write_ctx, df=data, columns=columns, on_conflict=on_conflict)
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return self._write_tuples(
                write_ctx,
                rows=data,
                columns=columns,
                on_conflict=on_conflict,
            )

        msg = f"Unsupported data type for write: {type(data).__name__}"
        raise TypeError(msg)

    def insert(
        self,
        table_key: str,
        data: pd.DataFrame | Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None = None,
    ) -> WriteResult:
        """
        Insert rows into a table (wrapper over write for clarity).

        Parameters
        ----------
        table_key
            Target table in 'schema.table' format.
        data
            DataFrame or sequence of tuples to insert.
        columns
            Optional column list (required for tuples).

        Returns
        -------
        WriteResult
            Result containing rows affected and method used.
        """
        return self.write(table_key, data, columns=columns)

    def upsert(
        self,
        table_key: str,
        data: pd.DataFrame | Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None,
        conflict_columns: Sequence[str],
        update_columns: Sequence[str],
    ) -> WriteResult:
        """
        Insert-or-update rows using ON CONFLICT semantics.

        Parameters
        ----------
        table_key
            Target table in 'schema.table' format.
        data
            DataFrame or sequence of tuples to upsert.
        columns
            Column names to write.
        conflict_columns
            Columns defining the uniqueness constraint.
        update_columns
            Columns to update when a conflict occurs.

        Returns
        -------
        WriteResult
            Result containing rows affected and method used.
        """
        on_conflict = OnConflict(conflict_columns=conflict_columns, update_columns=update_columns)
        return self.write(table_key, data, columns=columns, on_conflict=on_conflict)

    def _write_ibis_expression(
        self,
        write_ctx: _WriteContext,
        *,
        expr: it.Table,
        columns: Sequence[str] | None,
        on_conflict: OnConflict | None,
    ) -> WriteResult:
        """Write an Ibis expression using INSERT...SELECT.

        Returns
        -------
        WriteResult
            Write operation result.
        """
        # Get column names from the expression if not provided
        resolved_columns = list(expr.columns) if columns is None else list(columns)

        # Generate SQL from Ibis expression
        select_sql = ibis.to_sql(expr, dialect=DUCKDB_DIALECT)

        if on_conflict is not None:
            # For UPSERT with Ibis expression, materialize to DataFrame first
            log.warning("UPSERT with Ibis expression not yet optimized; using temp table")
            df = expr.to_pandas()
            return self._write_dataframe(
                write_ctx, df=df, columns=resolved_columns, on_conflict=on_conflict
            )

        # Build INSERT...SELECT
        insert_sql = _build_insert_select(write_ctx.schema, write_ctx.table, resolved_columns, select_sql)
        log.debug("write INSERT...SELECT: %s", insert_sql[:200])

        # Execute
        self._gateway.con.execute(insert_sql)

        return WriteResult(table_key=write_ctx.table_key, rows_affected=-1, method="insert_select")

    def _write_dataframe(
        self,
        write_ctx: _WriteContext,
        *,
        df: pd.DataFrame,
        columns: Sequence[str] | None,
        on_conflict: OnConflict | None,
    ) -> WriteResult:
        """Write a DataFrame using INSERT...VALUES or UPSERT.

        Returns
        -------
        WriteResult
            Write operation result.
        """
        resolved_columns = list(df.columns) if columns is None else list(columns)
        rows = list(df.itertuples(index=False, name=None))
        return self._write_tuples(
            write_ctx, rows=rows, columns=resolved_columns, on_conflict=on_conflict
        )

    def _write_tuples(
        self,
        write_ctx: _WriteContext,
        *,
        rows: Sequence[tuple[object, ...]],
        columns: Sequence[str] | None,
        on_conflict: OnConflict | None,
    ) -> WriteResult:
        """Write tuples using INSERT...VALUES or UPSERT.

        Returns
        -------
        WriteResult
            Write operation result.

        Raises
        ------
        ValueError
            If columns is None when writing tuples.
        """
        if not rows:
            return WriteResult(table_key=write_ctx.table_key, rows_affected=0, method="noop")

        if columns is None:
            msg = "columns must be provided when writing tuples"
            raise ValueError(msg)

        resolved_columns = list(columns)

        if on_conflict is not None:
            return self._upsert_tuples(
                write_ctx,
                rows=rows,
                columns=resolved_columns,
                on_conflict=on_conflict,
            )

        # Normalize rows to convert numpy types to native Python types
        normalized_rows = [_normalize_row(row) for row in rows]

        # Build INSERT...VALUES
        insert_sql = _build_insert_values(write_ctx.schema, write_ctx.table, resolved_columns)
        log.debug("write INSERT...VALUES: %s (%d rows)", insert_sql[:100], len(normalized_rows))

        # Execute batch insert
        self._gateway.con.executemany(insert_sql, normalized_rows)

        return WriteResult(
            table_key=write_ctx.table_key, rows_affected=len(normalized_rows), method="insert_values"
        )

    def _upsert_tuples(
        self,
        write_ctx: _WriteContext,
        *,
        rows: Sequence[tuple[object, ...]],
        columns: Sequence[str],
        on_conflict: OnConflict,
    ) -> WriteResult:
        """Write tuples using UPSERT (INSERT...ON CONFLICT).

        Returns
        -------
        WriteResult
            Write operation result.

        Raises
        ------
        ValueError
            If no columns remain to update after conflict columns.
        """
        conflict_set = set(on_conflict.conflict_columns)
        update_columns = (
            list(on_conflict.update_columns)
            if on_conflict.update_columns is not None
            else [c for c in columns if c not in conflict_set]
        )

        if not update_columns:
            msg = "No columns to update on conflict"
            raise ValueError(msg)

        # Normalize rows to convert numpy types to native Python types
        normalized_rows = [_normalize_row(row) for row in rows]

        # Build UPSERT SQL
        upsert_sql = _build_upsert(
            write_ctx.schema, write_ctx.table, columns, on_conflict.conflict_columns, update_columns
        )
        log.debug("write UPSERT: %s (%d rows)", upsert_sql[:100], len(normalized_rows))

        # Execute batch upsert
        self._gateway.con.executemany(upsert_sql, normalized_rows)

        return WriteResult(
            table_key=write_ctx.table_key, rows_affected=len(normalized_rows), method="upsert"
        )

    def delete(
        self,
        table_key: str,
        *,
        where: it.BooleanValue | None = None,
    ) -> int:
        """Delete rows from a table.

        Parameters
        ----------
        table_key
            Target table in 'schema.table' format.
        where
            Ibis boolean expression for the WHERE clause.
            If None, deletes all rows (use with caution).

        Returns
        -------
        int
            Number of rows deleted (estimated, may be -1 if unknown).
        """
        schema, table = _split_table_key(table_key)
        quoted = f'"{schema}"."{table}"'

        if where is None:
            # Delete all
            sql = f"DELETE FROM {quoted}"  # noqa: S608
        else:
            # Build WHERE clause from Ibis expression
            # Get the SQL representation of the boolean expression
            # We need to wrap it in a SELECT to get valid SQL
            t = self.table(table_key)
            filter_expr = t.filter(where).limit(0)
            filter_sql = ibis.to_sql(filter_expr, dialect=DUCKDB_DIALECT)

            # Extract WHERE clause from the generated SQL
            # This is a bit hacky but works for simple expressions
            if "WHERE" in filter_sql.upper():
                where_idx = filter_sql.upper().index("WHERE")
                # Find the end of WHERE clause (before LIMIT)
                limit_idx = filter_sql.upper().find("LIMIT", where_idx)
                if limit_idx > 0:
                    where_clause = filter_sql[where_idx:limit_idx].strip()
                else:
                    where_clause = filter_sql[where_idx:].strip()
                sql = f"DELETE FROM {quoted} {where_clause}"  # noqa: S608
            else:
                sql = f"DELETE FROM {quoted}"  # noqa: S608

        log.debug("delete: %s", sql[:200])
        self._gateway.con.execute(sql)

        return -1  # DuckDB doesn't return affected count easily
