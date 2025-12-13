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

Architecture Note
-----------------
IbisGateway only depends on the MinimalGateway protocol, NOT on DuckDBPolicyBackend
directly. It accesses the policy backend via gateway.policy, which avoids circular
imports. MinimalStorageGateway is the composition root that creates both.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ibis
import ibis.expr.types as it
import pandas as pd
from sqlglot import exp, parse_one

from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
    from codeintel.storage.gateway.protocol import MinimalGateway

__all__ = ["IbisGateway", "OnConflict", "WriteResult"]

log = logging.getLogger(__name__)


def _convert_numpy_types(value: object) -> object:
    """Convert numpy types to native Python types for DuckDB compatibility.

    Returns
    -------
    object
        Native Python type or original value if not a numpy type.
    """
    item = getattr(value, "item", None)
    dtype = getattr(value, "dtype", None)
    if callable(item) and dtype is not None:
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


class IbisGateway:
    """Expose an Ibis backend for DuckDB access.

    Accepts a MinimalGateway that provides connection and policy backend access.

    This class provides a unified data access API:
    - `table()` / `view()`: Start Ibis read expressions
    - `write()`: Write data using Ibis + SQLGlot (INSERT, UPSERT)
    - `con`: Access underlying Ibis connection for advanced operations

    Note
    ----
    IbisGateway accesses the policy backend via `gateway.policy` to avoid
    circular imports. The MinimalStorageGateway acts as composition root.
    """

    def __init__(self, gateway: MinimalGateway) -> None:
        self._gateway = gateway
        self._ibis_con: DuckDBBackend | None = None

    @property
    def _policy(self) -> DuckDBPolicyBackend:
        """Return the policy backend via the gateway.

        Accesses the policy backend through the gateway reference,
        avoiding direct import dependencies.
        """
        return self._gateway.policy

    @property
    def con(self) -> DuckDBBackend:
        """
        Return an Ibis backend that reuses the gateway DuckDB connection.

        Returns
        -------
        DuckDBBackend
            Ibis backend bound to the DuckDB connection.

        Raises
        ------
        RuntimeError
            If the DuckDB backend cannot be initialized.
        """
        if self._ibis_con is None:
            self._ibis_con = ibis.duckdb.from_connection(self._gateway.con)
        if self._ibis_con is None:
            msg = "Failed to initialize DuckDB backend connection"
            raise RuntimeError(msg)
        return self._ibis_con

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
        schema, table = split_table_key(table_key)
        write_ctx = _WriteContext(schema=schema, table=table, table_key=table_key)

        if isinstance(data, it.Table):
            return self._write_ibis_expression(
                write_ctx, expr=data, columns=columns, on_conflict=on_conflict
            )
        if isinstance(data, pd.DataFrame):
            return self._write_dataframe(
                write_ctx, df=data, columns=columns, on_conflict=on_conflict
            )
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
        resolved_columns = list(expr.columns) if columns is None else list(columns)

        select_sql = ibis.to_sql(expr, dialect=DUCKDB_DIALECT)

        if on_conflict is not None:
            log.warning("UPSERT with Ibis expression not yet optimized; using temp table")
            df = expr.to_pandas()
            return self._write_dataframe(
                write_ctx, df=df, columns=resolved_columns, on_conflict=on_conflict
            )

        backend = self._policy
        backend.insert_select(write_ctx.table_key, columns=resolved_columns, select_sql=select_sql)

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
        normalized_rows = [_normalize_row(row) for row in df.itertuples(index=False, name=None)]
        backend = self._policy

        if on_conflict is None:
            count = backend.bulk_insert(
                write_ctx.table_key, normalized_rows, columns=resolved_columns
            )
            return WriteResult(
                table_key=write_ctx.table_key,
                rows_affected=count,
                method="insert_values",
            )

        count = backend.upsert(
            write_ctx.table_key,
            normalized_rows,
            columns=resolved_columns,
            conflict_columns=on_conflict.conflict_columns,
            update_columns=on_conflict.update_columns,
        )
        return WriteResult(
            table_key=write_ctx.table_key,
            rows_affected=count,
            method="upsert",
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
        normalized_rows = [_normalize_row(row) for row in rows]
        backend = self._policy

        if on_conflict is None:
            count = backend.bulk_insert(
                write_ctx.table_key, normalized_rows, columns=resolved_columns
            )
            return WriteResult(
                table_key=write_ctx.table_key,
                rows_affected=count,
                method="insert_values",
            )

        count = backend.upsert(
            write_ctx.table_key,
            normalized_rows,
            columns=resolved_columns,
            conflict_columns=on_conflict.conflict_columns,
            update_columns=on_conflict.update_columns,
        )
        return WriteResult(table_key=write_ctx.table_key, rows_affected=count, method="upsert")

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

        Raises
        ------
        ValueError
            If the WHERE clause cannot be derived from the provided filter.
        """
        backend = self._policy

        if where is None:
            backend.delete(table_key)
        else:
            t = self.table(table_key)
            filter_expr = t.filter(where).limit(0)
            filter_sql = ibis.to_sql(filter_expr, dialect=DUCKDB_DIALECT)
            select_ast = parse_one(filter_sql, dialect=DUCKDB_DIALECT)
            where_ast = select_ast.args.get("where")
            if not isinstance(where_ast, exp.Where):
                message = "Unable to derive WHERE clause for delete()"
                raise ValueError(message)
            backend.delete(table_key, where=where_ast)

        return -1
