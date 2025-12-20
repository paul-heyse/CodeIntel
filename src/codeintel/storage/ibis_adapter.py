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
from typing import TYPE_CHECKING, ClassVar

import ibis
import ibis.expr.types as it
import pandas as pd
from sqlglot import exp

from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.staging import registered_temp_relation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
    from codeintel.storage.gateway.protocol import MinimalGateway

__all__ = ["IbisGateway", "OnConflict", "WriteResult"]

log = logging.getLogger(__name__)

_DATAFRAME_FAST_LANE_MIN_ROWS = 10_000


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
        Method used: 'insert_select', 'upsert_select', 'insert_values', 'upsert'.
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

    ADAPTER_NAME: ClassVar[str] = "ibis_gateway"

    def __init__(self, gateway: MinimalGateway) -> None:
        self._gateway = gateway
        self._ibis_con: DuckDBBackend | None = None

    def initialize(self) -> None:
        """Initialize the adapter (no-op, connection is lazy via property)."""

    def close(self) -> None:
        """Close the adapter by clearing the cached Ibis connection."""
        self._ibis_con = None

    @property
    def is_available(self) -> bool:
        """Check if adapter is available.

        Returns
        -------
        bool
            True if gateway is available.
        """
        return self._gateway is not None

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
            database, name = split_table_key(table_name)
            return self.con.table(name, database=database)
        return self.con.table(table_name)

    def to_sqlglot(
        self,
        expr: it.Expr,
        *,
        params: Mapping[it.Expr, object] | None = None,
        limit: int | None = None,
    ) -> exp.Expression:
        """Compile an Ibis expression to a SQLGlot AST.

        Parameters
        ----------
        expr
            Ibis expression to compile.
        params
            Optional scalar parameter bindings for expressions containing
            ``ibis.param(...)``.
        limit
            Optional limit applied at compile time.

        Returns
        -------
        sqlglot.expressions.Expression
            SQLGlot AST for the expression.
        """
        limit_arg = str(limit) if limit is not None else None
        return self.con.compiler.to_sqlglot(expr, limit=limit_arg, params=params)

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

        select_ast = self.to_sqlglot(expr)

        if on_conflict is not None:
            backend = self._policy
            backend.upsert_select(
                write_ctx.table_key,
                columns=resolved_columns,
                select_sql=select_ast,
                conflict_columns=on_conflict.conflict_columns,
                update_columns=on_conflict.update_columns,
            )
            return WriteResult(
                table_key=write_ctx.table_key,
                rows_affected=-1,
                method="upsert_select",
            )

        backend = self._policy
        backend.insert_select(
            write_ctx.table_key,
            columns=resolved_columns,
            select_sql=select_ast,
        )

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
        backend = self._policy

        row_count = len(df)
        if row_count == 0:
            return WriteResult(table_key=write_ctx.table_key, rows_affected=0, method="noop")

        if row_count >= _DATAFRAME_FAST_LANE_MIN_ROWS:
            return self._write_dataframe_via_relation(
                write_ctx,
                df=df,
                columns=resolved_columns,
                on_conflict=on_conflict,
            )

        normalized_rows = [_normalize_row(row) for row in df.itertuples(index=False, name=None)]
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

    def _write_dataframe_via_relation(
        self,
        write_ctx: _WriteContext,
        *,
        df: pd.DataFrame,
        columns: Sequence[str],
        on_conflict: OnConflict | None,
    ) -> WriteResult:
        with registered_temp_relation(self._gateway.con, df, prefix="ci_df_") as temp_name:
            select_expr = exp.Select(
                expressions=[exp.Column(this=exp.to_identifier(col)) for col in columns],
            ).from_(exp.Table(this=exp.to_identifier(temp_name)))
            backend = self._policy
            if on_conflict is None:
                backend.insert_select(
                    write_ctx.table_key,
                    columns=columns,
                    select_sql=select_expr,
                )
                return WriteResult(
                    table_key=write_ctx.table_key,
                    rows_affected=len(df),
                    method="insert_select",
                )

            backend.upsert_select(
                write_ctx.table_key,
                columns=columns,
                select_sql=select_expr,
                conflict_columns=on_conflict.conflict_columns,
                update_columns=on_conflict.update_columns,
            )
            return WriteResult(
                table_key=write_ctx.table_key,
                rows_affected=-1,
                method="upsert_select",
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
            select_ast = self.to_sqlglot(filter_expr)
            where_ast = _extract_top_level_where(select_ast)
            if where_ast is None:
                message = "Unable to derive WHERE clause for delete()"
                raise ValueError(message)
            backend.delete(table_key, where=where_ast)

        return -1


def _extract_top_level_where(select_ast: exp.Expression) -> exp.Where | None:
    """Return the top-level WHERE clause for a compiled query AST.

    Parameters
    ----------
    select_ast
        SQLGlot AST root for a compiled SELECT query (may be wrapped in WITH).

    Returns
    -------
    sqlglot.expressions.Where | None
        Top-level WHERE clause when present, otherwise None.
    """
    query_ast = select_ast
    if isinstance(query_ast, exp.With):
        query_ast = query_ast.this
    if isinstance(query_ast, exp.Subquery):
        query_ast = query_ast.this

    if isinstance(query_ast, exp.Select):
        where_ast = query_ast.args.get("where")
        return where_ast if isinstance(where_ast, exp.Where) else None

    select_node = query_ast.find(exp.Select)
    if not isinstance(select_node, exp.Select):
        return None
    where_ast = select_node.args.get("where")
    return where_ast if isinstance(where_ast, exp.Where) else None
