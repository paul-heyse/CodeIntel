"""DuckDB helpers for bulk row insertion.

This module provides the canonical bulk insertion function for DuckDB tables.
For row count operations, use codeintel.storage.validation.data_checks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.errors import DUCKDB_ERRORS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from duckdb import DuckDBPyConnection

__all__ = [
    "DUCKDB_ERRORS",
    "macro_insert_rows",
]


def macro_insert_rows(
    con: DuckDBPyConnection,
    table_key: str,
    rows: Iterable[Sequence[object]],
) -> None:
    """Insert rows into a table using schema-driven column order.

    This is the canonical method for bulk row insertion into DuckDB tables.
    It uses the DatasetContract schema as the source of truth for column
    order, automatically padding missing trailing columns with NULL values.

    The function creates a temporary table to stage the data, then inserts
    it into the target table. This approach ensures type safety and handles
    schema evolution gracefully.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table name (schema.table).
    rows
        Iterable of row tuples. Each tuple should contain values in the order
        defined by the table's DatasetContract schema. Trailing columns may
        be omitted and will be padded with NULL.

    Raises
    ------
    ValueError
        If the table name is invalid or rows exceed the column count.
        If the table has no DatasetContract schema.

    Notes
    -----
    This function is used internally by the table accessor classes
    (CoreTables, GraphTables, AnalyticsTables) via the BaseTableAccessor
    base class. For direct bulk insertion, prefer using accessor methods
    like ``gateway.core.insert_goids(rows)`` which provide typed signatures.

    See Also
    --------
    codeintel.storage.gateway.base_accessor.BaseTableAccessor._insert_rows
        Base accessor method that calls this function.
    codeintel.storage.sql.builder.prepared_statements_dynamic
        Alternative approach for generating parameterized SQL.
    """
    rows_list = list(rows)
    if not rows_list:
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", table_key):
        message = f"Invalid table key: {table_key}"
        raise ValueError(message)

    contract = get_dataset_contracts_by_table_key().get(table_key)
    if contract is None or contract.schema is None:
        message = f"Cannot insert into {table_key}: missing DatasetContract schema"
        raise ValueError(message)
    columns = [col.name for col in contract.schema.columns if col.name is not None]
    schema_name, table_name = table_key.split(".", maxsplit=1)
    col_count = len(columns)
    normalized: list[tuple[object, ...]] = []
    for row in rows_list:
        if len(row) > col_count:
            message = f"Row for {table_key} has {len(row)} values, exceeds column count {col_count}"
            raise ValueError(message)
        padded = tuple(row) + (None,) * (col_count - len(row))
        normalized.append(padded)

    insert_sql = _build_insert_sql(schema_name, table_name, columns)
    con.executemany(insert_sql, normalized)


def _build_insert_sql(schema: str, table: str, columns: Sequence[str]) -> str:
    """Build an INSERT statement with placeholders using SQLGlot.

    Parameters
    ----------
    schema
        Target schema name.
    table
        Target table name.
    columns
        Columns to insert in order.

    Returns
    -------
    str
        DuckDB SQL string for executemany insertion.
    """
    table_expr = exp.Table(this=exp.to_identifier(table), db=exp.to_identifier(schema))
    schema_expr = exp.Schema(this=table_expr, expressions=[exp.to_identifier(c) for c in columns])
    placeholders = [exp.Placeholder() for _ in columns]
    values = exp.Values(expressions=[exp.Tuple(expressions=placeholders)])
    return exp.Insert(this=schema_expr, expression=values).sql(dialect="duckdb")
