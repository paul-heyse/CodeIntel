"""Export expression helpers for DuckDB SQL generation.

The export subsystem writes JSONL and Parquet files that are validated against
generated JSON Schemas from TableSchema definitions. Those schemas are
consumer-facing and intentionally differ from the storage layer's physical
DuckDB types (e.g., GOIDs stored as DECIMAL are exported as integers).

This module builds SQLGlot expressions that normalize storage types to match the
export schemas without relying on DuckDB macros or Ibis expressions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.core.schemas.contract_service import (
    column_order_for_table_key,
    get_contract_for_table_key,
)
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.sqlglot_tools import render_sql_duckdb

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def build_export_expr(
    gateway: StorageGateway,
    table_key: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> exp.Select:
    """Return a SQLGlot expression normalized for export.

    Normalization rules
    -------------------
    - Columns whose names contain ``goid_h128`` are cast to int64 (BIGINT).
    - TIMESTAMP/TIMESTAMPTZ columns are cast to string (ISO-like text) so that
      Parquet validation (pyarrow -> python types) matches the JSON Schemas.
    - Column order follows the DatasetContract TableSchema when available.

    Parameters
    ----------
    gateway
        StorageGateway used for DuckDB relation access.
    table_key
        Fully qualified table/view name (schema.table).
    limit
        Optional limit applied to the expression.
    offset
        Optional offset applied when a limit is provided.

    Returns
    -------
    exp.Select
        Normalized SQLGlot expression suitable for exporting.
    """
    try:
        schema = get_contract_for_table_key(table_key).schema
    except KeyError:
        schema = None
    relation_columns = tuple(_relation_columns(gateway, table_key))
    if schema is not None:
        column_order = list(column_order_for_table_key(table_key))
    else:
        column_order = list(relation_columns)

    existing_columns = set(relation_columns)
    select_exprs: list[exp.Expression] = []

    for col_name in column_order:
        if col_name not in existing_columns:
            continue
        col_expr = exp.Column(this=exp.to_identifier(col_name))
        lower_name = col_name.lower()

        if "goid_h128" in lower_name:
            select_exprs.append(_cast_column(col_expr, col_name, "BIGINT"))
            continue

        if schema is not None:
            declared = next((col for col in schema.columns if col.name == col_name), None)
            if declared is not None and declared.type in {"TIMESTAMP", "TIMESTAMPTZ"}:
                select_exprs.append(_cast_column(col_expr, col_name, "VARCHAR"))
                continue

        select_exprs.append(col_expr)

    schema_name, table_name = split_table_key(table_key)
    table_expr = exp.Table(
        this=exp.to_identifier(table_name),
        db=exp.to_identifier(schema_name),
    )
    expr = exp.select(*select_exprs).from_(table_expr)
    if limit is not None or offset:
        limit_value = limit if limit is not None else 9_223_372_036_854_775_807
        expr = expr.limit(limit_value)
        if offset:
            expr = expr.offset(offset)

    return expr


def compile_export_sql(expr: exp.Expression) -> str:
    """Compile an export expression to DuckDB SQL.

    Parameters
    ----------
    expr
        SQLGlot expression to compile.

    Returns
    -------
    str
        DuckDB SQL string.
    """
    return render_sql_duckdb(expr)


def _relation_columns(gateway: StorageGateway, table_key: str) -> Iterable[str]:
    relation = gateway.relation_from_table_key(table_key)
    return relation.columns


def _cast_column(
    col_expr: exp.Expression,
    col_name: str,
    type_name: str,
) -> exp.Expression:
    return exp.alias_(
        exp.Cast(this=col_expr, to=exp.DataType.build(type_name)),
        col_name,
        quoted=True,
    )


__all__ = [
    "DUCKDB_DIALECT",
    "build_export_expr",
    "compile_export_sql",
]
