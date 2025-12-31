"""Export expression helpers for DuckDB relation plans.

The export subsystem writes JSONL and Parquet files that are validated against
generated JSON Schemas from TableSchema definitions. Those schemas are
consumer-facing and intentionally differ from the storage layer's physical
DuckDB types (e.g., GOIDs stored as DECIMAL are exported as integers).

This module builds DuckDB Expression API projections that normalize storage
types to match the export schemas without relying on raw SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

from codeintel.core.schemas.contract_service import (
    column_order_for_table_key,
    get_contract_for_table_key,
)
from codeintel.storage.duckdb_types import ColumnExpression, DuckDBRelation, Expression

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def build_export_relation_plan(
    gateway: StorageGateway,
    table_key: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> DuckDBRelation:
    """Return a DuckDB relation normalized for export.

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
    DuckDBRelation
        Normalized relation suitable for exporting.
    """
    relation = gateway.relation_from_table_key(table_key)
    try:
        schema = get_contract_for_table_key(table_key).schema
    except KeyError:
        schema = None
    relation_columns = tuple(relation.columns)
    if schema is not None:
        column_order = list(column_order_for_table_key(table_key))
    else:
        column_order = list(relation_columns)

    existing_columns = set(relation_columns)
    select_exprs: list[Expression] = []

    for col_name in column_order:
        if col_name not in existing_columns:
            continue
        col_expr = ColumnExpression(col_name)
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

    relation = relation.select(*select_exprs)
    if limit is not None or offset:
        limit_value = limit if limit is not None else 9_223_372_036_854_775_807
        relation = relation.limit(limit_value, offset=offset)

    return relation


def _cast_column(col_expr: Expression, col_name: str, type_name: str) -> Expression:
    cast_type = duckdb.sqltype(type_name)
    return col_expr.cast(cast_type).alias(col_name)


__all__ = [
    "build_export_relation_plan",
]
