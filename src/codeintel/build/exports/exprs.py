"""Export expression helpers for Ibis.

The export subsystem writes JSONL and Parquet files that are validated against
generated JSON Schemas from TableSchema definitions. Those schemas are
consumer-facing and intentionally differ from the storage layer's physical
DuckDB types (e.g., GOIDs stored as DECIMAL are exported as integers).

This module builds Ibis expressions that normalize storage types to match the
export schemas without relying on DuckDB macros.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis

from codeintel.build.schemas import get_contract_for_table_key
from codeintel.storage.gateway import ibis_facade

if TYPE_CHECKING:
    import ibis.expr.types as it

    from codeintel.storage.gateway import StorageGateway

DUCKDB_DIALECT = "duckdb"


def build_export_expr(
    gateway: StorageGateway,
    table_key: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> it.Table:
    """Return an Ibis expression normalized for export.

    Normalization rules
    -------------------
    - Columns whose names contain ``goid_h128`` are cast to int64 (BIGINT).
    - TIMESTAMP/TIMESTAMPTZ columns are cast to string (ISO-like text) so that
      Parquet validation (pyarrow -> python types) matches the JSON Schemas.
    - Column order follows the DatasetContract TableSchema when available.

    Parameters
    ----------
    gateway
        StorageGateway used for Ibis table access.
    table_key
        Fully qualified table/view name (schema.table).
    limit
        Optional limit applied to the expression.
    offset
        Optional offset applied when a limit is provided.

    Returns
    -------
    it.Table
        Normalized table expression suitable for exporting.
    """
    base = ibis_facade.table(gateway, table_key)
    expr = base

    if limit is not None or offset:
        limit_value = limit if limit is not None else 9_223_372_036_854_775_807
        expr = expr.limit(limit_value, offset=offset)

    try:
        contract = get_contract_for_table_key(table_key)
        schema = contract.schema
    except KeyError:
        contract = None
        schema = None
    if schema is not None:
        column_order = [col.name for col in schema.columns]
    else:
        column_order = list(expr.columns)

    normalized: list[it.Value] = []
    for col_name in column_order:
        if col_name not in expr.columns:
            continue
        col_expr = expr[col_name]
        lower_name = col_name.lower()

        if "goid_h128" in lower_name:
            normalized.append(col_expr.cast("int64").name(col_name))
            continue

        if schema is not None:
            declared = next((col for col in schema.columns if col.name == col_name), None)
            if declared is not None and declared.type in {"TIMESTAMP", "TIMESTAMPTZ"}:
                normalized.append(col_expr.cast("string").name(col_name))
                continue

        normalized.append(col_expr)

    return expr.select(*normalized)


def compile_export_sql(expr: it.Table) -> str:
    """Compile an export expression to DuckDB SQL.

    Parameters
    ----------
    expr
        Ibis table expression.

    Returns
    -------
    str
        DuckDB SQL string.
    """
    return ibis.to_sql(expr, dialect=DUCKDB_DIALECT)


__all__ = [
    "DUCKDB_DIALECT",
    "build_export_expr",
    "compile_export_sql",
]
