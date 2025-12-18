"""Schema round-trip helpers (TableSchema ↔ Ibis ↔ SQLGlot).

This module provides the single bridge between the project's contract language
(`TableSchema`) and the libraries used to generate/execute SQL:

- `TableSchema` is the canonical contract representation.
- `ibis.Schema` provides typed schema semantics (including nullability).
- `sqlglot` expressions are used to render DDL in a dialect-safe way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import sqlglot.expressions as exp

from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from ibis.expr.datatypes import DataType

    from codeintel.core.schemas.primitives import TableSchema


def ibis_schema_from_table_schema(table: TableSchema) -> ibis.Schema:
    """Convert a `TableSchema` to an `ibis.Schema`.

    Parameters
    ----------
    table
        Canonical schema contract.

    Returns
    -------
    ibis.Schema
        Ibis schema, including nullability.

    Raises
    ------
    ValueError
        If the table contains an unknown column type.
    """
    dtype_map = {
        "BOOLEAN": ibis.dtype("boolean"),
        "INTEGER": ibis.dtype("int32"),
        "BIGINT": ibis.dtype("int64"),
        "DOUBLE": ibis.dtype("float64"),
        "VARCHAR": ibis.dtype("string"),
        "JSON": ibis.dtype("json"),
        "TIMESTAMP": ibis.dtype("timestamp"),
        "TIMESTAMPTZ": ibis.dtype('timestamp("UTC")'),
        "DECIMAL": ibis.dtype("decimal"),
        "DECIMAL(38,0)": ibis.dtype("decimal(38,0)"),
    }

    cols: dict[str, DataType] = {}
    for col in table.columns:
        dtype = dtype_map.get(col.type)
        if dtype is None:
            msg = f"Unsupported column type: {col.type}"
            raise ValueError(msg)
        if not col.nullable:
            dtype = dtype.copy(nullable=False)
        cols[col.name] = dtype

    return ibis.schema(cols)


def create_table_ast(table: TableSchema, *, if_not_exists: bool) -> exp.Create:
    """Build a DuckDB table-creation DDL AST from `TableSchema`.

    Parameters
    ----------
    table
        Canonical schema contract.
    if_not_exists
        When True, includes an "IF NOT EXISTS" guard.

    Returns
    -------
    sqlglot.expressions.Create
        SQLGlot AST for table creation.
    """
    ibis_schema = ibis_schema_from_table_schema(table)
    column_defs = ibis_schema.to_sqlglot_column_defs(dialect=DUCKDB_DIALECT)

    schema_expr = exp.Schema(
        this=exp.Table(
            this=exp.to_identifier(table.name),
            db=exp.to_identifier(table.schema),
        ),
        expressions=column_defs,
    )

    if table.primary_key:
        schema_expr.expressions.append(
            exp.PrimaryKey(expressions=[exp.to_identifier(c) for c in table.primary_key])
        )

    return exp.Create(
        this=schema_expr,
        kind="TABLE",
        exists=if_not_exists,
    )


__all__ = ["create_table_ast", "ibis_schema_from_table_schema"]
