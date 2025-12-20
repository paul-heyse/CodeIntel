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
from sqlglot.errors import SqlglotError

from codeintel.storage.constants import DUCKDB_DIALECT

if TYPE_CHECKING:
    from ibis.expr.datatypes import DataType

    from codeintel.core.schemas.primitives import TableSchema

_DECIMAL_PARTS_MIN = 2
_MAP_PARTS_MIN = 2
_ARRAY_TYPES = frozenset({exp.DataType.Type.LIST, exp.DataType.Type.ARRAY})
_TIMESTAMP_TYPE_MAP: dict[exp.DataType.Type, str] = {
    exp.DataType.Type.TIMESTAMPTZ: "timestamp('UTC')",
    exp.DataType.Type.TIMESTAMPLTZ: "timestamp('UTC')",
    exp.DataType.Type.TIMESTAMP_S: "timestamp('s')",
    exp.DataType.Type.TIMESTAMP_MS: "timestamp('ms')",
    exp.DataType.Type.TIMESTAMP_NS: "timestamp('ns')",
    exp.DataType.Type.TIMESTAMP: "timestamp",
    exp.DataType.Type.TIMESTAMPNTZ: "timestamp",
}
_BASE_TYPE_MAP: dict[exp.DataType.Type, str] = {
    exp.DataType.Type.BOOLEAN: "boolean",
    exp.DataType.Type.TINYINT: "int8",
    exp.DataType.Type.SMALLINT: "int16",
    exp.DataType.Type.INT: "int32",
    exp.DataType.Type.BIGINT: "int64",
    exp.DataType.Type.UTINYINT: "uint8",
    exp.DataType.Type.USMALLINT: "uint16",
    exp.DataType.Type.UINT: "uint32",
    exp.DataType.Type.UBIGINT: "uint64",
    exp.DataType.Type.FLOAT: "float32",
    exp.DataType.Type.DOUBLE: "float64",
    exp.DataType.Type.TEXT: "string",
    exp.DataType.Type.VARCHAR: "string",
    exp.DataType.Type.CHAR: "string",
    exp.DataType.Type.BINARY: "binary",
    exp.DataType.Type.VARBINARY: "binary",
    exp.DataType.Type.BLOB: "binary",
    exp.DataType.Type.JSON: "json",
    exp.DataType.Type.UUID: "uuid",
    exp.DataType.Type.DATE: "date",
    exp.DataType.Type.TIME: "time",
}


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
    """
    cols: dict[str, DataType] = {}
    for col in table.columns:
        dtype = _ibis_dtype_from_duckdb_type(col.type)
        if not col.nullable:
            dtype = dtype.copy(nullable=False)
        cols[col.name] = dtype

    return ibis.schema(cols)


def _ibis_dtype_from_duckdb_type(type_str: str) -> DataType:
    try:
        parsed = exp.DataType.build(type_str, dialect=DUCKDB_DIALECT)
    except (SqlglotError, TypeError, ValueError) as exc:
        msg = f"Unsupported column type: {type_str}"
        raise ValueError(msg) from exc
    return _ibis_dtype_from_sqlglot(parsed)


def _ibis_dtype_from_sqlglot(dtype: exp.DataType) -> DataType:
    dtype_type = dtype.this

    if dtype_type == exp.DataType.Type.DECIMAL:
        return _ibis_decimal_dtype(dtype)

    if dtype_type in _TIMESTAMP_TYPE_MAP:
        return _timestamp_dtype(dtype_type)

    if dtype_type in _ARRAY_TYPES:
        return _array_dtype(dtype)

    if dtype_type == exp.DataType.Type.MAP:
        return _map_dtype(dtype)

    if dtype_type == exp.DataType.Type.STRUCT:
        return _struct_dtype(dtype)

    return _base_dtype(dtype)


def _ibis_decimal_dtype(dtype: exp.DataType) -> DataType:
    parts = dtype.expressions or []
    if len(parts) >= _DECIMAL_PARTS_MIN:
        precision = parts[0].this
        scale = parts[1].this
        return ibis.dtype(f"decimal({precision},{scale})")
    return ibis.dtype("decimal")


def _timestamp_dtype(dtype_type: exp.DataType.Type) -> DataType:
    mapped = _TIMESTAMP_TYPE_MAP[dtype_type]
    return ibis.dtype(mapped)


def _array_dtype(dtype: exp.DataType) -> DataType:
    element = _require_dtype_expression(dtype, index=0, context="LIST")
    element_dtype = _ibis_dtype_from_sqlglot(element)
    return ibis.dtype(f"array<{element_dtype}>")


def _map_dtype(dtype: exp.DataType) -> DataType:
    if not dtype.expressions or len(dtype.expressions) < _MAP_PARTS_MIN:
        msg = "Unsupported column type: MAP without key/value types"
        raise ValueError(msg)
    key_dtype = _ibis_dtype_from_sqlglot(_require_dtype_expression(dtype, index=0, context="MAP"))
    value_dtype = _ibis_dtype_from_sqlglot(_require_dtype_expression(dtype, index=1, context="MAP"))
    return ibis.dtype(f"map<{key_dtype}, {value_dtype}>")


def _struct_dtype(dtype: exp.DataType) -> DataType:
    fields: list[str] = []
    for column_def in dtype.expressions or []:
        if not isinstance(column_def, exp.ColumnDef):
            msg = "Unsupported column type: STRUCT fields must be ColumnDef"
            raise TypeError(msg)
        field_name = column_def.name
        field_type = column_def.args.get("kind")
        if not isinstance(field_type, exp.DataType):
            msg = f"Unsupported column type: STRUCT field {field_name}"
            raise TypeError(msg)
        field_dtype = _ibis_dtype_from_sqlglot(field_type)
        fields.append(f"{field_name}: {field_dtype}")
    if not fields:
        msg = "Unsupported column type: STRUCT without fields"
        raise ValueError(msg)
    return ibis.dtype(f"struct<{', '.join(fields)}>")


def _base_dtype(dtype: exp.DataType) -> DataType:
    mapped = _BASE_TYPE_MAP.get(dtype.this)
    if mapped is None:
        msg = f"Unsupported column type: {dtype.sql(dialect=DUCKDB_DIALECT)}"
        raise ValueError(msg)
    return ibis.dtype(mapped)


def _require_dtype_expression(
    dtype: exp.DataType,
    *,
    index: int,
    context: str,
) -> exp.DataType:
    parts = dtype.expressions or []
    if len(parts) <= index:
        msg = f"Unsupported column type: {context} without element type"
        raise ValueError(msg)
    part = parts[index]
    if not isinstance(part, exp.DataType):
        msg = f"Unsupported column type: {context} element is not a datatype"
        raise TypeError(msg)
    return part


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
