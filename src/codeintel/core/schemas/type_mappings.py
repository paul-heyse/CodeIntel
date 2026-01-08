"""Cross-engine column type normalization and conversion helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import sqlglot.expressions as exp

from codeintel.core.columnar.conversion import table_to_frame
from codeintel.core.schemas.arrow_gen import arrow_type_for_column_type
from codeintel.core.schemas.primitives import (
    COMPLEX_TYPE_BASES,
    Column,
    ColumnType,
    TableSchema,
    column_type_base,
    normalize_column_type,
)

if TYPE_CHECKING:
    from duckdb.typing import DuckDBPyType
    from polars import DataType as PolarsDataType
else:
    PolarsDataType = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

_DECIMAL_PREFIX = "DECIMAL("
_MAP_PARAM_COUNT = 2


def normalize_engine_column_type(column_type: ColumnType | None) -> ColumnType | None:
    """Return a canonical column type string for cross-engine usage.

    Parameters
    ----------
    column_type
        Column type string to normalize.

    Returns
    -------
    ColumnType | None
        Normalized column type string, or None when input is None.
    """
    if column_type is None:
        return None
    normalized = normalize_column_type(str(column_type))
    base = column_type_base(normalized)
    if base in COMPLEX_TYPE_BASES or normalized.upper().startswith(_DECIMAL_PREFIX):
        try:
            data_type = exp.DataType.build(normalized, dialect="duckdb")
        except (TypeError, ValueError):
            return normalized
        return data_type.sql(dialect="duckdb")
    return normalized


def _duckdb_pytype_from_sql(type_sql: str) -> DuckDBPyType | None:
    try:
        return duckdb.sqltype(type_sql)
    except (TypeError, ValueError):
        return None


def _decimal_params(data_type: exp.DataType) -> tuple[int, int] | None:
    params: list[int] = []
    for param in data_type.expressions:
        if not isinstance(param, exp.DataTypeParam):
            continue
        literal = param.this
        if isinstance(literal, exp.Literal) and not literal.is_string:
            try:
                params.append(int(literal.this))
            except (TypeError, ValueError):
                return None
    if not params:
        return None
    precision = params[0]
    scale = params[1] if len(params) > 1 else 0
    return precision, scale


def _duckdb_list_type(data_type: exp.DataType) -> DuckDBPyType | None:
    if not data_type.expressions:
        return None
    nested = data_type.expressions[0]
    if not isinstance(nested, exp.DataType):
        return None
    element_type = _duckdb_pytype_from_datatype(nested)
    if element_type is None:
        return None
    return duckdb.list_type(element_type)


def _duckdb_map_type(data_type: exp.DataType) -> DuckDBPyType | None:
    if len(data_type.expressions) < _MAP_PARAM_COUNT:
        return None
    key_expr, value_expr = data_type.expressions[:_MAP_PARAM_COUNT]
    if not isinstance(key_expr, exp.DataType) or not isinstance(value_expr, exp.DataType):
        return None
    key_type = _duckdb_pytype_from_datatype(key_expr)
    value_type = _duckdb_pytype_from_datatype(value_expr)
    if key_type is None or value_type is None:
        return None
    return duckdb.map_type(key_type, value_type)


def _duckdb_fields_from_datatype(data_type: exp.DataType) -> dict[str, DuckDBPyType] | None:
    fields: dict[str, DuckDBPyType] = {}
    for field in data_type.expressions:
        if not isinstance(field, exp.ColumnDef):
            return None
        field_kind = field.args.get("kind")
        if not isinstance(field_kind, exp.DataType):
            return None
        field_type = _duckdb_pytype_from_datatype(field_kind)
        if field_type is None:
            return None
        fields[field.name] = field_type
    if not fields:
        return None
    return fields


def _duckdb_struct_type(data_type: exp.DataType) -> DuckDBPyType | None:
    fields = _duckdb_fields_from_datatype(data_type)
    if fields is None:
        return None
    return duckdb.struct_type(fields)


def _duckdb_union_type(data_type: exp.DataType) -> DuckDBPyType | None:
    fields = _duckdb_fields_from_datatype(data_type)
    if fields is None:
        return None
    return duckdb.union_type(fields)


def _duckdb_decimal_type(data_type: exp.DataType) -> DuckDBPyType | None:
    params = _decimal_params(data_type)
    if params is None:
        return None
    precision, scale = params
    return duckdb.decimal_type(precision, scale)


def _duckdb_pytype_from_datatype(data_type: exp.DataType) -> DuckDBPyType | None:
    handlers: dict[exp.DataType.Type, Callable[[exp.DataType], DuckDBPyType | None]] = {
        exp.DataType.Type.ARRAY: _duckdb_list_type,
        exp.DataType.Type.DECIMAL: _duckdb_decimal_type,
        exp.DataType.Type.LIST: _duckdb_list_type,
        exp.DataType.Type.MAP: _duckdb_map_type,
        exp.DataType.Type.STRUCT: _duckdb_struct_type,
        exp.DataType.Type.UNION: _duckdb_union_type,
    }
    handler = handlers.get(data_type.this)
    if handler is None:
        return _duckdb_pytype_from_sql(data_type.sql(dialect="duckdb"))
    resolved = handler(data_type)
    if resolved is None:
        return _duckdb_pytype_from_sql(data_type.sql(dialect="duckdb"))
    return resolved


def duckdb_pytype_from_column_type(column_type: ColumnType | None) -> DuckDBPyType | None:
    """Return a DuckDBPyType for a normalized column type string.

    Parameters
    ----------
    column_type
        Column type string to convert.

    Returns
    -------
    duckdb.typing.DuckDBPyType | None
        DuckDB type when available, otherwise None.
    """
    if column_type is None:
        return None
    try:
        normalized = normalize_engine_column_type(column_type)
    except ValueError:
        normalized = str(column_type).strip()
    if not normalized:
        return None
    try:
        data_type = exp.DataType.build(normalized, dialect="duckdb")
    except (TypeError, ValueError):
        return _duckdb_pytype_from_sql(normalized)
    return _duckdb_pytype_from_datatype(data_type)


def normalize_table_schema_types(table_schema: TableSchema) -> TableSchema:
    """Return a TableSchema with canonicalized column types.

    Parameters
    ----------
    table_schema
        Source table schema.

    Returns
    -------
    TableSchema
        New schema with normalized column type strings.
    """
    normalized_columns = [
        Column(
            name=column.name,
            type=normalize_engine_column_type(column.type) or column.type,
            nullable=column.nullable,
            description=column.description,
        )
        for column in table_schema.columns
    ]
    return TableSchema(
        schema=table_schema.schema,
        name=table_schema.name,
        columns=normalized_columns,
        primary_key=table_schema.primary_key,
        indexes=table_schema.indexes,
        description=table_schema.description,
        write_policy=table_schema.write_policy,
    )


def arrow_type_from_column_type(column_type: ColumnType) -> pa.DataType:
    """Return a PyArrow type for a normalized column type string.

    Parameters
    ----------
    column_type
        Column type string to convert.

    Returns
    -------
    pyarrow.DataType
        Arrow type corresponding to the column type.

    Raises
    ------
    ValueError
        If the column type is missing after normalization.
    """
    normalized = normalize_engine_column_type(column_type)
    if normalized is None:
        msg = "column_type is required for Arrow type conversion"
        raise ValueError(msg)
    return arrow_type_for_column_type(normalized)


@lru_cache(maxsize=256)
def polars_type_from_column_type(column_type: ColumnType) -> PolarsDataType | None:
    """Return a Polars dtype for a column type string when possible.

    Parameters
    ----------
    column_type
        Column type string to convert.

    Returns
    -------
    polars.DataType | None
        Polars dtype when available, otherwise None.
    """
    if pl is None:  # pragma: no cover
        return None
    arrow_type = arrow_type_from_column_type(column_type)
    try:
        table = pa.Table.from_arrays([pa.array([], type=arrow_type)], names=["_col"])
    except (TypeError, ValueError):
        return None
    frame = table_to_frame(table)
    return frame.schema.get("_col")


@dataclass(frozen=True, slots=True)
class ComplexTypeMapping:
    """Normalized type mapping for complex/nested column types."""

    column_type: ColumnType
    duckdb_type: DuckDBPyType | None
    arrow_type: pa.DataType
    polars_type: PolarsDataType | None


def complex_type_mapping(column_type: ColumnType) -> ComplexTypeMapping | None:
    """Return the unified mapping for complex/nested column types.

    Parameters
    ----------
    column_type
        Column type string to normalize and map.

    Returns
    -------
    ComplexTypeMapping | None
        Normalized mapping when the column type is complex, otherwise None.
    """
    normalized = normalize_engine_column_type(column_type)
    if normalized is None:
        return None
    base = column_type_base(normalized)
    if base not in COMPLEX_TYPE_BASES:
        return None
    duckdb_type = duckdb_pytype_from_column_type(normalized)
    arrow_type = arrow_type_from_column_type(normalized)
    return ComplexTypeMapping(
        column_type=normalized,
        duckdb_type=duckdb_type,
        arrow_type=arrow_type,
        polars_type=polars_type_from_column_type(normalized),
    )


__all__ = [
    "ComplexTypeMapping",
    "arrow_type_from_column_type",
    "complex_type_mapping",
    "duckdb_pytype_from_column_type",
    "normalize_engine_column_type",
    "normalize_table_schema_types",
    "polars_type_from_column_type",
]
