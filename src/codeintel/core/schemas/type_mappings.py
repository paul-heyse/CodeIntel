"""Cross-engine column type normalization and conversion helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import pyarrow as pa
import sqlglot.expressions as exp

from codeintel.core.schemas.arrow_gen import arrow_type_for_column_type
from codeintel.core.schemas.primitives import (
    Column,
    ColumnType,
    TableSchema,
    column_type_base,
    normalize_column_type,
)

if TYPE_CHECKING:
    from polars import DataType as PolarsDataType
else:
    PolarsDataType = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

_COMPLEX_BASE_TYPES = frozenset({"STRUCT", "LIST", "MAP", "UNION"})
_DECIMAL_PREFIX = "DECIMAL("


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
    if base in _COMPLEX_BASE_TYPES or normalized.upper().startswith(_DECIMAL_PREFIX):
        try:
            data_type = exp.DataType.build(normalized, dialect="duckdb")
        except (TypeError, ValueError):
            return normalized
        return data_type.sql(dialect="duckdb")
    return normalized


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
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.DataFrame):
        return frame.schema.get("_col")
    if isinstance(frame, pl.Series):
        return frame.dtype
    return None


__all__ = [
    "arrow_type_from_column_type",
    "normalize_engine_column_type",
    "normalize_table_schema_types",
    "polars_type_from_column_type",
]
