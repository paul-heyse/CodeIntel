"""Row model generation from core TableSchema primitives.

Row models are a derived convenience for typed row-shaped data interchange,
not a separate schema authority. They are generated on demand from ``TableSchema``
and cached for reuse.

This module also provides ``GeneratedRowBinding``, a schema-generated row binding
that includes provenance metadata (table_key, schema_hash) for cache invalidation
and debugging.

Additionally, it provides utilities for generating TypedDicts from Pandera
DataFrameSchema definitions for interoperability with validation boundaries.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, make_dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from codeintel.core.schemas.hashing import schema_hash

if TYPE_CHECKING:
    from pandera import DataFrameSchema

    from codeintel.core.schemas.primitives import ColumnType, TableSchema

_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _row_model_class_name(*, schema: str, name: str) -> str:
    schema_part = schema[:1].upper() + schema[1:]
    return f"{schema_part}__{name}__Row"


def _python_type_for_column_type(col_type: ColumnType) -> type[object]:
    if col_type in {"INTEGER", "BIGINT", "DECIMAL(38,0)"}:
        return int
    if col_type in {"DOUBLE", "DECIMAL"}:
        return float
    if col_type == "BOOLEAN":
        return bool
    if col_type == "VARCHAR":
        return str
    if col_type == "JSON":
        return object
    if col_type in {"TIMESTAMP", "TIMESTAMPTZ"}:
        return dt.datetime
    msg = f"Unsupported ColumnType for row model generation: {col_type}"
    raise ValueError(msg)


def _row_model_signature(schema: TableSchema) -> tuple[tuple[str, ColumnType, bool], ...]:
    return tuple((col.name, col.type, col.nullable) for col in schema.columns)


@lru_cache(maxsize=2048)
def _row_model_cached(
    schema: str,
    name: str,
    signature: tuple[tuple[str, ColumnType, bool], ...],
) -> type[object]:
    class_name = _row_model_class_name(schema=schema, name=name)

    fields: list[tuple[str, object]] = []
    for col_name, col_type, nullable in signature:
        if not _VALID_IDENTIFIER_RE.match(col_name):
            msg = f"Column name is not a valid identifier for row model: {col_name}"
            raise ValueError(msg)
        base = _python_type_for_column_type(col_type)
        annotated: object = base | None if nullable else base
        fields.append((col_name, annotated))

    return make_dataclass(class_name, fields=fields, frozen=True, slots=True)


def row_model_for_table_schema(*, table_schema: TableSchema) -> type[object]:
    """Return a cached dataclass row model for a TableSchema.

    Parameters
    ----------
    table_schema
        Source TableSchema.

    Returns
    -------
    type[object]
        Frozen dataclass type with fields matching the schema column order.
    """
    return _row_model_cached(
        table_schema.schema,
        table_schema.name,
        _row_model_signature(table_schema),
    )


RowSerializer = Callable[[Mapping[str, object]], tuple[object, ...]]


@lru_cache(maxsize=2048)
def _row_serializer_cached(signature: tuple[tuple[str, ColumnType, bool], ...]) -> RowSerializer:
    column_names = tuple(name for name, _col_type, _nullable in signature)

    def _serialize(row: Mapping[str, object]) -> tuple[object, ...]:
        return tuple(row[col] for col in column_names)

    return _serialize


def row_serializer_for_table_schema(*, table_schema: TableSchema) -> RowSerializer:
    """Return a cached mapping->tuple serializer using the schema column order.

    Parameters
    ----------
    table_schema
        Source TableSchema.

    Returns
    -------
    RowSerializer
        Function that serializes a row mapping into an ordered tuple.
    """
    return _row_serializer_cached(_row_model_signature(table_schema))


@dataclass(frozen=True)
class GeneratedRowBinding:
    """Schema-generated row binding with provenance metadata.

    This class provides a schema-generated binding while adding provenance for
    cache invalidation and debugging.

    Parameters
    ----------
    row_model
        Generated frozen dataclass type with fields matching the schema.
    serializer
        Function that converts a row mapping to an ordered tuple.
    table_key
        Fully qualified table key (schema.table) for provenance.
    schema_hash
        SHA-256 hash of the source TableSchema for cache invalidation.

    Examples
    --------
    >>> from codeintel.core.schemas import TableSchema, Column
    >>> schema = TableSchema(
    ...     schema="test",
    ...     name="example",
    ...     columns=[
    ...         Column(name="id", type="INTEGER", nullable=False),
    ...     ],
    ... )
    >>> binding = row_binding_for_table_schema(table_schema=schema)
    >>> binding.table_key
    'test.example'
    """

    row_model: type[object]
    serializer: RowSerializer
    table_key: str
    schema_hash: str


def row_binding_for_table_schema(*, table_schema: TableSchema) -> GeneratedRowBinding:
    """Generate a complete row binding from a TableSchema.

    This function creates a ``GeneratedRowBinding`` containing both the row
    model (frozen dataclass) and serializer, along with provenance metadata
    for cache management.

    Parameters
    ----------
    table_schema
        Source TableSchema defining the table structure.

    Returns
    -------
    GeneratedRowBinding
        Complete binding with row model, serializer, and provenance.

    Examples
    --------
    >>> from codeintel.core.schemas import TableSchema, Column
    >>> schema = TableSchema(
    ...     schema="analytics",
    ...     name="metrics",
    ...     columns=[
    ...         Column(name="repo", type="VARCHAR", nullable=False),
    ...         Column(name="loc", type="INTEGER", nullable=True),
    ...     ],
    ... )
    >>> binding = row_binding_for_table_schema(table_schema=schema)
    >>> binding.table_key
    'analytics.metrics'
    >>> len(binding.schema_hash)
    64
    """
    model = row_model_for_table_schema(table_schema=table_schema)
    serializer = row_serializer_for_table_schema(table_schema=table_schema)

    return GeneratedRowBinding(
        row_model=model,
        serializer=serializer,
        table_key=table_schema.table_key,
        schema_hash=schema_hash(table_schema),
    )


# ---------------------------------------------------------------------------
# Pandera-based row model generation
# ---------------------------------------------------------------------------

_PANDAS_TYPE_MAP: dict[type[object], type[Any]] = {
    pd.Int64Dtype: int,
    pd.Float64Dtype: float,
    pd.BooleanDtype: bool,
    pd.StringDtype: str,
}

_STRING_MARKERS: list[tuple[str, type[Any]]] = [
    ("int", int),
    ("float", float),
    ("double", float),
    ("bool", bool),
    ("datetime", dt.datetime),
]


def _pandera_dtype_to_python(dtype: object) -> type[Any]:
    """Map a Pandera column dtype to a Python type.

    This function performs a best-effort mapping from Pandera/pandas dtypes
    to Python types suitable for TypedDict annotations.

    Parameters
    ----------
    dtype
        Pandera column dtype (may be a pandas dtype, numpy dtype, or type).

    Returns
    -------
    type[Any]
        Corresponding Python type.

    Examples
    --------
    >>> _pandera_dtype_to_python(pd.Int64Dtype())
    <class 'int'>
    """
    for pandera_type, python_type in _PANDAS_TYPE_MAP.items():
        if isinstance(dtype, pandera_type):
            return python_type

    dtype_str = str(dtype).lower()
    for marker, python_type in _STRING_MARKERS:
        if marker in dtype_str:
            return python_type

    return str


def typed_dict_from_pandera(
    name: str,
    schema: DataFrameSchema,
    *,
    nullable_as_optional: bool = True,
) -> type[Any]:
    """Generate a TypedDict from a Pandera DataFrameSchema.

    This enables automatic derivation of row types from the canonical Pandera
    schema, eliminating manual TypedDict maintenance.

    Parameters
    ----------
    name
        Name for the generated TypedDict class.
    schema
        Pandera DataFrameSchema to derive from.
    nullable_as_optional
        If True, nullable columns become union types with None (e.g., `int | None`).

    Returns
    -------
    type[Any]
        Generated TypedDict class with appropriate field annotations.

    Examples
    --------
    >>> from pandera import DataFrameSchema, Column
    >>> schema = DataFrameSchema(
    ...     {
    ...         "repo": Column(str),
    ...         "loc": Column(int, nullable=True),
    ...     }
    ... )
    >>> RowModel = typed_dict_from_pandera("MyRow", schema)
    >>>
    """
    annotations: dict[str, Any] = {}

    for col_name, column in schema.columns.items():
        py_type = _pandera_dtype_to_python(column.dtype)

        if nullable_as_optional and column.nullable:
            annotations[col_name] = py_type | None
        else:
            annotations[col_name] = py_type

    td_class = type(name, (), {"__annotations__": annotations, "__total__": True})
    return cast("type[Any]", td_class)


def row_serializer_from_pandera(
    schema: DataFrameSchema,
) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    """Generate a row serializer function from a Pandera schema.

    The serializer converts a row dictionary to a tuple in column order,
    suitable for database INSERT operations.

    Parameters
    ----------
    schema
        Pandera DataFrameSchema defining column order.

    Returns
    -------
    Callable[[Mapping[str, Any]], tuple[Any, ...]]
        Serializer function that converts row dicts to tuples.

    Examples
    --------
    >>> from pandera import DataFrameSchema, Column
    >>> schema = DataFrameSchema({"a": Column(str), "b": Column(int)})
    >>> serialize = row_serializer_from_pandera(schema)
    >>> serialize({"a": "hello", "b": 42})
    ('hello', 42)
    """
    columns = tuple(schema.columns.keys())

    def serialize(row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Serialize a row dictionary to a tuple in column order.

        Parameters
        ----------
        row
            Row data as a mapping from column name to value.

        Returns
        -------
        tuple[Any, ...]
            Values ordered according to schema columns.
        """
        return tuple(row[col] for col in columns)

    return serialize


__all__ = [
    "GeneratedRowBinding",
    "RowSerializer",
    "row_binding_for_table_schema",
    "row_model_for_table_schema",
    "row_serializer_for_table_schema",
    "row_serializer_from_pandera",
    "typed_dict_from_pandera",
]
