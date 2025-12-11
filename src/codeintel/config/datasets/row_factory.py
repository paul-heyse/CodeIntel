"""Row model factory for generating TypedDicts from Pandera schemas.

This module provides utilities for automatically deriving TypedDict row models
and row serializers from Pandera DataFrameSchema definitions, eliminating the
need for manual TypedDict maintenance.

Examples
--------
>>> from pandera import DataFrameSchema, Column
>>> schema = DataFrameSchema({"repo": Column(str), "loc": Column(int, nullable=True)})
>>> RowModel = typed_dict_from_pandera("ExampleRow", schema)
>>> # RowModel is a TypedDict with repo: str, loc: int | None
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pandera import DataFrameSchema

__all__ = [
    "row_serializer_from_pandera",
    "typed_dict_from_pandera",
]

# Pre-built dtype mapping at module load time
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
    ("datetime", datetime.datetime),
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
    >>> # RowModel has annotations: {"repo": str, "loc": int | None}
    """
    annotations: dict[str, Any] = {}

    for col_name, column in schema.columns.items():
        py_type = _pandera_dtype_to_python(column.dtype)

        if nullable_as_optional and column.nullable:
            # Use union with None for nullable columns
            annotations[col_name] = py_type | None
        else:
            annotations[col_name] = py_type

    # Create TypedDict using class creation
    # We use __annotations__ assignment which is how TypedDict classes work internally
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
