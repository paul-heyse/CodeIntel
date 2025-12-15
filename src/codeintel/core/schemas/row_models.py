"""Row model generation from core TableSchema primitives.

Row models are a derived convenience for typed row-shaped data interchange,
not a separate schema authority. They are generated on demand from ``TableSchema``
and cached for reuse.

This module also provides ``GeneratedRowBinding``, a schema-generated equivalent
to the legacy ``RowBinding`` that includes provenance metadata (table_key,
schema_hash) for cache invalidation and debugging.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, make_dataclass
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.core.schemas.hashing import schema_hash

if TYPE_CHECKING:
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
    if col_type in {"VARCHAR", "JSON"}:
        return str
    if col_type in {"TIMESTAMP", "TIMESTAMPTZ"}:
        return datetime
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
    for col_name, col_type, _nullable in signature:
        if not _VALID_IDENTIFIER_RE.match(col_name):
            msg = f"Column name is not a valid identifier for row model: {col_name}"
            raise ValueError(msg)
        base = _python_type_for_column_type(col_type)
        annotated: object = base | None
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
        return tuple(row.get(col) for col in column_names)

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

    This class provides a drop-in replacement for the legacy ``RowBinding``
    dataclass while adding schema provenance for cache invalidation and
    debugging. The ``row_type`` and ``to_tuple`` properties maintain
    compatibility with the legacy interface.

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

    @property
    def row_type(self) -> type[object]:
        """Return the row model type for legacy compatibility.

        Returns
        -------
        type[object]
            The generated dataclass row model.
        """
        return self.row_model

    @property
    def to_tuple(self) -> RowSerializer:
        """Return the serializer function for legacy compatibility.

        Returns
        -------
        RowSerializer
            Function that serializes a row mapping to a tuple.
        """
        return self.serializer


def row_binding_for_table_schema(*, table_schema: TableSchema) -> GeneratedRowBinding:
    """Generate a complete RowBinding from a TableSchema.

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


__all__ = [
    "GeneratedRowBinding",
    "RowSerializer",
    "row_binding_for_table_schema",
    "row_model_for_table_schema",
    "row_serializer_for_table_schema",
]
