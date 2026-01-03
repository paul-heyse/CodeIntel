"""Row model generation from core TableSchema primitives.

Row models are a derived convenience for typed row-shaped data interchange,
not a separate schema authority. They are generated on demand from ``TableSchema``
and cached for reuse.

This module also provides ``GeneratedRowBinding``, a schema-generated row binding
that includes provenance metadata (table_key, schema_hash) for cache invalidation
and debugging.

Row models are derived from ``TableSchema`` and avoid pandas-specific types.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, make_dataclass
from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from codeintel.core.helpers.json import normalize_duckdb_json_value
from codeintel.core.helpers.payload import PayloadValue, encode_payload
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import COLUMN_TYPE_REGISTRY, column_type_base
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType, TableSchema

_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ROW_MODEL_NAME_PARTS = 3


def _row_model_class_name(*, schema: str, name: str) -> str:
    schema_part = schema[:1].upper() + schema[1:]
    return f"{schema_part}__{name}__Row"


def _register_row_model(name: str, model: type[object]) -> None:
    if globals().get(name) is model:
        return
    globals()[name] = model


def _python_type_for_column_type(col_type: ColumnType) -> type[object]:
    return COLUMN_TYPE_REGISTRY.python_type_for(col_type)


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

    model = make_dataclass(class_name, fields=fields, frozen=True, slots=True, module=__name__)
    _register_row_model(class_name, model)
    return model


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

_ROW_VALUE_CONTAINERS: tuple[type[object], ...] = (dict, list, tuple, set)
_ROW_VALUE_BINARY: tuple[type[object], ...] = (bytes, bytearray, memoryview)
_ROW_VALUE_NON_MISSING: tuple[type[object], ...] = _ROW_VALUE_CONTAINERS + _ROW_VALUE_BINARY


def _is_missing_value(value: object) -> bool:
    if isinstance(value, _ROW_VALUE_NON_MISSING):
        return False
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, Decimal):
        return value.is_nan()
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def normalize_row_value(value: object) -> object:
    """Normalize row values for serialization and insertion.

    Returns
    -------
    object
        Normalized value suitable for row serialization.
    """
    if value is None:
        return None
    if _is_missing_value(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_payload_value(
    value: object,
) -> PayloadValue | bytes | bytearray | memoryview | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence):
        return value
    msg = f"Unsupported payload value type: {type(value).__name__}"
    raise TypeError(msg)


def normalize_row_value_for_type(value: object, column_type: ColumnType | None) -> object:
    """Normalize row values with awareness of column types.

    Parameters
    ----------
    value
        Raw value from a row mapping.
    column_type
        Column type from the table schema, when available.

    Returns
    -------
    object
        Normalized value for insertion/serialization.
    """
    normalized = normalize_row_value(value)
    if normalized is None:
        return None
    if column_type is None:
        return normalized
    base = column_type_base(column_type)
    if base == "BLOB":
        payload_value = _coerce_payload_value(normalized)
        return encode_payload(payload_value)
    if base in {"JSON", "STRUCT", "MAP", "LIST", "UNION"}:
        return normalize_duckdb_json_value(normalized)
    return normalized


@lru_cache(maxsize=2048)
def _row_serializer_cached(signature: tuple[tuple[str, ColumnType, bool], ...]) -> RowSerializer:
    column_types: tuple[tuple[str, ColumnType], ...] = tuple(
        (name, col_type) for name, col_type, _nullable in signature
    )

    def _serialize(row: Mapping[str, object]) -> tuple[object, ...]:
        return tuple(
            normalize_row_value_for_type(row[name], col_type) for name, col_type in column_types
        )

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


def _row_model_name_parts(name: str) -> tuple[str, str] | None:
    parts = name.split("__")
    if len(parts) != _ROW_MODEL_NAME_PARTS or parts[2] != "Row":
        return None
    schema_part, table_name, _suffix = parts
    if not schema_part or not table_name:
        return None
    schema = f"{schema_part[:1].lower()}{schema_part[1:]}"
    return schema, table_name


def __getattr__(name: str) -> object:
    parts = _row_model_name_parts(name)
    if parts is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    schema, table_name = parts
    table_key = f"{schema}.{table_name}"
    table_schema = TABLE_SCHEMAS.get(table_key)
    if table_schema is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return row_model_for_table_schema(table_schema=table_schema)


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
    derivation_kind
        Optional derivation kind for the originating schema.
    derivation_source
        Optional derivation source identifier for the originating schema.

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
    derivation_kind: str | None = None
    derivation_source: str | None = None


def row_binding_for_table_schema(
    *,
    table_schema: TableSchema,
    derivation_kind: str | None = None,
    derivation_source: str | None = None,
) -> GeneratedRowBinding:
    """Generate a complete row binding from a TableSchema.

    This function creates a ``GeneratedRowBinding`` containing both the row
    model (frozen dataclass) and serializer, along with provenance metadata
    for cache management.

    Parameters
    ----------
    table_schema
        Source TableSchema defining the table structure.
    derivation_kind
        Optional derivation kind for the originating schema.
    derivation_source
        Optional derivation source identifier for the originating schema.

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
        derivation_kind=derivation_kind,
        derivation_source=derivation_source,
    )


__all__ = [
    "GeneratedRowBinding",
    "RowSerializer",
    "normalize_row_value",
    "normalize_row_value_for_type",
    "row_binding_for_table_schema",
    "row_model_for_table_schema",
    "row_serializer_for_table_schema",
]
