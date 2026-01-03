"""Centralized row serialization backed by the schema registry."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import msgspec

from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.row_models import RowSerializer, RowStructSerializer


@lru_cache(maxsize=2048)
def row_serializer_for_table_key(table_key: str) -> RowSerializer:
    """Return a cached row serializer for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    RowSerializer
        Serializer that orders row values according to the schema registry.

    Raises
    ------
    RuntimeError
        If the schema provider cannot resolve the requested table key.
    """
    service = get_schema_service()
    binding = service.get_row_binding(table_key)
    if binding is None:
        msg = f"{table_key} missing from schema provider"
        raise RuntimeError(msg)
    return binding.serializer


@lru_cache(maxsize=2048)
def row_struct_serializer_for_table_key(table_key: str) -> RowStructSerializer:
    """Return a cached struct serializer for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    RowStructSerializer
        Serializer that orders struct fields according to the schema registry.

    Raises
    ------
    RuntimeError
        If the schema provider cannot resolve the requested table key.
    """
    service = get_schema_service()
    binding = service.get_row_binding(table_key)
    if binding is None:
        msg = f"{table_key} missing from schema provider"
        raise RuntimeError(msg)
    return binding.struct_serializer


def row_to_tuple(table_key: str, row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a row mapping into a tuple for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    row
        Row mapping from column name to value.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the table schema.
    """
    return row_serializer_for_table_key(table_key)(row)


def row_struct_to_tuple(table_key: str, row: msgspec.Struct) -> tuple[object, ...]:
    """Serialize a row struct into a tuple for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    row
        Msgspec Struct row instance.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the table schema.
    """
    return row_struct_serializer_for_table_key(table_key)(row)


def clear_row_serializer_cache() -> None:
    """Clear the cached row serializers."""
    row_serializer_for_table_key.cache_clear()
    row_struct_serializer_for_table_key.cache_clear()


__all__ = [
    "clear_row_serializer_cache",
    "row_serializer_for_table_key",
    "row_struct_serializer_for_table_key",
    "row_struct_to_tuple",
    "row_to_tuple",
]
