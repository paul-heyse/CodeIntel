"""Centralized row serialization backed by the schema registry."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.schemas.row_models import RowSerializer

_Column = TypeVar("_Column", bound=str)


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


def row_to_tuple_by_columns(
    row: Mapping[_Column, object],
    columns: Sequence[_Column],
) -> tuple[object, ...]:
    """Serialize a row mapping using an explicit column order.

    Parameters
    ----------
    row
        Row mapping from column name to value.
    columns
        Ordered column names.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the provided columns.
    """
    return tuple(row[column] for column in columns)


def clear_row_serializer_cache() -> None:
    """Clear the cached row serializers."""
    row_serializer_for_table_key.cache_clear()


__all__ = [
    "clear_row_serializer_cache",
    "row_serializer_for_table_key",
    "row_to_tuple",
    "row_to_tuple_by_columns",
]
