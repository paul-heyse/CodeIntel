"""Helpers for schema-derived row serialization."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.row_registry import get_row_binding

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.row_models import RowSerializer


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
        Callable that converts a row mapping into an ordered tuple.
    """
    return get_row_binding(table_key).serializer


def row_to_tuple(table_key: str, row: Mapping[str, object]) -> tuple[object, ...]:
    """Convert a row mapping to an ordered tuple for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    row
        Row mapping from column name to value.

    Returns
    -------
    tuple[object, ...]
        Row values ordered per the table schema.
    """
    return row_serializer_for_table_key(table_key)(row)


__all__ = [
    "row_serializer_for_table_key",
    "row_to_tuple",
]
