"""Row serialization helpers for ingestion workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.schemas.row_models import RowSerializer


def row_serializer_for_table_key(table_key: str) -> RowSerializer:
    """Return the schema-derived row serializer for a table key.

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
    try:
        service = get_schema_service()
    except RuntimeError as exc:
        columns = load_columns_by_table().get(table_key, [])
        if not columns:
            msg = f"{table_key} missing from schema provider"
            raise RuntimeError(msg) from exc
        return lambda row: serialize_row(row, columns)

    binding = service.get_row_binding(table_key)
    if binding is None:
        msg = f"{table_key} missing from schema provider"
        raise RuntimeError(msg)
    return binding.serializer


def row_to_tuple(table_key: str, row: Mapping[str, object]) -> tuple[object, ...]:
    """Serialize a row mapping into a tuple for a table key.

    Returns
    -------
    tuple[object, ...]
        Row values ordered according to the table schema.
    """
    return row_serializer_for_table_key(table_key)(row)


__all__ = [
    "row_serializer_for_table_key",
    "row_to_tuple",
]
