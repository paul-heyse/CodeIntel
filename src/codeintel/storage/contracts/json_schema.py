"""JSON Schema helpers built from dataset contracts and TableSchema definitions."""

from __future__ import annotations

from typing import Any

from codeintel.core.schemas.service import get_schema_service


def get_json_schema_for_table_key(table_key: str) -> dict[str, Any] | None:
    """Return JSON Schema for a fully qualified table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    dict[str, Any] | None
        JSON Schema 2020-12 dictionary, or None if not found.
    """
    service = get_schema_service()
    schema = service.get_json_schema(table_key)
    if schema is None:
        return None
    return dict(schema)


__all__ = [
    "get_json_schema_for_table_key",
]
