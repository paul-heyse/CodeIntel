"""JSON Schema helpers built from dataset contracts and TableSchema definitions."""

from __future__ import annotations

from typing import Any

from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema
from codeintel.core.schemas.service import get_schema_service
from codeintel.storage.contracts.provider import get_contract_for_table_key


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
    try:
        service = get_schema_service()
    except RuntimeError:
        service = None
    if service is not None:
        schema = service.get_json_schema(table_key)
        if schema is not None:
            return schema

    try:
        contract = get_contract_for_table_key(table_key)
    except KeyError:
        return None
    if contract.schema is None:
        return None
    return json_schema_from_table_schema(
        contract.schema,
        schema_id=f"urn:codeintel:schema:{contract.table_key}",
    )


__all__ = [
    "get_json_schema_for_table_key",
]
