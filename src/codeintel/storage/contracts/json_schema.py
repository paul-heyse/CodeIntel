"""JSON Schema helpers built from dataset contracts and TableSchema definitions."""

from __future__ import annotations

from typing import Any

from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema
from codeintel.storage.contracts.provider import iter_contracts


def get_json_schema_for_dataset_name(dataset_name: str) -> dict[str, Any] | None:
    """Return JSON Schema by dataset name for backward compatibility.

    Parameters
    ----------
    dataset_name
        Dataset name without schema prefix.

    Returns
    -------
    dict[str, Any] | None
        JSON Schema 2020-12 dictionary, or None if not found.
    """
    for contract in iter_contracts():
        if contract.name != dataset_name:
            continue
        if contract.schema is None:
            return None
        return json_schema_from_table_schema(
            contract.schema,
            schema_id=f"urn:codeintel:schema:{contract.table_key}",
        )
    return None


__all__ = [
    "get_json_schema_for_dataset_name",
]
