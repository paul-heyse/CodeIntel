"""JSON Schema registry for generated schemas.

Provide cached access to JSON Schemas generated from TableSchema definitions.
This replaces the hand-maintained JSON schema files in config/schemas/export/.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from codeintel.build.schemas.service import get_schema_service
from codeintel.core.errors.schema import (
    SchemaDigestError,
    SchemaLoadError,
    SchemaNotFoundError,
)
from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema

log = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def get_json_schema(table_key: str) -> dict[str, Any]:
    """Return generated JSON Schema for a table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    dict[str, Any]
        JSON Schema 2020-12 dictionary.

    Raises
    ------
    KeyError
        If the table key is not registered.

    Examples
    --------
    >>> schema = get_json_schema("analytics.function_metrics")
    >>> schema["$schema"]
    'https://json-schema.org/draft/2020-12/schema'
    """
    schema = get_schema_service().get_json_schema(table_key)
    if schema is None:
        msg = f"Unknown table schema: {table_key}"
        raise KeyError(msg)
    return schema


def get_json_schema_for_table_schema(
    table_schema: TableSchema,
    *,
    schema_id: str | None = None,
) -> dict[str, Any]:
    """Generate JSON Schema from a TableSchema directly.

    Useful for ad-hoc schema generation without going through the registry.

    Parameters
    ----------
    table_schema
        Source TableSchema.
    schema_id
        Optional ``$id`` URI. Defaults to ``urn:codeintel:schema:{table_key}``.

    Returns
    -------
    dict[str, Any]
        JSON Schema 2020-12 dictionary.
    """
    effective_id = schema_id or f"urn:codeintel:schema:{table_schema.table_key}"
    return json_schema_from_table_schema(table_schema, schema_id=effective_id)


def compute_json_schema_digest(table_key: str) -> str:
    """Compute a stable digest of the generated JSON Schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    str
        SHA-256 hex digest of the canonical JSON representation.

    Raises
    ------
    SchemaNotFoundError
        If the table key is not found in the schema registry.
    SchemaLoadError
        If schema loading fails for any reason.
    SchemaDigestError
        If digest computation fails for any reason.
    """
    try:
        digest = get_schema_service().compute_json_schema_digest(table_key)
    except KeyError as exc:
        raise SchemaNotFoundError(table_key) from exc
    except Exception as exc:
        raise SchemaLoadError(table_key, exc) from exc

    if digest is None:
        raise SchemaDigestError(table_key, RuntimeError("JSON schema digest unavailable"))
    return digest


def clear_json_schema_cache() -> None:
    """Clear the JSON schema cache.

    Useful for testing when schema definitions change.
    """
    get_json_schema.cache_clear()


__all__ = [
    "SchemaDigestError",
    "SchemaLoadError",
    "SchemaNotFoundError",
    "clear_json_schema_cache",
    "compute_json_schema_digest",
    "get_json_schema",
    "get_json_schema_for_table_schema",
]
