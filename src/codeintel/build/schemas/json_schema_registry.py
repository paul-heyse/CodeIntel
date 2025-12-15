"""JSON Schema registry for generated schemas.

Provide cached access to JSON Schemas generated from TableSchema definitions.
This replaces the hand-maintained JSON schema files in config/schemas/export/.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from codeintel.core.errors.schema import (
    SchemaDigestError,
    SchemaLoadError,
    SchemaNotFoundError,
)
from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema

if TYPE_CHECKING:
    from types import ModuleType

    from codeintel.core.schemas.primitives import TableSchema

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lazy Module Access
# -----------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _get_module(name: str) -> ModuleType:
    """Load a module lazily with caching.

    Parameters
    ----------
    name
        Fully qualified module name.

    Returns
    -------
    ModuleType
        The loaded module.
    """
    return importlib.import_module(name)


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

    Examples
    --------
    >>> schema = get_json_schema("analytics.function_metrics")
    >>> schema["$schema"]
    'https://json-schema.org/draft/2020-12/schema'
    """
    # Lazy import to avoid circular dependencies at module load time
    registry_mod = _get_module("codeintel.build.schemas.registry")
    table_schema = registry_mod.get_schema_provider().require_table_schema(table_key)
    return json_schema_from_table_schema(
        table_schema,
        schema_id=f"urn:codeintel:schema:{table_key}",
    )


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


def get_json_schema_for_dataset_name(dataset_name: str) -> dict[str, Any] | None:
    """Return JSON Schema by dataset name for backward compatibility.

    Maps dataset names (e.g., "function_profile") to table keys and returns
    the generated JSON Schema.

    Parameters
    ----------
    dataset_name
        Dataset name without schema prefix.

    Returns
    -------
    dict[str, Any] | None
        JSON Schema 2020-12 dictionary, or None if not found.
    """
    # Lazy import to avoid circular dependencies
    contract_mod = _get_module("codeintel.build.schemas.contract_provider")

    for contract in contract_mod.iter_contracts():
        if contract.name == dataset_name and contract.schema is not None:
            return json_schema_from_table_schema(
                contract.schema,
                schema_id=f"urn:codeintel:schema:{contract.table_key}",
            )
    return None


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
        schema = get_json_schema(table_key)
    except KeyError as e:
        raise SchemaNotFoundError(table_key) from e
    except Exception as e:
        raise SchemaLoadError(table_key, e) from e

    try:
        # Use sorted keys for deterministic output
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception as e:
        raise SchemaDigestError(table_key, e) from e


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
    "get_json_schema_for_dataset_name",
    "get_json_schema_for_table_schema",
]
