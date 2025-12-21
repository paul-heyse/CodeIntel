"""Storage-owned schema provider façade.

Storage needs access to the canonical table/view schemas for DDL generation,
schema hashing, and validation, but must not import `codeintel.build.*`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.config.datasets.declared_schemas import TABLE_SCHEMAS
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


@lru_cache(maxsize=1)
def _fallback_provider() -> SchemaProvider:
    return MappingSchemaProvider(TABLE_SCHEMAS)


@lru_cache(maxsize=1)
def get_schema_provider() -> SchemaProvider:
    """Return the canonical SchemaProvider for storage.

    When a global SchemaService has been configured (for example by the build
    layer), its table provider is used to align storage with DAG-first schemas.
    Otherwise, fall back to declared table schemas.

    Returns
    -------
    SchemaProvider
        Provider for table/view schemas.
    """
    try:
        service = get_schema_service()
    except RuntimeError:
        return _fallback_provider()
    return service.table_provider


def require_table_schema(table_key: str) -> TableSchema:
    """Return schema for table_key, raising when unknown.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        Declared schema for the requested key.

    Raises
    ------
    KeyError
        If no schema is declared for the table key.
    """
    schema = get_schema_provider().get_table_schema(table_key)
    if schema is None:
        msg = f"Unknown table schema: {table_key}"
        raise KeyError(msg)
    return schema


def iter_table_schemas() -> Iterable[TableSchema]:
    """Iterate all known table/view schemas.

    Returns
    -------
    Iterable[TableSchema]
        Iterable of all declared schemas.
    """
    return get_schema_provider().iter_table_schemas()


def clear_schema_provider_cache() -> None:
    """Clear the cached schema provider (for testing)."""
    get_schema_provider.cache_clear()


__all__ = [
    "clear_schema_provider_cache",
    "get_schema_provider",
    "iter_table_schemas",
    "require_table_schema",
]
