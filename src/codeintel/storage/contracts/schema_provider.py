"""Storage-owned schema provider façade.

Storage needs access to the canonical table/view schemas for DDL generation,
schema hashing, and validation, but must not import `codeintel.build.*`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS
from codeintel.storage.contracts.catalog_state import contract_catalog_table_schemas

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.service import SchemaService


@lru_cache(maxsize=1)
def get_schema_provider() -> SchemaProvider:
    """Return the canonical SchemaProvider for storage.

    Returns
    -------
    SchemaProvider
        Provider for table/view schemas.
    """
    catalog_schemas = contract_catalog_table_schemas()
    if catalog_schemas:
        return MappingSchemaProvider(catalog_schemas)
    return _fallback_schema_provider()


def _fallback_schema_provider() -> SchemaProvider:
    """Resolve a schema provider when the contract catalog is unavailable.

    Returns
    -------
    SchemaProvider
        Best-effort schema provider for storage bootstrap.
    """
    try:
        return get_schema_service().table_provider
    except RuntimeError:
        service_factory = cast(
            "Callable[[], SchemaService]",
            lazy_getattr("codeintel.build.schemas.service", "get_schema_service"),
        )
        try:
            return service_factory().table_provider
        except RuntimeError:
            return MappingSchemaProvider(TABLE_SCHEMAS)


def require_table_schema(table_key: str) -> TableSchema:
    """Return schema for table_key, raising when unknown.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    TableSchema
        Schema for the requested key.

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
        Iterable of all schemas.
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
