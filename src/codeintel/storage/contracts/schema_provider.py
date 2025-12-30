"""Storage-owned schema provider façade.

Storage needs access to the canonical table/view schemas for DDL generation,
schema hashing, and validation, but must not import `codeintel.build.*`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.core.schemas.provider import MappingSchemaProvider, SchemaProvider
from codeintel.core.schemas.service import get_schema_service
from codeintel.storage.contracts.catalog_state import contract_catalog_table_schemas
from codeintel.storage.views.inventory import discover_derived_docs_views, view_builder_modules
from codeintel.storage.views.schema_inference import derive_view_schemas

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import TableSchema


@dataclass
class _ViewSchemaProvider:
    """SchemaProvider wrapper that adds derived docs view schemas."""

    base: SchemaProvider
    _view_schema_cache: dict[str, TableSchema] = field(default_factory=dict, repr=False)
    _view_schema_loaded: bool = field(default=False, repr=False)

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        schema = self.base.get_table_schema(table_key)
        if schema is not None:
            return schema
        return self._view_schema_map().get(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        seen: set[str] = set()
        for schema in self.base.iter_table_schemas():
            seen.add(schema.table_key)
            yield schema
        for table_key, schema in self._view_schema_map().items():
            if table_key in seen:
                continue
            yield schema

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        return self.base.derivation(table_key)

    def _view_schema_map(self) -> dict[str, TableSchema]:
        if self._view_schema_loaded:
            return self._view_schema_cache
        view_keys = discover_derived_docs_views()
        self._view_schema_cache = derive_view_schemas(
            provider=self.base,
            view_keys=view_keys,
            modules=view_builder_modules(),
        )
        self._view_schema_loaded = True
        return self._view_schema_cache


@lru_cache(maxsize=1)
def get_schema_provider() -> SchemaProvider:
    """Return the canonical SchemaProvider for storage.

    Returns
    -------
    SchemaProvider
        Provider for table/view schemas.

    Raises
    ------
    RuntimeError
        Raised when the contract catalog is not loaded.
    """
    try:
        service = get_schema_service()
    except RuntimeError:
        service = None
    else:
        return service.table_provider
    catalog_schemas = contract_catalog_table_schemas()
    if not catalog_schemas:
        msg = "Contract catalog not loaded; schema provider unavailable"
        raise RuntimeError(msg)
    return _ViewSchemaProvider(MappingSchemaProvider(catalog_schemas))


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
