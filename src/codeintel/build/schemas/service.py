"""Build-owned SchemaService factory.

This module wires the canonical SchemaService to build-specific providers:
- Unified table schema provider (Hamilton inference + declared schemas)
- DatasetSchema registry (Pandera-backed unified schemas)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from codeintel.build.schemas.provider_unified import (
    clear_unified_provider_cache,
    unified_schema_provider,
)
from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas import (
    DatasetSchemaLike,
    SchemaService,
    clear_schema_service,
    set_schema_service,
)
from codeintel.core.schemas.row_models import row_binding_for_table_schema

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.core.schemas.row_models import GeneratedRowBinding


class _DatasetSchemaRegistry(Protocol):
    """Protocol for the build-owned DatasetSchema registry."""

    def get(self, table_key: str) -> DatasetSchemaLike | None:
        """Return the DatasetSchema for the table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        DatasetSchema | None
            Dataset schema metadata if available.
        """
        ...

    def values(self) -> Iterable[DatasetSchemaLike]:
        """Iterate all registered DatasetSchema objects.

        Returns
        -------
        Iterable[DatasetSchema]
            Registered dataset schemas.
        """
        ...


@runtime_checkable
class _SchemaDerivationProvider(Protocol):
    """Protocol for schema providers that expose derivation metadata."""

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        """Return derivation metadata for the table key."""
        ...


@dataclass(frozen=True, slots=True)
class _BuildDatasetSchemaProvider:
    """Adapter exposing DatasetSchema registry via the DatasetSchemaProvider protocol."""

    registry_module: str = "codeintel.build.hamilton.contracts.schemas.registry"
    registry_attr: str = "SCHEMA_REGISTRY"

    def _registry(self) -> _DatasetSchemaRegistry:
        registry = lazy_getattr(self.registry_module, self.registry_attr)
        return cast("_DatasetSchemaRegistry", registry)

    def get_dataset_schema(self, table_key: str) -> DatasetSchemaLike | None:
        """Return dataset schema for the table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        DatasetSchema | None
            Dataset schema metadata if available.
        """
        return self._registry().get(table_key)

    def iter_dataset_schemas(self) -> Iterable[DatasetSchemaLike]:
        """Iterate all registered dataset schemas.

        Returns
        -------
        Iterable[DatasetSchema]
            Registered dataset schemas.
        """
        return self._registry().values()


_DECLARED_SOURCE_KIND = "declared_source"
_DECLARED_SOURCE_NAME = "declared"


def _row_binding_for_provider(
    table_schema: TableSchema,
    schema_provider: SchemaProvider,
) -> GeneratedRowBinding:
    table_key = table_schema.table_key
    if isinstance(schema_provider, _SchemaDerivationProvider):
        derivation = schema_provider.derivation(table_key)
        if derivation is not None:
            return row_binding_for_table_schema(
                table_schema=table_schema,
                derivation_kind=derivation.source_kind,
                derivation_source=derivation.source_ref,
            )
    return row_binding_for_table_schema(
        table_schema=table_schema,
        derivation_kind=_DECLARED_SOURCE_KIND,
        derivation_source=_DECLARED_SOURCE_NAME,
    )


@lru_cache(maxsize=1)
def get_schema_service() -> SchemaService:
    """Return the canonical SchemaService configured for build.

    Returns
    -------
    SchemaService
        Configured schema service.
    """
    schema_provider = unified_schema_provider()

    def _row_binding_factory(table_schema: TableSchema) -> GeneratedRowBinding:
        return _row_binding_for_provider(table_schema, schema_provider)

    service = SchemaService(
        table_provider=schema_provider,
        dataset_provider=_BuildDatasetSchemaProvider(),
        row_binding_factory=_row_binding_factory,
    )
    set_schema_service(service)
    return service


def clear_schema_service_cache() -> None:
    """Clear cached SchemaService instance and core registry."""
    get_schema_service.cache_clear()
    clear_schema_service()
    clear_unified_provider_cache()


__all__ = [
    "clear_schema_service_cache",
    "get_schema_service",
]
