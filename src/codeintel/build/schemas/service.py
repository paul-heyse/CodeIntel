"""Build-owned SchemaService factory.

This module wires the canonical SchemaService to build-specific providers:
- Unified table schema provider (Hamilton inference + declared schemas)
- DatasetSchema registry (Pandera-backed unified schemas)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

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

if TYPE_CHECKING:
    from collections.abc import Iterable


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


@lru_cache(maxsize=1)
def get_schema_service() -> SchemaService:
    """Return the canonical SchemaService configured for build.

    Returns
    -------
    SchemaService
        Configured schema service.
    """
    service = SchemaService(
        table_provider=unified_schema_provider(),
        dataset_provider=_BuildDatasetSchemaProvider(),
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
