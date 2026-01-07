"""Build-owned SchemaService factory.

This module wires the canonical SchemaService to build-specific providers:
- Unified table schema provider (Hamilton inference + declared schemas)
- Arrow schema generation for boundary validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.core.imports.lazy import lazy_import
from codeintel.core.schemas import (
    SchemaService,
    clear_schema_service,
    set_schema_service,
)
from codeintel.core.schemas import (
    get_schema_service as get_core_schema_service,
)
from codeintel.core.schemas.resolution import (
    ResolvedArrowSchemaProvider,
    ResolvedSchemaProvider,
)
from codeintel.core.schemas.row_models import row_binding_for_table_schema

if TYPE_CHECKING:
    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.core.schemas.resolution import SchemaObservationProvider
    from codeintel.core.schemas.row_models import GeneratedRowBinding
    from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


@runtime_checkable
class _SchemaDerivationProvider(Protocol):
    """Protocol for schema providers that expose derivation metadata."""

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        """Return derivation metadata for the table key."""
        ...


_DECLARED_SOURCE_KIND = "declared_source"
_DECLARED_SOURCE_NAME = "declared"
_SCHEMA_SERVICE_STATE: dict[str, str | None] = {"fingerprint": None}


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


def configure_schema_service(
    *,
    runtime: HamiltonRuntimeBundle,
    observation_provider: SchemaObservationProvider | None = None,
) -> SchemaService:
    """Configure the canonical SchemaService for a runtime bundle.

    Parameters
    ----------
    runtime
        Runtime bundle providing schema index and DAG catalog metadata.
    observation_provider
        Optional observation provider for observed-first schema resolution.

    Returns
    -------
    SchemaService
        Configured schema service.
    """
    if runtime.fingerprint == _SCHEMA_SERVICE_STATE["fingerprint"]:
        return get_core_schema_service()

    schema_provider = _unified_schema_provider(runtime=runtime)
    resolved_provider = ResolvedSchemaProvider(
        observation_provider=observation_provider,
        fallback_provider=schema_provider,
    )
    arrow_provider = ResolvedArrowSchemaProvider(
        observation_provider=observation_provider,
        fallback_provider=schema_provider,
    )

    def _row_binding_factory(table_schema: TableSchema) -> GeneratedRowBinding:
        return _row_binding_for_provider(table_schema, resolved_provider)

    service = SchemaService(
        table_provider=resolved_provider,
        arrow_provider=arrow_provider,
        row_binding_factory=_row_binding_factory,
    )
    set_schema_service(service)
    _SCHEMA_SERVICE_STATE["fingerprint"] = runtime.fingerprint
    return service


def get_schema_service() -> SchemaService:
    """Return the configured SchemaService.

    Returns
    -------
    SchemaService
        Configured schema service.
    """
    return get_core_schema_service()


def clear_schema_service_cache() -> None:
    """Clear cached SchemaService instance and core registry."""
    _SCHEMA_SERVICE_STATE["fingerprint"] = None
    clear_schema_service()
    _clear_unified_provider_cache()


def _unified_schema_provider(*, runtime: HamiltonRuntimeBundle) -> SchemaProvider:
    module = lazy_import("codeintel.build.schemas.provider_unified")
    provider_factory = module.unified_schema_provider
    return provider_factory(runtime=runtime)


def _clear_unified_provider_cache() -> None:
    module = lazy_import("codeintel.build.schemas.provider_unified")
    cache_clear = module.clear_unified_provider_cache
    cache_clear()


__all__ = [
    "clear_schema_service_cache",
    "configure_schema_service",
    "get_schema_service",
]
