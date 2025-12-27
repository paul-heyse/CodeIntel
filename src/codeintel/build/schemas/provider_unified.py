"""Unified schema provider rooted in the global Hamilton DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.schemas.authority import SchemaAuthority
from codeintel.core.schemas.declared import source_declared_schema_provider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


_DECLARED_SOURCE_KIND = "declared_source"
_DECLARED_SOURCE_NAME = "declared"


@lru_cache(maxsize=1)
def declared_schema_provider() -> SchemaProvider:
    """Return a source-only declared schema provider for build usage.

    Returns
    -------
    SchemaProvider
        Provider exposing only source table schemas (excluding DAG outputs).
    """
    service = get_target_metadata_service()
    exclude_table_keys = service.system.all_table_keys
    return source_declared_schema_provider(exclude_table_keys=exclude_table_keys)


@dataclass(frozen=True, slots=True)
class _SchemaIndexProvider:
    schema_index: SchemaIndex
    allow_inference: bool = True

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        return self.schema_index.get_table_schema(
            table_key,
            allow_inference=self.allow_inference,
        )

    def require_table_schema(self, table_key: str) -> TableSchema:
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        return self.schema_index.iter_table_schemas(allow_inference=self.allow_inference)


def _dag_sources(schema_index: SchemaIndex) -> Mapping[str, tuple[str, str]]:
    return {
        table_key: (derivation.kind, derivation.source)
        for table_key, derivation in schema_index.derivations.items()
    }


@dataclass
class UnifiedSchemaProvider:
    """Schema provider that resolves DAG outputs first, sources last."""

    declared: SchemaProvider
    schema_index: SchemaIndex
    allow_inference: bool = True
    fallback_to_override_on_error: bool = True
    schema_authority: SchemaAuthority = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the SchemaAuthority backing this provider."""
        dag_provider = _SchemaIndexProvider(
            schema_index=self.schema_index,
            allow_inference=self.allow_inference,
        )
        self.schema_authority = SchemaAuthority(
            dag_provider=dag_provider,
            declared_provider=self.declared,
            dag_sources=_dag_sources(self.schema_index),
            declared_source_kind=_DECLARED_SOURCE_KIND,
            declared_source_ref=_DECLARED_SOURCE_NAME,
        )

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return the schema for a table key if known.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Resolved table schema, or None if not found.
        """
        return self.schema_authority.get_table_schema(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key, raising when unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Resolved schema.

        Raises
        ------
        KeyError
            If table_key is unknown to all providers in the fallback chain.
        """
        schema = self.schema_authority.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known table schemas.

        Returns
        -------
        Iterable[TableSchema]
            Table schemas from the DAG and declared providers.
        """
        return self.schema_authority.iter_table_schemas()

    @property
    def inferable_table_keys(self) -> frozenset[str]:
        """Return table keys that can be inferred from the DAG.

        Returns
        -------
        frozenset[str]
            Table keys eligible for inference.
        """
        return self.schema_index.inferable_table_keys

    def derivation(self, table_key: str) -> SchemaDerivation | None:
        """Return SchemaAuthority derivation metadata when available.

        Returns
        -------
        SchemaDerivation | None
            Derivation metadata when available, otherwise None.
        """
        return self.schema_authority.derivation(table_key)

    def with_inference(self, *, allow_inference: bool) -> UnifiedSchemaProvider:
        """Return a copy with inference enabled or disabled.

        Returns
        -------
        UnifiedSchemaProvider
            Provider configured with the requested inference setting.
        """
        return UnifiedSchemaProvider(
            declared=self.declared,
            schema_index=self.schema_index,
            allow_inference=allow_inference,
            fallback_to_override_on_error=self.fallback_to_override_on_error,
        )


@lru_cache(maxsize=2)
def _build_provider(*, allow_inference: bool) -> UnifiedSchemaProvider:
    service = get_target_metadata_service()
    return UnifiedSchemaProvider(
        declared=declared_schema_provider(),
        schema_index=service.schema_index,
        allow_inference=allow_inference,
    )


def unified_schema_provider() -> UnifiedSchemaProvider:
    """Return the DAG-first schema provider.

    Returns
    -------
    UnifiedSchemaProvider
        Provider with inference enabled.
    """
    return _build_provider(allow_inference=True)


@lru_cache(maxsize=1)
def non_inferable_schema_provider() -> UnifiedSchemaProvider:
    """Return a schema provider with inference disabled.

    Returns
    -------
    UnifiedSchemaProvider
        Provider with inference disabled.
    """
    return _build_provider(allow_inference=False)


def clear_unified_provider_cache() -> None:
    """Clear the unified provider cache.

    Useful for testing when schema definitions or targets may change.
    """
    _build_provider.cache_clear()


__all__ = [
    "UnifiedSchemaProvider",
    "clear_unified_provider_cache",
    "declared_schema_provider",
    "non_inferable_schema_provider",
    "unified_schema_provider",
]
