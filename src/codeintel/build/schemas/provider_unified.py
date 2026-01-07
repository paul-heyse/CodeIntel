"""Unified schema provider rooted in the global Hamilton DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.core.schemas.authority import SchemaAuthority
from codeintel.core.schemas.declared import source_declared_schema_provider
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.core.schemas.authority import SchemaDerivation
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


_DECLARED_SOURCE_KIND = "declared_source"
_DECLARED_SOURCE_NAME = "declared"


_DECLARED_PROVIDER_CACHE: dict[str, SchemaProvider] = {}
_PROVIDER_CACHE: dict[tuple[str, bool], UnifiedSchemaProvider] = {}


def _schema_index_from_runtime(runtime: HamiltonRuntimeBundle) -> SchemaIndex:
    schema_index = runtime.schema_index
    if schema_index is None:
        msg = "HamiltonRuntimeBundle.schema_index is required to build a schema provider"
        raise ValueError(msg)
    return schema_index


def declared_schema_provider(*, runtime: HamiltonRuntimeBundle) -> SchemaProvider:
    """Return a source-only declared schema provider for build usage.

    Returns
    -------
    SchemaProvider
        Provider exposing only source table schemas (excluding DAG outputs).
    """
    cache_key = runtime.fingerprint
    cached = _DECLARED_PROVIDER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    exclude_table_keys = frozenset(runtime.catalog.table_outputs)
    provider = source_declared_schema_provider(exclude_table_keys=exclude_table_keys)
    _DECLARED_PROVIDER_CACHE[cache_key] = provider
    return provider


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
    _view_schema_cache: dict[str, TableSchema] = field(default_factory=dict, repr=False)
    _view_schema_loaded: bool = field(default=False, repr=False)

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
        schema = self.schema_authority.get_table_schema(table_key)
        if schema is not None:
            return schema
        return self._view_schema(table_key)

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
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known table schemas.

        Yields
        ------
        TableSchema
            Table schemas from the DAG, declared providers, and derived views.
        """
        seen: set[str] = set()
        for schema in self.schema_authority.iter_table_schemas():
            seen.add(schema.table_key)
            yield schema
        for table_key, view_schema in self._view_schema_map().items():
            if table_key in seen:
                continue
            yield view_schema

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

    def _view_schema_map(self) -> dict[str, TableSchema]:
        if self._view_schema_loaded:
            return self._view_schema_cache
        # View schemas now flow from observed outputs; avoid SQL-based inference here.
        self._view_schema_cache = {}
        self._view_schema_loaded = True
        return self._view_schema_cache

    def _view_schema(self, table_key: str) -> TableSchema | None:
        return self._view_schema_map().get(table_key)

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


def _build_provider(
    *,
    runtime: HamiltonRuntimeBundle,
    allow_inference: bool,
) -> UnifiedSchemaProvider:
    cache_key = (runtime.fingerprint, allow_inference)
    cached = _PROVIDER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    provider = UnifiedSchemaProvider(
        declared=declared_schema_provider(runtime=runtime),
        schema_index=_schema_index_from_runtime(runtime),
        allow_inference=allow_inference,
    )
    _PROVIDER_CACHE[cache_key] = provider
    return provider


def unified_schema_provider(*, runtime: HamiltonRuntimeBundle) -> UnifiedSchemaProvider:
    """Return the DAG-first schema provider.

    Returns
    -------
    UnifiedSchemaProvider
        Provider with inference enabled.
    """
    return _build_provider(runtime=runtime, allow_inference=True)


def non_inferable_schema_provider(*, runtime: HamiltonRuntimeBundle) -> UnifiedSchemaProvider:
    """Return a schema provider with inference disabled.

    Returns
    -------
    UnifiedSchemaProvider
        Provider with inference disabled.
    """
    return _build_provider(runtime=runtime, allow_inference=False)


def clear_unified_provider_cache() -> None:
    """Clear the unified provider cache.

    Useful for testing when schema definitions or targets may change.
    """
    _DECLARED_PROVIDER_CACHE.clear()
    _PROVIDER_CACHE.clear()


__all__ = [
    "UnifiedSchemaProvider",
    "clear_unified_provider_cache",
    "declared_schema_provider",
    "non_inferable_schema_provider",
    "unified_schema_provider",
]
