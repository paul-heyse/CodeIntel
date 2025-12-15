"""Unified schema provider with fallback chain.

This module provides a unified schema provider that resolves schemas through
a three-tier fallback chain:

1. Hamilton-native inference (for q__-driven Ibis compute nodes)
2. Target-declared schemas from OutputContract.tables (for plugin wrappers)
3. Raw declared schemas from declared_schema_provider() (for source tables)

This enables all table keys to be resolvable through a single interface while
preferring dynamically inferred schemas where available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas.provider_declared import declared_schema_provider

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import OutputTarget, TargetGraph
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


def _get_target_graph() -> TargetGraph:
    """Return the target graph, importing lazily to avoid circular dependencies.

    Returns
    -------
    TargetGraph
        The singleton target graph instance.
    """
    # Deferred import to avoid circular dependency at module load time.
    # This module is imported by registry.py, which also imports get_target_graph.
    from codeintel.build.registry import get_target_graph as _get_graph  # noqa: PLC0415

    return _get_graph()


def _find_producing_target(table_key: str) -> OutputTarget | None:
    """Find the target that produces a given table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    OutputTarget | None
        The target producing this table key, or None if not found.
    """
    graph = _get_target_graph()
    for target in graph.all_targets:
        if table_key in target.contract.table_keys:
            return target
    return None


@dataclass
class UnifiedSchemaProvider:
    """Schema provider with fallback chain: inferred -> target-declared -> declared.

    This provider implements a three-tier resolution strategy:

    1. **Hamilton-native inference**: For table keys produced by q__-driven
       Ibis compute nodes, infer the schema by executing the compute function
       in an ephemeral environment.

    2. **Target-declared schemas**: For table keys produced by targets that
       declare their output schemas in OutputContract.tables (e.g., plugin
       wrappers, legacy compute).

    3. **Raw declared schemas**: Fall back to the declared_schema_provider()
       for source tables or tables not yet migrated to the target system.

    Parameters
    ----------
    declared
        The base declared schema provider for fallback.
    inferable_table_keys
        Set of table keys that can be inferred via Hamilton native compute.
    fallback_to_declared_on_error
        When True, fall back to declared schemas if inference fails.
        Defaults to True.

    Examples
    --------
    >>> from codeintel.build.schemas.provider_unified import unified_schema_provider
    >>> provider = unified_schema_provider()
    >>> schema = provider.get_table_schema("analytics.function_metrics")
    >>> schema is not None
    True
    """

    declared: SchemaProvider
    inferable_table_keys: frozenset[str]
    fallback_to_declared_on_error: bool = True
    _cache: dict[str, TableSchema] = field(default_factory=dict)

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Resolve schema via fallback chain.

        Resolution order:
        1. Hamilton-native inference (for inferable targets)
        2. Target-declared output schema (for non-inferable targets)
        3. Raw declared schema (for source/raw tables)

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Resolved schema, or None when unknown to all sources.
        """
        # Check cache first
        cached = self._cache.get(table_key)
        if cached is not None:
            return cached

        # 1. Try Hamilton-native inference
        if table_key in self.inferable_table_keys:
            try:
                # Lazy import to avoid circular dependency.
                from codeintel.build.schemas.provider_hamilton import (  # noqa: PLC0415
                    infer_schema_for_table_key,
                )

                inferred = infer_schema_for_table_key(
                    table_key=table_key,
                    declared_provider=self.declared,
                )
            except Exception:
                if not self.fallback_to_declared_on_error:
                    raise
                # Fall through to next resolution step
            else:
                self._cache[table_key] = inferred
                return inferred

        # 2. Try target-declared output schema
        target = _find_producing_target(table_key)
        if target is not None:
            output_schema = target.contract.get_table(table_key)
            if output_schema is not None:
                self._cache[table_key] = output_schema
                return output_schema

        # 3. Fall back to raw declared schema
        declared_schema = self.declared.get_table_schema(table_key)
        if declared_schema is not None:
            self._cache[table_key] = declared_schema
        return declared_schema

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
        """Iterate all known table schemas from all sources.

        Yields schemas in priority order (inferred > target-declared > declared),
        deduplicating by table_key.

        Yields
        ------
        TableSchema
            Each known table schema.
        """
        seen: set[str] = set()

        # 1. Yield inferred schemas first (highest priority)
        for table_key in sorted(self.inferable_table_keys):
            if table_key in seen:
                continue
            schema = self.get_table_schema(table_key)
            if schema is not None:
                seen.add(table_key)
                yield schema

        # 2. Yield target-declared schemas
        graph = _get_target_graph()
        for target in graph.all_targets:
            for schema in target.contract.tables:
                if schema.table_key not in seen:
                    seen.add(schema.table_key)
                    yield schema

        # 3. Yield declared schemas (lowest priority)
        for schema in self.declared.iter_table_schemas():
            if schema.table_key not in seen:
                seen.add(schema.table_key)
                yield schema


@lru_cache
def unified_schema_provider() -> UnifiedSchemaProvider:
    """Return the unified schema provider with full fallback chain.

    The provider is cached for the lifetime of the process. Use
    `clear_unified_provider_cache()` to reset if needed.

    Returns
    -------
    UnifiedSchemaProvider
        Cached unified provider instance.

    Examples
    --------
    >>> provider = unified_schema_provider()
    >>> schema = provider.require_table_schema("analytics.function_metrics")
    >>> schema.table_key
    'analytics.function_metrics'
    """
    # Lazy import to avoid circular dependency.
    from codeintel.build.schemas.provider_hamilton import (  # noqa: PLC0415
        inferable_native_table_keys,
    )

    declared = declared_schema_provider()
    graph = _get_target_graph()
    inferable = inferable_native_table_keys(graph=graph)

    return UnifiedSchemaProvider(
        declared=declared,
        inferable_table_keys=inferable,
        fallback_to_declared_on_error=True,
    )


def clear_unified_provider_cache() -> None:
    """Clear the unified provider cache.

    Useful for testing when schema definitions or targets may change.
    """
    unified_schema_provider.cache_clear()


__all__ = [
    "UnifiedSchemaProvider",
    "clear_unified_provider_cache",
    "unified_schema_provider",
]
