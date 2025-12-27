"""Runtime bundle types for Hamilton execution."""

from __future__ import annotations

from dataclasses import dataclass

import hamilton.driver as h_driver
from hamilton.caching.adapter import HamiltonCacheAdapter

from codeintel.build.hamilton.cache_index import CacheIndex
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.build.serving.semantic_compile import CompiledSemanticRegistry
from codeintel.core.hamilton.tag_query import TagQuery


@dataclass(frozen=True, slots=True)
class RuntimeKey:
    """Identity key for runtime bundle caching."""

    repo_fingerprint: str
    config_fingerprint: str
    modules_fingerprint: str
    build_profile: str | None


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    """Immutable runtime bundle for execution and planning."""

    driver: h_driver.Driver
    catalog: DagCatalog
    tag_query: TagQuery
    cache_adapter: HamiltonCacheAdapter | None
    cache_index: CacheIndex | None
    cache_key_resolver: CacheKeyResolver | None
    schema_index: SchemaIndex | None
    semantic_registry: CompiledSemanticRegistry | None
    fingerprint: str
    created_at_utc: str

    @property
    def dr(self) -> h_driver.Driver:
        """Compatibility alias for the Hamilton driver."""
        return self.driver


__all__ = [
    "RuntimeBundle",
    "RuntimeKey",
]
