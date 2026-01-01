"""Runtime bundle types for Hamilton execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import hamilton.driver as h_driver
from hamilton.caching.adapter import HamiltonCacheAdapter

from codeintel.build.hamilton.cache_index import CacheIndex
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.core.runtime.variants import VariantConfig
from codeintel.runtime.module_resolver import ModuleProvenance
from codeintel.runtime.plugins.spec import TargetPack
from codeintel.serving.semantic_compile import CompiledSemanticRegistry


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
    variants: VariantConfig
    cache_adapter: HamiltonCacheAdapter | None
    cache_index: CacheIndex | None
    cache_key_resolver: CacheKeyResolver | None
    schema_index: SchemaIndex | None
    semantic_registry: CompiledSemanticRegistry | None
    packs: tuple[TargetPack, ...]
    module_provenance: Mapping[str, ModuleProvenance]
    modules_fingerprint: str
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
