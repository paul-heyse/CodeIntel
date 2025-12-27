"""Execution input bundle injected into Hamilton runs."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.hamilton.cache_index import CacheIndex
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.planning.model import PlanRequest
from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.build.serving.semantic_compile import CompiledSemanticRegistry
from codeintel.core.hamilton.tag_query import TagQuery


@dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """Stable input contract for DAG execution."""

    env: BuildEnv
    catalog: DagCatalog
    tag_query: TagQuery | None = None
    cache_index: CacheIndex | None = None
    cache_key_resolver: CacheKeyResolver | None = None
    schema_index: SchemaIndex | None = None
    semantic_registry: CompiledSemanticRegistry | None = None
    runtime_fingerprint: str | None = None
    plan_request: PlanRequest | None = None


__all__ = ["ExecutionInputs"]
