"""Execution input bundle injected into Hamilton runs."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.hamilton.cache_index import CacheIndex
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.planning.model import PlanRequest
from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.core.hamilton.tag_query import TagQuery
from codeintel.serving.semantic_compile import CompiledSemanticRegistry


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


def execution_input_mapping(inputs: ExecutionInputs) -> dict[str, object]:
    """Return Hamilton execution inputs as a dict for cache hashing.

    Parameters
    ----------
    inputs
        Execution inputs to convert into a mapping.

    Returns
    -------
    dict[str, object]
        Mapping of input names to values for execution/caching.
    """
    mapping: dict[str, object] = {
        "env": inputs.env,
        "catalog": inputs.catalog,
    }
    optional: dict[str, object | None] = {
        "tag_query": inputs.tag_query,
        "cache_index": inputs.cache_index,
        "cache_key_resolver": inputs.cache_key_resolver,
        "schema_index": inputs.schema_index,
        "semantic_registry": inputs.semantic_registry,
        "runtime_fingerprint": inputs.runtime_fingerprint,
        "plan_request": inputs.plan_request,
    }
    mapping.update({key: value for key, value in optional.items() if value is not None})
    return mapping


__all__ = ["ExecutionInputs", "execution_input_mapping"]
