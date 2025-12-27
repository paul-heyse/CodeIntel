"""Cache key resolver for Hamilton cache adapter state."""

from __future__ import annotations

from dataclasses import dataclass

from hamilton.caching.adapter import HamiltonCacheAdapter, NodeRoleInTaskExecution


@dataclass(frozen=True, slots=True)
class CacheKeySnapshot:
    """Snapshot of cache key and data version for a node."""

    cache_key: str | None
    cache_version: str | None


class CacheKeyResolver:
    """Resolve cache keys and data versions from a Hamilton cache adapter."""

    def resolve(
        self,
        adapter: HamiltonCacheAdapter,
        *,
        run_id: str,
        node_name: str,
        task_id: str | None = None,
    ) -> CacheKeySnapshot:
        """Return cache key and version snapshot for the requested node."""
        cache_key = _peek_cache_key(adapter, run_id, node_name, task_id)
        cache_version = _peek_cache_version(adapter, run_id, node_name, task_id)
        return CacheKeySnapshot(cache_key=cache_key, cache_version=cache_version)


def _peek_cache_key(
    adapter: HamiltonCacheAdapter,
    run_id: str,
    node_name: str,
    task_id: str | None,
) -> str | None:
    node_role = adapter._get_node_role(run_id=run_id, node_name=node_name, task_id=task_id)
    cache_keys = adapter.cache_keys.get(run_id, {})
    if node_role == NodeRoleInTaskExecution.INSIDE:
        nested = cache_keys.get(node_name, {})
        if isinstance(nested, dict):
            return nested.get(task_id)
        return None
    value = cache_keys.get(node_name)
    return value if isinstance(value, str) else None


def _peek_cache_version(
    adapter: HamiltonCacheAdapter,
    run_id: str,
    node_name: str,
    task_id: str | None,
) -> str | None:
    node_role = adapter._get_node_role(run_id=run_id, node_name=node_name, task_id=task_id)
    data_versions = adapter.data_versions.get(run_id, {})
    if node_role == NodeRoleInTaskExecution.INSIDE:
        nested = data_versions.get(node_name, {})
        if isinstance(nested, dict):
            version = nested.get(task_id)
            return version if isinstance(version, str) else None
        return None
    value = data_versions.get(node_name)
    return value if isinstance(value, str) else None


__all__ = [
    "CacheKeyResolver",
    "CacheKeySnapshot",
]
