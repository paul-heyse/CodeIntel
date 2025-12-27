"""Cache key resolver for Hamilton cache adapter state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from hamilton.caching import fingerprinting
from hamilton.caching.adapter import HamiltonCacheAdapter
from hamilton.caching.cache_key import create_cache_key

from codeintel.build.hamilton.cache_adapter import CacheStore


@dataclass(frozen=True, slots=True)
class CacheKeySnapshot:
    """Snapshot of cache key and data version for a node."""

    cache_key: str | None
    cache_version: str | None


@dataclass(frozen=True, slots=True)
class CacheKeyResolver:
    """Resolve cache keys for planning and cache adapter snapshots."""

    code_versions: Mapping[str, str] = field(default_factory=dict)
    node_dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cache_store: CacheStore | None = None

    @staticmethod
    def resolve(
        adapter: HamiltonCacheAdapter,
        *,
        run_id: str,
        node_name: str,
        task_id: str | None = None,
    ) -> CacheKeySnapshot:
        """Return cache key and version snapshot for the requested node.

        Returns
        -------
        CacheKeySnapshot
            Snapshot containing cache key and cache version.
        """
        cache_key = _peek_cache_key(adapter, run_id, node_name, task_id)
        cache_version = _peek_cache_version(adapter, run_id, node_name, cache_key, task_id)
        return CacheKeySnapshot(cache_key=cache_key, cache_version=cache_version)

    def resolve_node_versions(
        self,
        *,
        nodes: Iterable[str],
        input_values: Mapping[str, object],
    ) -> dict[str, str]:
        """Resolve cache keys for nodes using dependency data versions.

        Returns
        -------
        dict[str, str]
            Mapping of node names to resolved cache keys.
        """
        node_set = set(nodes)
        if not node_set:
            return {}

        data_versions: dict[str, str | None] = {}
        input_versions: dict[str, str | None] = {}
        cache_keys: dict[str, str] = {}

        for node in _topo_sort_nodes(node_set, self.node_dependencies):
            dep_versions = _resolve_dependency_versions(
                node,
                node_set=node_set,
                node_dependencies=self.node_dependencies,
                data_versions=data_versions,
                input_values=input_values,
                input_versions=input_versions,
            )
            if dep_versions is None:
                data_versions[node] = None
                continue

            code_version = self.code_versions.get(node)
            if not code_version:
                data_versions[node] = None
                continue

            cache_key = create_cache_key(
                node_name=node,
                code_version=code_version,
                dependencies_data_versions=dep_versions,
            )
            cache_keys[node] = cache_key

            if self.cache_store is None:
                data_versions[node] = None
                continue

            data_versions[node] = self.cache_store.get_data_version(cache_key)

        return cache_keys


def _topo_sort_nodes(
    node_set: set[str],
    node_dependencies: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    in_degree: dict[str, int] = dict.fromkeys(node_set, 0)
    graph: dict[str, list[str]] = {node: [] for node in node_set}

    for node in node_set:
        for dep in node_dependencies.get(node, ()):
            if dep not in node_set:
                continue
            graph[dep].append(node)
            in_degree[node] += 1

    ready = sorted([node for node, degree in in_degree.items() if degree == 0])
    ordered: list[str] = []

    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for neighbor in sorted(graph[current]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                ready.append(neighbor)
        ready.sort()

    if len(ordered) != len(node_set):
        return tuple(sorted(node_set))
    return tuple(ordered)


def _resolve_dependency_versions(
    node: str,
    *,
    node_set: set[str],
    node_dependencies: Mapping[str, tuple[str, ...]],
    data_versions: Mapping[str, str | None],
    input_values: Mapping[str, object],
    input_versions: dict[str, str | None],
) -> dict[str, str] | None:
    dependencies = node_dependencies.get(node, ())
    if not dependencies:
        return {}

    resolved: dict[str, str] = {}
    for dep in dependencies:
        if dep in node_set:
            version = data_versions.get(dep)
        else:
            version = _input_version(dep, input_values, input_versions)
        if version is None:
            return None
        resolved[dep] = version
    return resolved


def _input_version(
    name: str,
    input_values: Mapping[str, object],
    input_versions: dict[str, str | None],
) -> str | None:
    if name in input_versions:
        return input_versions[name]
    if name not in input_values:
        input_versions[name] = None
        return None
    version = fingerprinting.hash_value(input_values[name])
    if version == fingerprinting.UNHASHABLE:
        input_versions[name] = None
        return None
    input_versions[name] = version
    return version


def _peek_cache_key(
    adapter: HamiltonCacheAdapter,
    run_id: str,
    node_name: str,
    task_id: str | None,
) -> str | None:
    cache_key = adapter.get_cache_key(run_id=run_id, node_name=node_name, task_id=task_id)
    return cache_key if isinstance(cache_key, str) else None


def _peek_cache_version(
    adapter: HamiltonCacheAdapter,
    run_id: str,
    node_name: str,
    cache_key: str | None,
    task_id: str | None,
) -> str | None:
    data_version = adapter.get_data_version(
        run_id=run_id,
        node_name=node_name,
        cache_key=cache_key,
        task_id=task_id,
    )
    return data_version if isinstance(data_version, str) else None


__all__ = [
    "CacheKeyResolver",
    "CacheKeySnapshot",
]
