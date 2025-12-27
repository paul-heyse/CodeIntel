"""Helpers for cache-backed state tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hamilton.caching.cache_key import create_cache_key
from hamilton.caching.stores.sqlite import SQLiteMetadataStore

from codeintel.build.hamilton.cache_adapter import CacheStore
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver


@dataclass(frozen=True, slots=True)
class CacheFixture:
    """Cache store and resolver fixture bundle."""

    cache_store: CacheStore
    cache_key_resolver: CacheKeyResolver


def make_cache_store(path: Path) -> CacheStore:
    """Create a cache store rooted at the given path.

    Returns
    -------
    CacheStore
        Cache store backed by a metadata store on disk.
    """
    metadata_store = SQLiteMetadataStore(path=str(path))
    return CacheStore(metadata_store=metadata_store, result_store=None)


def make_cache_key_resolver(
    *,
    node_dependencies: Mapping[str, tuple[str, ...]],
    cache_store: CacheStore | None = None,
) -> CacheKeyResolver:
    """Create a CacheKeyResolver with deterministic code versions.

    Returns
    -------
    CacheKeyResolver
        Cache key resolver with deterministic code versions.
    """
    code_versions = {node: f"code::{node}" for node in node_dependencies}
    return CacheKeyResolver(
        code_versions=code_versions,
        node_dependencies=node_dependencies,
        cache_store=cache_store,
    )


def seed_cache_store(
    cache_store: CacheStore,
    resolver: CacheKeyResolver,
    *,
    nodes: Iterable[str] | None = None,
    run_id: str = "test",
) -> dict[str, str]:
    """Seed cache entries for the specified nodes.

    Returns
    -------
    dict[str, str]
        Mapping of node name to cache key.
    """
    node_set = set(nodes or resolver.node_dependencies)
    order = _topo_sort_nodes(node_set, resolver.node_dependencies)
    cache_keys: dict[str, str] = {}
    data_versions: dict[str, str] = {}

    for node in order:
        if node not in node_set:
            continue
        dep_versions = {
            dep: data_versions[dep]
            for dep in resolver.node_dependencies.get(node, ())
            if dep in node_set
        }
        cache_key = create_cache_key(
            node_name=node,
            code_version=resolver.code_versions[node],
            dependencies_data_versions=dep_versions,
        )
        data_version = f"data::{node}"
        cache_store.metadata_store.set(
            cache_key=cache_key,
            data_version=data_version,
            run_id=run_id,
            node_name=node,
            code_version=resolver.code_versions[node],
        )
        cache_keys[node] = cache_key
        data_versions[node] = data_version

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


__all__ = [
    "CacheFixture",
    "make_cache_key_resolver",
    "make_cache_store",
    "seed_cache_store",
]
