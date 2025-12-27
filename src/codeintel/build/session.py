"""Session-scoped cache state for build operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.cache_index import CacheIndex
    from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
    from codeintel.config.primitives import SnapshotRef

__all__ = [
    "BuildSession",
]


@dataclass
class BuildSession:
    """Session-scoped caches and state for a build run.

    Provides cache key and cache-hit information for build targets.

    Attributes
    ----------
    snapshot
        The repository snapshot for this build session.
    cache_index
        Cache index used to probe cache presence.
    cache_key_resolver
        Cache key resolver used to compute expected cache keys.
    input_values
        External input values used for cache key hashing.

    Examples
    --------
    >>> session = BuildSession(snapshot, cache_index, cache_key_resolver, {})
    >>> session.preload_cache_keys()
    """

    snapshot: SnapshotRef
    cache_index: CacheIndex | None
    cache_key_resolver: CacheKeyResolver | None
    input_values: Mapping[str, object]
    _cache_keys: dict[str, str] = field(default_factory=dict, repr=False)
    _cache_keys_preloaded: bool = field(default=False, repr=False)

    def preload_cache_keys(self) -> None:
        """Compute cache keys for all resolvable nodes in the graph."""
        if self._cache_keys_preloaded:
            return
        resolver = self.cache_key_resolver
        if resolver is None:
            self._cache_keys_preloaded = True
            return
        node_set = set(resolver.node_dependencies)
        node_set.difference_update(self.input_values)
        self._cache_keys = resolver.resolve_node_versions(
            nodes=node_set,
            input_values=self.input_values,
        )
        self._cache_keys_preloaded = True

    def cache_key_for_node(self, node_name: str) -> str | None:
        """Return the computed cache key for a node, if available.

        Returns
        -------
        str | None
            Cache key for the node when available.
        """
        if not self._cache_keys_preloaded:
            self.preload_cache_keys()
        return self._cache_keys.get(node_name)

    def cache_hit(self, node_name: str) -> bool:
        """Return True when a cache entry exists for the node.

        Returns
        -------
        bool
            True when a cache entry exists for the node.
        """
        cache_key = self.cache_key_for_node(node_name)
        if cache_key is None:
            return False
        if self.cache_index is None:
            return False
        return self.cache_index.has(node=node_name, version=cache_key)

    def clear_caches(self) -> None:
        """Clear all cached cache keys."""
        self._cache_keys.clear()
        self._cache_keys_preloaded = False
