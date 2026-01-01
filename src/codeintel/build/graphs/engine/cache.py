"""Graph caching utilities.

This module provides the caching layer for graph engines,
supporting efficient reuse of computed graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.cache import CacheStatsCollector

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from codeintel.build.graphs.engine.protocol import GraphKind
    from codeintel.core.cache import CacheStats


@dataclass(frozen=True, slots=True)
class GraphCacheMetadata:
    """Parquet metadata used for cache invalidation."""

    repo: str
    commit: str
    build_id: str | None
    schema_hash: str | None


@dataclass(slots=True)
class GraphCacheEntry:
    """Cached graph plus its associated metadata."""

    graph: nx.Graph
    metadata: GraphCacheMetadata | None


class GraphCache:
    """Cache for storing computed graphs by kind and metadata.

    This class manages the lifecycle of graph instances, supporting
    both explicit seeding and lazy loading of graphs.
    """

    def __init__(self) -> None:
        """Initialize an empty graph cache."""
        self._cache: dict[GraphKind, GraphCacheEntry] = {}
        self._stats = CacheStatsCollector()

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics.

        Returns
        -------
        CacheStats
            Current cache statistics including hits, misses, and size.
        """
        return self._stats.to_stats(size=len(self._cache))

    def seed(
        self,
        kind: GraphKind,
        graph: nx.Graph | None,
        *,
        metadata: GraphCacheMetadata | None = None,
    ) -> None:
        """
        Pre-populate the cache when a graph is already available.

        Parameters
        ----------
        kind : GraphKind
            Type of graph being cached.
        graph : nx.Graph | None
            Graph instance to cache, or None to skip.
        metadata : GraphCacheMetadata | None
            Optional metadata used to validate cache entries.
        """
        if graph is None:
            return
        self._cache[kind] = GraphCacheEntry(graph=graph, metadata=metadata)

    def get(
        self,
        kind: GraphKind,
        loader: Callable[[], nx.Graph],
        *,
        metadata: GraphCacheMetadata | None = None,
    ) -> nx.Graph:
        """
        Retrieve a graph from cache or load it using the provided loader.

        Parameters
        ----------
        kind : GraphKind
            Type of graph to retrieve.
        loader : Callable[[], nx.Graph]
            Function to load the graph if not cached.
        metadata : GraphCacheMetadata | None
            Optional metadata used to validate cache entries.

        Returns
        -------
        nx.Graph
            Cached or freshly loaded graph.
        """
        entry = self._cache.get(kind)
        if entry is None:
            self._stats.record_miss()
            graph = loader()
            self._cache[kind] = GraphCacheEntry(graph=graph, metadata=metadata)
            return graph
        if metadata is None or entry.metadata == metadata:
            self._stats.record_hit()
            return entry.graph
        self._stats.record_miss()
        graph = loader()
        self._cache[kind] = GraphCacheEntry(graph=graph, metadata=metadata)
        return graph

    def clear(self) -> int:
        """Clear all cached graphs.

        Returns
        -------
        int
            Number of graphs cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        self._stats.reset()
        return count

    def invalidate(self, kind: GraphKind) -> None:
        """
        Invalidate a specific graph kind from the cache.

        Parameters
        ----------
        kind : GraphKind
            Type of graph to invalidate.
        """
        self._cache.pop(kind, None)

    def has(self, kind: GraphKind, *, metadata: GraphCacheMetadata | None = None) -> bool:
        """
        Check if a graph kind is in the cache.

        Parameters
        ----------
        kind : GraphKind
            Type of graph to check.
        metadata : GraphCacheMetadata | None
            Optional metadata to compare against the cached entry.

        Returns
        -------
        bool
            True if the graph is cached and metadata matches when provided.
        """
        entry = self._cache.get(kind)
        if entry is None:
            return False
        if metadata is None:
            return True
        return entry.metadata == metadata


__all__ = ["GraphCache", "GraphCacheEntry", "GraphCacheMetadata"]
