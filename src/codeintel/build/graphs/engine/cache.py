"""Graph caching utilities.

This module provides the caching layer for graph engines,
supporting efficient reuse of computed graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.core.cache import CacheStatsCollector

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from codeintel.build.graphs.engine.protocol import GraphKind
    from codeintel.core.cache import CacheStats


class GraphCache:
    """Cache for storing computed graphs by kind.

    This class manages the lifecycle of graph instances, supporting
    both explicit seeding and lazy loading of graphs.
    """

    def __init__(self) -> None:
        """Initialize an empty graph cache."""
        self._cache: dict[GraphKind, nx.Graph] = {}
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

    def seed(self, kind: GraphKind, graph: nx.Graph | None) -> None:
        """
        Pre-populate the cache when a graph is already available.

        Parameters
        ----------
        kind : GraphKind
            Type of graph being cached.
        graph : nx.Graph | None
            Graph instance to cache, or None to skip.
        """
        if graph is None:
            return
        self._cache[kind] = graph

    def get(self, kind: GraphKind, loader: Callable[[], nx.Graph]) -> nx.Graph:
        """
        Retrieve a graph from cache or load it using the provided loader.

        Parameters
        ----------
        kind : GraphKind
            Type of graph to retrieve.
        loader : Callable[[], nx.Graph]
            Function to load the graph if not cached.

        Returns
        -------
        nx.Graph
            Cached or freshly loaded graph.
        """
        graph = self._cache.get(kind)
        if graph is None:
            self._stats.record_miss()
            graph = loader()
            self._cache[kind] = graph
        else:
            self._stats.record_hit()
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

    def has(self, kind: GraphKind) -> bool:
        """
        Check if a graph kind is in the cache.

        Parameters
        ----------
        kind : GraphKind
            Type of graph to check.

        Returns
        -------
        bool
            True if the graph is cached.
        """
        return kind in self._cache


__all__ = ["GraphCache"]
