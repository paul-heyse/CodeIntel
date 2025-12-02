"""Graph caching utilities.

This module provides the caching layer for graph engines,
supporting efficient reuse of computed graphs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from codeintel.graphs.engine.protocol import GraphKind

if TYPE_CHECKING:
    import networkx as nx


class GraphCache:
    """Cache for storing computed graphs by kind.

    This class manages the lifecycle of graph instances, supporting
    both explicit seeding and lazy loading of graphs.
    """

    def __init__(self) -> None:
        """Initialize an empty graph cache."""
        self._cache: dict[GraphKind, nx.Graph] = {}

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
            graph = loader()
            self._cache[kind] = graph
        return graph

    def clear(self) -> None:
        """Clear all cached graphs."""
        self._cache.clear()

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
