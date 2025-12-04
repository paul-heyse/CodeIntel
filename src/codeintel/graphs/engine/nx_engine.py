"""NetworkX-backed GraphEngine implementation.

This module provides the primary GraphEngine implementation using
NetworkX for graph representation and DuckDB for data loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import networkx as nx

from codeintel.graphs.engine import views
from codeintel.graphs.engine.backend import BackendEnablement
from codeintel.graphs.engine.cache import GraphCache
from codeintel.graphs.engine.protocol import GraphKind

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass
class NxGraphEngine:
    """NetworkX-backed GraphEngine powered by DuckDB views."""

    gateway: StorageGateway
    snapshot: SnapshotRef
    use_gpu: bool = False
    effective_use_gpu: bool = False
    backend_info: BackendEnablement | None = None
    _cache: GraphCache = field(default_factory=GraphCache)

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
        self._cache.seed(kind, graph)

    @property
    def repo(self) -> str:
        """Repository identifier for the bound snapshot."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the bound snapshot."""
        return self.snapshot.commit

    def call_graph(self) -> nx.DiGraph:
        """
        Return the call graph for the configured snapshot.

        Returns
        -------
        nx.DiGraph
            Cached or freshly materialized call graph.
        """
        graph = self._cache.get(
            GraphKind.CALL_GRAPH,
            lambda: views.load_call_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
        )
        return cast("nx.DiGraph", graph)

    def load_call_graph(self) -> nx.DiGraph:
        """
        Alias for call_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.DiGraph
            Directed call graph.
        """
        return self.call_graph()

    def import_graph(self) -> nx.DiGraph:
        """
        Return the import graph for the configured snapshot.

        Returns
        -------
        nx.DiGraph
            Cached or freshly materialized import graph.
        """
        graph = self._cache.get(
            GraphKind.IMPORT_GRAPH,
            lambda: views.load_import_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
        )
        return cast("nx.DiGraph", graph)

    def load_import_graph(self) -> nx.DiGraph:
        """
        Alias for import_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.DiGraph
            Directed import graph.
        """
        return self.import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        """
        Return the symbol coupling graph aggregated at module granularity.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized symbol-module graph.
        """
        return self._cache.get(
            GraphKind.SYMBOL_MODULE_GRAPH,
            lambda: views.load_symbol_module_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
        )

    def load_symbol_module_graph(self) -> nx.Graph:
        """
        Alias for symbol_module_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Symbol-module coupling graph.
        """
        return self.symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        """
        Return the symbol coupling graph aggregated at function granularity.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized symbol-function graph.
        """
        return self._cache.get(
            GraphKind.SYMBOL_FUNCTION_GRAPH,
            lambda: views.load_symbol_function_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
        )

    def load_symbol_function_graph(self) -> nx.Graph:
        """
        Alias for symbol_function_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Symbol-function coupling graph.
        """
        return self.symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        """
        Return the config key <-> module bipartite graph.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized config bipartite graph.
        """
        return self._cache.get(
            GraphKind.CONFIG_MODULE_BIPARTITE,
            lambda: views.load_config_module_bipartite(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
        )

    def load_config_module_bipartite(self) -> nx.Graph:
        """
        Alias for config_module_bipartite to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Config-module bipartite graph.
        """
        return self.config_module_bipartite()

    def test_function_bipartite(self) -> nx.Graph:
        """
        Return the test <-> function bipartite graph.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized test/function bipartite graph.
        """
        return self._cache.get(
            GraphKind.TEST_FUNCTION_BIPARTITE,
            lambda: views.load_test_function_bipartite(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
        )

    def load_test_function_bipartite(self) -> nx.Graph:
        """Alias for test_function_bipartite to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Test-function bipartite graph.
        """
        return self.test_function_bipartite()

    def clear_cache(self) -> None:
        """Clear all cached graphs.

        Forces graphs to be reloaded on next access.
        """
        self._cache.clear()


__all__ = ["NxGraphEngine"]
