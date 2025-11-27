"""Graph engine abstractions for analytics consumers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Protocol, cast

import networkx as nx

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs import nx_views
from codeintel.storage.gateway import StorageGateway


class GraphKind(Flag):
    """Enumerated set of graphs surfaced through the engine."""

    NONE = 0
    CALL_GRAPH = auto()
    IMPORT_GRAPH = auto()
    CFG_GRAPH = auto()
    SYMBOL_MODULE_GRAPH = auto()
    SYMBOL_FUNCTION_GRAPH = auto()
    CONFIG_MODULE_BIPARTITE = auto()
    TEST_FUNCTION_BIPARTITE = auto()
    SYMBOL = SYMBOL_MODULE_GRAPH | SYMBOL_FUNCTION_GRAPH
    ALL = (
        CALL_GRAPH
        | IMPORT_GRAPH
        | CFG_GRAPH
        | SYMBOL
        | CONFIG_MODULE_BIPARTITE
        | TEST_FUNCTION_BIPARTITE
    )


class GraphEngine(Protocol):
    """
    Backend-agnostic interface for building and caching analytics graphs.

    Implementations may cache results and route to CPU or GPU backends without
    exposing those details to analytics consumers.
    """

    def load_call_graph(self) -> nx.DiGraph:
        """Return the directed call graph."""
        ...

    def load_import_graph(self) -> nx.DiGraph:
        """Return the directed import graph."""
        ...

    def load_symbol_module_graph(self) -> nx.Graph:
        """Return the undirected symbol-module coupling graph."""
        ...

    def load_symbol_function_graph(self) -> nx.Graph:
        """Return the undirected symbol-function coupling graph."""
        ...

    def load_config_module_bipartite(self) -> nx.Graph:
        """Return the config key <-> module bipartite graph."""
        ...

    def load_test_function_bipartite(self) -> nx.Graph:
        """Return the test <-> function bipartite graph."""
        ...

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference this engine is bound to."""
        ...


@dataclass
class NxGraphEngine:
    """NetworkX-backed GraphEngine powered by DuckDB views."""

    gateway: StorageGateway
    snapshot: SnapshotRef
    use_gpu: bool = False
    _cache: dict[GraphKind, nx.Graph] = field(default_factory=dict)

    def seed(self, kind: GraphKind, graph: nx.Graph | None) -> None:
        """Pre-populate the cache when a graph is already available."""
        if graph is None:
            return
        self._cache[kind] = graph

    def _get(self, kind: GraphKind, loader: Callable[[], nx.Graph]) -> nx.Graph:
        graph = self._cache.get(kind)
        if graph is None:
            graph = loader()
            self._cache[kind] = graph
        return graph

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
        graph = self._get(
            GraphKind.CALL_GRAPH,
            lambda: nx_views.load_call_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )
        return cast("nx.DiGraph", graph)

    def load_call_graph(self) -> nx.DiGraph:
        """Alias for call_graph to satisfy GraphEngine protocol."""
        return self.call_graph()

    def import_graph(self) -> nx.DiGraph:
        """
        Return the import graph for the configured snapshot.

        Returns
        -------
        nx.DiGraph
            Cached or freshly materialized import graph.
        """
        graph = self._get(
            GraphKind.IMPORT_GRAPH,
            lambda: nx_views.load_import_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )
        return cast("nx.DiGraph", graph)

    def load_import_graph(self) -> nx.DiGraph:
        """Alias for import_graph to satisfy GraphEngine protocol."""
        return self.import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        """
        Return the symbol coupling graph aggregated at module granularity.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized symbol-module graph.
        """
        return self._get(
            GraphKind.SYMBOL_MODULE_GRAPH,
            lambda: nx_views.load_symbol_module_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    def load_symbol_module_graph(self) -> nx.Graph:
        """Alias for symbol_module_graph to satisfy GraphEngine protocol."""
        return self.symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        """
        Return the symbol coupling graph aggregated at function granularity.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized symbol-function graph.
        """
        return self._get(
            GraphKind.SYMBOL_FUNCTION_GRAPH,
            lambda: nx_views.load_symbol_function_graph(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    def load_symbol_function_graph(self) -> nx.Graph:
        """Alias for symbol_function_graph to satisfy GraphEngine protocol."""
        return self.symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        """
        Return the config key <-> module bipartite graph.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized config bipartite graph.
        """
        return self._get(
            GraphKind.CONFIG_MODULE_BIPARTITE,
            lambda: nx_views.load_config_module_bipartite(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    def load_config_module_bipartite(self) -> nx.Graph:
        """Alias for config_module_bipartite to satisfy GraphEngine protocol."""
        return self.config_module_bipartite()

    def test_function_bipartite(self) -> nx.Graph:
        """
        Return the test <-> function bipartite graph.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized test/function bipartite graph.
        """
        return self._get(
            GraphKind.TEST_FUNCTION_BIPARTITE,
            lambda: nx_views.load_test_function_bipartite(
                self.gateway,
                self.repo,
                self.commit,
                use_gpu=self.use_gpu,
            ),
        )

    def load_test_function_bipartite(self) -> nx.Graph:
        """Alias for test_function_bipartite to satisfy GraphEngine protocol."""
        return self.test_function_bipartite()
