"""Compatibility shim for rustworkx engine selection.

This wrapper delegates to the NetworkX engine while rustworkx migration
work is in progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.build.graphs.engine.backend import BackendEnablement
    from codeintel.build.graphs.engine.nx_engine import NxGraphEngine
    from codeintel.build.graphs.engine.protocol import GraphKind
    from codeintel.config.primitives import SnapshotRef


@dataclass(slots=True)
class RxGraphEngine:
    """Compatibility wrapper for rustworkx engine selection."""

    delegate: NxGraphEngine

    @property
    def backend_info(self) -> BackendEnablement | None:
        """Backend enablement info captured during delegate construction.

        Returns
        -------
        BackendEnablement | None
            Backend enablement details captured by the delegate.
        """
        return self.delegate.backend_info

    @property
    def use_gpu(self) -> bool:
        """Rustworkx is CPU-only, so GPU is never enabled.

        Returns
        -------
        bool
            False, indicating GPU acceleration is disabled.
        """
        return False

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference delegated from the NetworkX engine.

        Returns
        -------
        SnapshotRef
            Snapshot reference used for graph construction.
        """
        return self.delegate.snapshot

    def seed(self, kind: GraphKind, graph: nx.Graph | None) -> None:
        """Pre-populate the delegate cache when a graph is already available."""
        self.delegate.seed(kind, graph)

    def call_graph(self) -> nx.DiGraph:
        """Return the directed call graph.

        Returns
        -------
        nx.DiGraph
            Directed call graph for the snapshot.
        """
        return self.delegate.call_graph()

    def load_call_graph(self) -> nx.DiGraph:
        """Return the directed call graph.

        Returns
        -------
        nx.DiGraph
            Directed call graph for the snapshot.
        """
        return self.delegate.load_call_graph()

    def import_graph(self) -> nx.DiGraph:
        """Return the directed import graph.

        Returns
        -------
        nx.DiGraph
            Directed import graph for the snapshot.
        """
        return self.delegate.import_graph()

    def load_import_graph(self) -> nx.DiGraph:
        """Return the directed import graph.

        Returns
        -------
        nx.DiGraph
            Directed import graph for the snapshot.
        """
        return self.delegate.load_import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        """Return the undirected symbol-module coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-module graph for the snapshot.
        """
        return self.delegate.symbol_module_graph()

    def load_symbol_module_graph(self) -> nx.Graph:
        """Return the undirected symbol-module coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-module graph for the snapshot.
        """
        return self.delegate.load_symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        """Return the undirected symbol-function coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-function graph for the snapshot.
        """
        return self.delegate.symbol_function_graph()

    def load_symbol_function_graph(self) -> nx.Graph:
        """Return the undirected symbol-function coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-function graph for the snapshot.
        """
        return self.delegate.load_symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        """Return the config key <-> module bipartite graph.

        Returns
        -------
        nx.Graph
            Bipartite graph connecting config keys to modules.
        """
        return self.delegate.config_module_bipartite()

    def load_config_module_bipartite(self) -> nx.Graph:
        """Return the config key <-> module bipartite graph.

        Returns
        -------
        nx.Graph
            Bipartite graph connecting config keys to modules.
        """
        return self.delegate.load_config_module_bipartite()
