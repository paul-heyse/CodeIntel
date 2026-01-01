"""Graph engine protocol and graph kind enumeration.

This module defines the backend-agnostic interface for graph engines
and the enumeration of supported graph types.
"""

from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.config.primitives import SnapshotRef


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

    @property
    def use_gpu(self) -> bool:
        """Preferred backend GPU flag."""
        ...

    def call_graph(self) -> nx.DiGraph:
        """
        Return the directed call graph (cached or freshly loaded).

        Returns
        -------
        nx.DiGraph
            Directed call graph.
        """
        ...

    def load_call_graph(self) -> nx.DiGraph:
        """Return the directed call graph."""
        ...

    def import_graph(self) -> nx.DiGraph:
        """
        Return the directed import graph (cached or freshly loaded).

        Returns
        -------
        nx.DiGraph
            Directed import graph.
        """
        ...

    def load_import_graph(self) -> nx.DiGraph:
        """Return the directed import graph."""
        ...

    def symbol_module_graph(self) -> nx.Graph:
        """
        Return the undirected symbol-module coupling graph (cached or loaded).

        Returns
        -------
        nx.Graph
            Symbol-module coupling graph.
        """
        ...

    def load_symbol_module_graph(self) -> nx.Graph:
        """Return the undirected symbol-module coupling graph."""
        ...

    def symbol_function_graph(self) -> nx.Graph:
        """
        Return the undirected symbol-function coupling graph (cached or loaded).

        Returns
        -------
        nx.Graph
            Symbol-function coupling graph.
        """
        ...

    def load_symbol_function_graph(self) -> nx.Graph:
        """Return the undirected symbol-function coupling graph."""
        ...

    def config_module_bipartite(self) -> nx.Graph:
        """
        Return the config key <-> module bipartite graph (cached or loaded).

        Returns
        -------
        nx.Graph
            Config-module bipartite graph.
        """
        ...

    def load_config_module_bipartite(self) -> nx.Graph:
        """Return the config key <-> module bipartite graph."""
        ...

    def test_function_bipartite(self) -> nx.Graph:
        """
        Return the test <-> function bipartite graph (cached or loaded).

        Returns
        -------
        nx.Graph
            Test-function bipartite graph.
        """
        ...

    def load_test_function_bipartite(self) -> nx.Graph:
        """Return the test <-> function bipartite graph."""
        ...

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference this engine is bound to."""
        ...


__all__ = [
    "GraphEngine",
    "GraphKind",
]
