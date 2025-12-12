"""Engine port interface for graph engine access.

This module defines the EnginePort protocol that abstracts graph engine
operations, providing access to cached graphs without exposing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    import networkx as nx


@dataclass(frozen=True)
class GraphData:
    """Lightweight graph data transfer object.

    Attributes
    ----------
    nodes
        Sequence of node identifiers.
    edges
        Sequence of (source, target) edge tuples.
    node_attrs
        Mapping of node ID to attribute dict.
    edge_attrs
        Mapping of (source, target) to attribute dict.
    """

    nodes: tuple[int | str, ...]
    edges: tuple[tuple[int | str, int | str], ...]
    node_attrs: Mapping[int | str, Mapping[str, object]] | None = None
    edge_attrs: Mapping[tuple[int | str, int | str], Mapping[str, object]] | None = None

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph.

        Returns
        -------
        int
            Node count.
        """
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph.

        Returns
        -------
        int
            Edge count.
        """
        return len(self.edges)

    @classmethod
    def empty(cls) -> GraphData:
        """Create an empty graph data object.

        Returns
        -------
        GraphData
            Empty graph with no nodes or edges.
        """
        return cls(nodes=(), edges=())

    @classmethod
    def from_nx(cls, graph: nx.Graph | nx.DiGraph) -> GraphData:
        """Create GraphData from a NetworkX graph.

        Parameters
        ----------
        graph
            NetworkX graph to convert.

        Returns
        -------
        GraphData
            Graph data extracted from NetworkX.
        """
        nodes = tuple(graph.nodes())

        edges_list = cast("list[tuple[Any, Any]]", list(graph.edges()))
        edges = tuple(edges_list)
        node_attrs = {node: dict(graph.nodes[node]) for node in graph.nodes()}
        edge_attrs_dict: dict[tuple[int | str, int | str], Mapping[str, object]] = {}
        for src, dst in edges_list:
            edge_attrs_dict[src, dst] = dict(graph.edges[src, dst])
        return cls(
            nodes=nodes,
            edges=edges,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs_dict,
        )


@runtime_checkable
class EnginePort(Protocol):
    """Protocol for graph engine operations.

    Implementations provide access to cached or freshly loaded graphs
    without exposing the underlying graph library.
    """

    @property
    def repo(self) -> str:
        """Repository identifier.

        Returns
        -------
        str
            Repository slug.
        """
        ...

    @property
    def commit(self) -> str:
        """Commit identifier.

        Returns
        -------
        str
            Commit hash.
        """
        ...

    def call_graph(self) -> nx.DiGraph:
        """Get the call graph.

        Returns
        -------
        nx.DiGraph
            Directed call graph.
        """
        ...

    def import_graph(self) -> nx.DiGraph:
        """Get the import graph.

        Returns
        -------
        nx.DiGraph
            Directed import graph.
        """
        ...

    def symbol_module_graph(self) -> nx.Graph:
        """Get the symbol-module coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-module graph.
        """
        ...

    def symbol_function_graph(self) -> nx.Graph:
        """Get the symbol-function coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-function graph.
        """
        ...

    def config_module_bipartite(self) -> nx.Graph:
        """Get the config-module bipartite graph.

        Returns
        -------
        nx.Graph
            Bipartite graph linking configs to modules.
        """
        ...

    def test_function_bipartite(self) -> nx.Graph:
        """Get the test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Bipartite graph linking tests to functions.
        """
        ...

    def call_graph_data(self) -> GraphData:
        """Get call graph as lightweight data object.

        Returns
        -------
        GraphData
            Call graph data without NetworkX dependency.
        """
        ...

    def import_graph_data(self) -> GraphData:
        """Get import graph as lightweight data object.

        Returns
        -------
        GraphData
            Import graph data without NetworkX dependency.
        """
        ...

    def clear_cache(self) -> None:
        """Clear all cached graphs.

        Forces graphs to be reloaded on next access.
        """
        ...


__all__ = [
    "EnginePort",
    "GraphData",
]
