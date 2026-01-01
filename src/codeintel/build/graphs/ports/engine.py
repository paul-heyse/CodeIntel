"""Engine data types for graph operations.

This module defines data classes for graph engine operations.

Data Classes
------------
- GraphData: Lightweight graph data transfer object

See Also
--------
codeintel.build.graphs.runtime : GraphRuntime for graph access
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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

    nodes: tuple[Hashable, ...]
    edges: tuple[tuple[Hashable, Hashable], ...]
    node_attrs: Mapping[Hashable, Mapping[str, object]] | None = None
    edge_attrs: Mapping[tuple[Hashable, Hashable], Mapping[str, object]] | None = None

    @property
    def node_count(self) -> int:
        """Return number of nodes in the graph.

        Returns
        -------
        int
            Node count.
        """
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return number of edges in the graph.

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
        edge_attrs_dict: dict[tuple[Hashable, Hashable], Mapping[str, object]] = {}
        for src, dst in edges_list:
            edge_attrs_dict[src, dst] = dict(graph.edges[src, dst])
        return cls(
            nodes=nodes,
            edges=edges,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs_dict,
        )


__all__ = [
    "GraphData",
]
