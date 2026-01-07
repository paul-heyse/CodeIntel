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
from typing import TYPE_CHECKING, Any

from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload

if TYPE_CHECKING:
    from collections.abc import Mapping


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
    def from_store(cls, graph: GraphInput) -> GraphData:
        """Create GraphData from a rustworkx-backed graph input.

        Parameters
        ----------
        graph
            Rustworkx graph input to convert.

        Returns
        -------
        GraphData
            Graph data extracted from rustworkx.
        """
        store = ensure_store(graph)
        nodes = tuple(store.node_ids())
        edges_list: list[tuple[Any, Any]] = [
            (store.index_to_id[src_idx], store.index_to_id[dst_idx])
            for src_idx, dst_idx in store.graph.edge_list()
        ]
        edges = tuple(edges_list)
        node_attrs = {node: store.get_node_attrs(node) for node in store.node_ids()}
        edge_attrs_dict: dict[tuple[Hashable, Hashable], Mapping[str, object]] = {}
        for src_id, dst_id in edges_list:
            src_idx = store.id_to_index[src_id]
            dst_idx = store.id_to_index[dst_id]
            payload = store.graph.get_edge_data(src_idx, dst_idx)
            edge_attrs_dict[src_id, dst_id] = {"weight": edge_weight_from_payload(payload)}
        return cls(
            nodes=nodes,
            edges=edges,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs_dict,
        )


__all__ = [
    "GraphData",
]
