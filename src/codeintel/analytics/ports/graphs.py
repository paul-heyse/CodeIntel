"""Graph runtime port interface for analytics.

This module defines the GraphRuntimePort protocol that abstracts access
to graph engines and cached graph instances used by analytics plugins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.graphs.engine import GraphKind


@runtime_checkable
class GraphRuntimePort(Protocol):
    """Protocol for graph runtime access in analytics.

    Implementations provide access to graph engines and cached graph
    instances without exposing storage or construction details.
    """

    def get_graph(self, kind: GraphKind) -> nx.DiGraph | nx.Graph:
        """Retrieve a graph by kind.

        Parameters
        ----------
        kind
            The type of graph to retrieve (e.g., "call_graph", "import_graph").

        Returns
        -------
        nx.DiGraph | nx.Graph
            The requested graph instance.

        Raises
        ------
        KeyError
            If the graph kind is not available.
        """
        ...

    def has_graph(self, kind: GraphKind) -> bool:
        """Check if a graph kind is available.

        Parameters
        ----------
        kind
            The type of graph to check.

        Returns
        -------
        bool
            True if the graph is available, False otherwise.
        """
        ...

    @property
    def available_graphs(self) -> frozenset[GraphKind]:
        """Available graph kinds in this runtime.

        Returns
        -------
        frozenset[GraphKind]
            Set of available graph kinds.
        """
        ...


__all__ = [
    "GraphRuntimePort",
]
