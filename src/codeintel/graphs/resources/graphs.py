"""Graph engine resource provider.

This module provides a resource provider for graph engine access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import networkx as nx

from codeintel.graphs.ports.engine import GraphData

if TYPE_CHECKING:
    from codeintel.graphs.engine import NxGraphEngine


@dataclass
class GraphResource:
    """Resource provider for graph engine access.

    Implements both ResourceProvider and EnginePort protocols,
    providing unified access to graph operations.

    Attributes
    ----------
    engine
        Underlying graph engine.
    """

    RESOURCE_NAME: ClassVar[str] = "graphs"

    engine: NxGraphEngine

    @property
    def resource_name(self) -> str:
        """Resource identifier.

        Returns
        -------
        str
            Resource name.
        """
        return self.RESOURCE_NAME

    def get(self) -> GraphResource:
        """Get graph resource.

        Returns
        -------
        GraphResource
            Self, providing access to engine and port methods.
        """
        return self

    def invalidate(self) -> None:
        """Invalidate cached graphs.

        Clears the engine's internal cache.
        """
        self.engine.clear_cache()

    @property
    def repo(self) -> str:
        """Repository identifier.

        Returns
        -------
        str
            Repository slug.
        """
        return self.engine.repo

    @property
    def commit(self) -> str:
        """Commit identifier.

        Returns
        -------
        str
            Commit hash.
        """
        return self.engine.commit

    def call_graph(self) -> nx.DiGraph:
        """Get the call graph.

        Returns
        -------
        nx.DiGraph
            Directed call graph.
        """
        return self.engine.call_graph()

    def import_graph(self) -> nx.DiGraph:
        """Get the import graph.

        Returns
        -------
        nx.DiGraph
            Directed import graph.
        """
        return self.engine.import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        """Get the symbol-module coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-module graph.
        """
        return self.engine.symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        """Get the symbol-function coupling graph.

        Returns
        -------
        nx.Graph
            Undirected symbol-function graph.
        """
        return self.engine.symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        """Get the config-module bipartite graph.

        Returns
        -------
        nx.Graph
            Bipartite graph linking configs to modules.
        """
        return self.engine.config_module_bipartite()

    def test_function_bipartite(self) -> nx.Graph:
        """Get the test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Bipartite graph linking tests to functions.
        """
        return self.engine.test_function_bipartite()

    def call_graph_data(self) -> GraphData:
        """Get call graph as lightweight data object.

        Returns
        -------
        GraphData
            Call graph data without NetworkX dependency.
        """
        return GraphData.from_nx(self.call_graph())

    def import_graph_data(self) -> GraphData:
        """Get import graph as lightweight data object.

        Returns
        -------
        GraphData
            Import graph data without NetworkX dependency.
        """
        return GraphData.from_nx(self.import_graph())


__all__ = ["GraphResource"]
