"""Mock graph runtime implementations for testing.

This module provides mock implementations of GraphRuntime that satisfy
the GraphRuntimeLike protocol, enabling tests without real database
connections or complex setup.

The mocks follow the Testing Charter:
- They implement the same interface as production code
- They preserve key invariants (graph types, attribute availability)
- They can be used in dev/staging environments

Example
-------
>>> from tests._helpers.fakes.graph_runtimes import MockGraphRuntime
>>> import networkx as nx
>>>
>>> # Create a mock with custom graphs
>>> runtime = MockGraphRuntime(
...     call_graph=nx.DiGraph(),
...     import_graph=nx.DiGraph(),
... )
>>>
>>> # Use with GraphProvider
>>> from codeintel.analytics.resources.graphs import GraphProvider
>>> provider = GraphProvider.from_runtime(runtime)
>>> resources = provider.get()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from tests._helpers.graphs import GraphFixtures, standard_graph_fixtures

if TYPE_CHECKING:
    from codeintel.config.primitives import GraphBackendConfig


@dataclass
class MockGraphRuntime:
    """Mock GraphRuntime for testing graph resource providers.

    Provides configurable graph responses for testing lazy loading and
    resource access patterns. Implements the GraphRuntimeLike protocol
    so it can be used with GraphProvider.from_runtime() without type
    suppressions.

    Attributes
    ----------
    call_graph
        Optional call graph (directed).
    import_graph
        Optional import graph (directed).
    symbol_module_graph
        Optional symbol-to-module bipartite graph.
    symbol_function_graph
        Optional symbol-to-function bipartite graph.
    config_module_bipartite
        Optional config-to-module bipartite graph.
    test_function_bipartite
        Optional test-to-function bipartite graph.
    cfg_graph
        Optional control flow graph (directed).
    backend
        Optional backend configuration.
    use_gpu
        Whether GPU execution is enabled.

    Examples
    --------
    Create a mock with a simple call graph:

    >>> g = nx.DiGraph()
    >>> g.add_edge("func_a", "func_b")
    >>> runtime = MockGraphRuntime(call_graph=g)
    >>> runtime.call_graph.number_of_edges()
    1

    Create a mock with multiple graphs:

    >>> call_g = nx.DiGraph([("a", "b")])
    >>> import_g = nx.DiGraph([("mod1", "mod2")])
    >>> runtime = MockGraphRuntime(call_graph=call_g, import_graph=import_g)
    """

    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None
    symbol_module_graph: nx.Graph | None = None
    symbol_function_graph: nx.Graph | None = None
    config_module_bipartite: nx.Graph | None = None
    test_function_bipartite: nx.Graph | None = None
    cfg_graph: nx.DiGraph | None = None
    backend: GraphBackendConfig | None = None
    use_gpu: bool = False

    def ensure_call_graph(self) -> nx.DiGraph | None:
        """Return call graph (used by GraphProvider's _ensure_graph).

        Returns
        -------
        nx.DiGraph | None
            The call graph or None.
        """
        return self.call_graph

    def ensure_import_graph(self) -> nx.DiGraph | None:
        """Return import graph (used by GraphProvider's _ensure_graph).

        Returns
        -------
        nx.DiGraph | None
            The import graph or None.
        """
        return self.import_graph

    def ensure_symbol_module_graph(self) -> nx.Graph | None:
        """Return symbol-module graph.

        Returns
        -------
        nx.Graph | None
            The symbol-module graph or None.
        """
        return self.symbol_module_graph

    def ensure_symbol_function_graph(self) -> nx.Graph | None:
        """Return symbol-function graph.

        Returns
        -------
        nx.Graph | None
            The symbol-function graph or None.
        """
        return self.symbol_function_graph

    def ensure_config_module_bipartite(self) -> nx.Graph | None:
        """Return config-module bipartite graph.

        Returns
        -------
        nx.Graph | None
            The config-module bipartite graph or None.
        """
        return self.config_module_bipartite

    def ensure_test_function_bipartite(self) -> nx.Graph | None:
        """Return test-function bipartite graph.

        Returns
        -------
        nx.Graph | None
            The test-function bipartite graph or None.
        """
        return self.test_function_bipartite

    def ensure_cfg_graph(self) -> nx.DiGraph | None:
        """Return control flow graph.

        Returns
        -------
        nx.DiGraph | None
            The control flow graph or None.
        """
        return self.cfg_graph


def create_mock_runtime_with_call_graph(
    edges: list[tuple[str, str]] | None = None,
) -> MockGraphRuntime:
    """Create a MockGraphRuntime with a populated call graph.

    Parameters
    ----------
    edges
        List of (caller, callee) tuples. Defaults to a simple chain.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with call graph.

    Examples
    --------
    >>> runtime = create_mock_runtime_with_call_graph([("a", "b"), ("b", "c")])
    >>> runtime.call_graph.number_of_edges()
    2
    """
    if edges is None:
        edges = [("func_a", "func_b"), ("func_b", "func_c")]

    call_g = nx.DiGraph()
    call_g.add_edges_from(edges)
    return MockGraphRuntime(call_graph=call_g)


def create_mock_runtime_with_import_graph(
    edges: list[tuple[str, str]] | None = None,
) -> MockGraphRuntime:
    """Create a MockGraphRuntime with a populated import graph.

    Parameters
    ----------
    edges
        List of (importer, imported) tuples. Defaults to a simple chain.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with import graph.

    Examples
    --------
    >>> runtime = create_mock_runtime_with_import_graph([("mod1", "mod2")])
    >>> runtime.import_graph.number_of_edges()
    1
    """
    if edges is None:
        edges = [("mod_a", "mod_b"), ("mod_b", "mod_c")]

    import_g = nx.DiGraph()
    import_g.add_edges_from(edges)
    return MockGraphRuntime(import_graph=import_g)


def create_mock_runtime_all_graphs() -> MockGraphRuntime:
    """Create a MockGraphRuntime with all graph types populated.

    Returns
    -------
    MockGraphRuntime
        Mock runtime with all graphs.

    Examples
    --------
    >>> runtime = create_mock_runtime_all_graphs()
    >>> runtime.call_graph is not None
    True
    >>> runtime.import_graph is not None
    True
    """
    call_g = nx.DiGraph([("f1", "f2"), ("f2", "f3")])
    import_g = nx.DiGraph([("m1", "m2"), ("m2", "m3")])
    symbol_mod_g = nx.Graph([("sym1", "mod1"), ("sym2", "mod2")])
    symbol_func_g = nx.Graph([("sym1", "func1"), ("sym2", "func2")])
    config_mod_g = nx.Graph([("config1", "mod1")])
    test_func_g = nx.Graph([("test1", "func1")])
    cfg_g = nx.DiGraph([("entry", "block1"), ("block1", "exit")])

    return MockGraphRuntime(
        call_graph=call_g,
        import_graph=import_g,
        symbol_module_graph=symbol_mod_g,
        symbol_function_graph=symbol_func_g,
        config_module_bipartite=config_mod_g,
        test_function_bipartite=test_func_g,
        cfg_graph=cfg_g,
    )


def create_mock_runtime_with_standard_graphs(
    fixtures: GraphFixtures | None = None,
) -> MockGraphRuntime:
    """Create a MockGraphRuntime seeded with standard graph shapes.

    Returns
    -------
    MockGraphRuntime
        Runtime with chain call graph, cycle import graph, and star symbol graphs.
    """
    graphs = fixtures or standard_graph_fixtures()
    return MockGraphRuntime(
        call_graph=graphs.call_graph,
        import_graph=graphs.import_graph,
        symbol_module_graph=graphs.symbol_module_graph,
        symbol_function_graph=graphs.symbol_function_graph,
        config_module_bipartite=graphs.config_graph,
        cfg_graph=graphs.cfg_graph,
    )


__all__ = [
    "MockGraphRuntime",
    "create_mock_runtime_all_graphs",
    "create_mock_runtime_with_call_graph",
    "create_mock_runtime_with_import_graph",
    "create_mock_runtime_with_standard_graphs",
]
