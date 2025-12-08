"""Deprecated shim - use GraphRuntimeDouble from tests._helpers.fakes.graph_runtime."""

from __future__ import annotations

import networkx as nx

from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as MockGraphRuntime,
)
from tests._helpers.graphs import GraphFixtures, standard_graph_fixtures


def create_mock_runtime_with_call_graph(
    edges: list[tuple[str, str]] | None = None,
) -> MockGraphRuntime:
    """Create a MockGraphRuntime with a populated call graph.

    Returns
    -------
    MockGraphRuntime
        Runtime seeded with a call graph.
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

    Returns
    -------
    MockGraphRuntime
        Runtime seeded with an import graph.
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
        Runtime seeded with all graph types.
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
        config_graph=config_mod_g,
        test_function_graph=test_func_g,
        cfg_graph=cfg_g,
    )


def create_mock_runtime_with_standard_graphs(
    fixtures: GraphFixtures | None = None,
) -> MockGraphRuntime:
    """Create a MockGraphRuntime seeded with standard graph shapes.

    Returns
    -------
    MockGraphRuntime
        Runtime seeded with standard fixtures.
    """
    graphs = fixtures or standard_graph_fixtures()
    return MockGraphRuntime.from_fixtures(graphs)


__all__ = [
    "MockGraphRuntime",
    "create_mock_runtime_all_graphs",
    "create_mock_runtime_with_call_graph",
    "create_mock_runtime_with_import_graph",
    "create_mock_runtime_with_standard_graphs",
]
