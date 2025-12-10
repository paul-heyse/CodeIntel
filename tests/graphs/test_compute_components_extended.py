"""Extended tests for graph components computation module.

This module provides additional test coverage for the components module
from `codeintel.graphs.compute.metrics.components`, including:

- Strongly connected component detection
- Condensation graph computation
- Topological layer computation
- Component size and membership tracking
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.components import (
    ComponentInfo,
    SCCResult,
    condensation_layers,
    find_strongly_connected,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    complex_sccs_graph,
    cyclic_graph,
    diamond_graph,
    two_sccs_graph,
)

SIMPLE_DAG_NODE_COUNT: Final = 4
SINGLE_CYCLE_SIZE: Final = 3
TWO_SCC_COUNT: Final = 2
COMPLEX_SCC_COUNT: Final = 3
CONDENSATION_NODE_COUNT: Final = 2
SIMPLE_DAG_LAST_LAYER: Final = 3
DIAMOND_SHARED_LAYER: Final = 1
DIAMOND_LAST_LAYER: Final = 2
SINGLE_NODE_LAYER: Final = 0
ROOT_COMPONENT_ID: Final = 0
TRIPLE_NODE_SIZE: Final = 3
DOUBLE_NODE_SIZE: Final = 2
SINGLE_NODE_SIZE: Final = 1
SINGLE_COMPONENT_COUNT: Final = 1


def test_find_strongly_connected_dag() -> None:
    """Find SCCs in a DAG (each node is its own SCC)."""
    g = chain_graph(4)

    result = find_strongly_connected(g)

    expect_length(result.components, SIMPLE_DAG_NODE_COUNT)
    # Each node is its own SCC
    for comp in result.components:
        expect_equal(comp.size, SINGLE_NODE_SIZE)


def test_find_strongly_connected_single_cycle() -> None:
    """Find SCCs in a single cycle graph."""
    g = cyclic_graph(3)

    result = find_strongly_connected(g)

    expect_length(result.components, SINGLE_COMPONENT_COUNT)
    comp = result.components[0]
    expect_equal(comp.size, SINGLE_CYCLE_SIZE)
    expect_equal(set(comp.nodes), {"A", "B", "C"})


def test_find_strongly_connected_two_sccs() -> None:
    """Find SCCs in a graph with two SCCs."""
    g = two_sccs_graph()

    result = find_strongly_connected(g)

    expect_length(result.components, TWO_SCC_COUNT)

    # Find sizes
    sizes = sorted(c.size for c in result.components)
    expect_equal(sizes, [DOUBLE_NODE_SIZE, DOUBLE_NODE_SIZE])


def test_find_strongly_connected_complex() -> None:
    """Find SCCs in a complex graph."""
    g = complex_sccs_graph()

    result = find_strongly_connected(g)

    # Should have 3 SCCs: A (1), B-C-D (3), E-F (2)
    expect_length(result.components, COMPLEX_SCC_COUNT)

    sizes = sorted(c.size for c in result.components)
    expect_equal(sizes, [SINGLE_NODE_SIZE, DOUBLE_NODE_SIZE, TRIPLE_NODE_SIZE])


def test_find_strongly_connected_empty_graph() -> None:
    """Find SCCs in an empty graph."""
    g = nx.DiGraph()

    result = find_strongly_connected(g)

    expect_length(result.components, 0)
    expect_equal(result.node_to_component, {})


def test_find_strongly_connected_single_node() -> None:
    """Find SCCs in a single node graph."""
    g = nx.DiGraph()
    g.add_node("A")

    result = find_strongly_connected(g)

    expect_length(result.components, SINGLE_COMPONENT_COUNT)
    expect_equal(result.components[0].size, SINGLE_NODE_SIZE)
    expect_true("A" in result.components[0].nodes)


def test_find_strongly_connected_node_to_component_mapping() -> None:
    """Node to component mapping is correct."""
    g = cyclic_graph(3)

    result = find_strongly_connected(g)

    # All nodes should map to component 0 (the only SCC)
    expect_true("A" in result.node_to_component)
    expect_true("B" in result.node_to_component)
    expect_true("C" in result.node_to_component)
    # All should be in same component
    expect_equal(result.node_to_component["A"], result.node_to_component["B"])
    expect_equal(result.node_to_component["B"], result.node_to_component["C"])


def test_find_strongly_connected_with_condensation() -> None:
    """Find SCCs with condensation graph computation."""
    g = two_sccs_graph()

    result = find_strongly_connected(g, compute_condensation=True)

    expect_true(result.condensation is not None)
    if result.condensation is None:
        pytest.fail("Expected condensation graph")
    # Condensation should have 2 nodes (one per SCC)
    expect_equal(result.condensation.number_of_nodes(), CONDENSATION_NODE_COUNT)


def test_find_strongly_connected_condensation_is_dag() -> None:
    """Condensation graph is always a DAG."""
    g = complex_sccs_graph()

    result = find_strongly_connected(g, compute_condensation=True)

    expect_true(result.condensation is not None)
    if result.condensation is None:
        pytest.fail("Expected condensation graph")
    expect_true(nx.is_directed_acyclic_graph(result.condensation))


def test_condensation_layers_dag() -> None:
    """Compute layers on a simple DAG."""
    g = chain_graph(4)
    scc_result = find_strongly_connected(g, compute_condensation=True)

    layers = condensation_layers(g, scc_result)

    # Each node should have a layer
    expect_length(layers, SIMPLE_DAG_NODE_COUNT)
    # A should be earliest (layer 0)
    expect_equal(layers["A"], SINGLE_NODE_LAYER)
    # D should be latest
    expect_equal(layers["D"], SIMPLE_DAG_LAST_LAYER)


def test_condensation_layers_diamond_dag() -> None:
    """Compute layers on a diamond DAG."""
    g = diamond_graph()
    scc_result = find_strongly_connected(g, compute_condensation=True)

    layers = condensation_layers(g, scc_result)

    # A at layer 0
    expect_equal(layers["A"], SINGLE_NODE_LAYER)
    # B and C at same layer
    expect_equal(layers["B"], layers["C"])
    expect_equal(layers["B"], DIAMOND_SHARED_LAYER)
    # D at layer 2
    expect_equal(layers["D"], DIAMOND_LAST_LAYER)


def test_condensation_layers_two_sccs() -> None:
    """Compute layers with multiple SCCs."""
    g = two_sccs_graph()
    scc_result = find_strongly_connected(g, compute_condensation=True)

    layers = condensation_layers(g, scc_result)

    # All nodes in first SCC have same layer
    expect_equal(layers["A"], layers["B"])
    # All nodes in second SCC have same layer
    expect_equal(layers["C"], layers["D"])
    # Second SCC is after first due to B -> C edge
    expect_true(layers["C"] > layers["A"])


def test_condensation_layers_empty_graph() -> None:
    """Compute layers on an empty graph."""
    g = nx.DiGraph()
    scc_result = find_strongly_connected(g, compute_condensation=True)

    layers = condensation_layers(g, scc_result)

    expect_equal(layers, {})


def test_condensation_layers_single_node() -> None:
    """Compute layers on a single node graph."""
    g = nx.DiGraph()
    g.add_node("A")
    scc_result = find_strongly_connected(g, compute_condensation=True)

    layers = condensation_layers(g, scc_result)

    expect_equal(layers["A"], SINGLE_NODE_LAYER)


def test_condensation_layers_cycle() -> None:
    """Compute layers on a cycle (single SCC)."""
    g = cyclic_graph(3)
    scc_result = find_strongly_connected(g, compute_condensation=True)

    layers = condensation_layers(g, scc_result)

    # All nodes in cycle should be at same layer
    expect_true(layers["A"] == layers["B"] == layers["C"])


def test_component_info_attributes() -> None:
    """ComponentInfo has correct attributes."""
    comp = ComponentInfo(
        component_id=ROOT_COMPONENT_ID,
        nodes=frozenset(["A", "B", "C"]),
        size=TRIPLE_NODE_SIZE,
    )

    expect_equal(comp.component_id, ROOT_COMPONENT_ID)
    expect_equal(comp.nodes, frozenset(["A", "B", "C"]))
    expect_equal(comp.size, TRIPLE_NODE_SIZE)


def test_scc_result_attributes() -> None:
    """SCCResult has correct attributes."""
    comp = ComponentInfo(
        component_id=ROOT_COMPONENT_ID,
        nodes=frozenset(["A"]),
        size=SINGLE_NODE_SIZE,
    )

    result = SCCResult(
        components=(comp,),
        node_to_component={"A": 0},
        condensation=None,
    )

    expect_length(result.components, SINGLE_COMPONENT_COUNT)
    expect_equal(result.node_to_component["A"], ROOT_COMPONENT_ID)
    expect_true(result.condensation is None)


def test_scc_result_with_condensation() -> None:
    """SCCResult can include condensation graph."""
    g = two_sccs_graph()
    result = find_strongly_connected(g, compute_condensation=True)

    expect_true(result.condensation is not None)
    expect_true(isinstance(result.condensation, nx.DiGraph))
