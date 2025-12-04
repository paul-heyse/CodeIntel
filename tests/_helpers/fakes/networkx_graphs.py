"""Standard NetworkX graph fixtures for testing.

This module provides pre-built graph structures commonly used across
graph computation tests. These standardized fixtures ensure consistency
and reduce duplication in test files.

Each function returns a fresh graph instance, so modifications in one
test don't affect others.

Example
-------
>>> from tests._helpers.fakes.networkx_graphs import chain_graph, star_graph
>>>
>>> g = chain_graph(4)  # A -> B -> C -> D
>>> g.number_of_nodes()
4
"""

from __future__ import annotations

from typing import Final

import networkx as nx

# Constants for default graph sizes
DEFAULT_CHAIN_LENGTH: Final[int] = 4
DEFAULT_SPOKES: Final[int] = 3
DEFAULT_CYCLE_SIZE: Final[int] = 3
DEFAULT_COMPLETE_SIZE: Final[int] = 5

# Internal constants
_ALPHABET_SIZE: Final[int] = 26
_MIN_CYCLE_SIZE: Final[int] = 2


def chain_graph(length: int = DEFAULT_CHAIN_LENGTH) -> nx.DiGraph:
    """Create a simple chain (path) graph.

    Structure: A -> B -> C -> D (for length=4)

    Parameters
    ----------
    length
        Number of nodes in the chain.

    Returns
    -------
    nx.DiGraph
        A directed chain graph.

    Example
    -------
    >>> g = chain_graph(4)
    >>> list(g.edges())
    [('A', 'B'), ('B', 'C'), ('C', 'D')]
    """
    g = nx.DiGraph()
    if length < 1:
        return g

    # Generate node labels A, B, C, ... (wraps for long chains)
    def node_label(i: int) -> str:
        return chr(ord("A") + i % _ALPHABET_SIZE) if i < _ALPHABET_SIZE else f"N{i}"

    nodes = [node_label(i) for i in range(length)]
    g.add_nodes_from(nodes)

    for i in range(length - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    return g


def star_graph(spokes: int = DEFAULT_SPOKES, *, inward: bool = False) -> nx.DiGraph:
    """Create a star graph with a central hub.

    Outward (default): hub -> spoke1, hub -> spoke2, ...
    Inward: spoke1 -> hub, spoke2 -> hub, ...

    Parameters
    ----------
    spokes
        Number of spoke nodes.
    inward
        If True, edges point from spokes to hub.
        If False, edges point from hub to spokes.

    Returns
    -------
    nx.DiGraph
        A directed star graph.

    Example
    -------
    >>> g = star_graph(3)  # Outward
    >>> list(g.edges())
    [('hub', 'spoke1'), ('hub', 'spoke2'), ('hub', 'spoke3')]

    >>> g = star_graph(3, inward=True)  # Inward
    >>> list(g.edges())
    [('spoke1', 'hub'), ('spoke2', 'hub'), ('spoke3', 'hub')]
    """
    g = nx.DiGraph()
    hub = "hub"
    spoke_nodes = [f"spoke{i + 1}" for i in range(spokes)]

    g.add_node(hub)
    g.add_nodes_from(spoke_nodes)

    for spoke in spoke_nodes:
        if inward:
            g.add_edge(spoke, hub)
        else:
            g.add_edge(hub, spoke)

    return g


def diamond_graph() -> nx.DiGraph:
    r"""Create a diamond-shaped graph.

    Structure: A -> B, A -> C, B -> D, C -> D

        A
       / \
      B   C
       \ /
        D

    Returns
    -------
    nx.DiGraph
        A diamond graph with 4 nodes.

    Example
    -------
    >>> g = diamond_graph()
    >>> g.number_of_nodes()
    4
    >>> g.number_of_edges()
    4
    """
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return g


def cyclic_graph(size: int = DEFAULT_CYCLE_SIZE) -> nx.DiGraph:
    """Create a simple cycle graph.

    Structure: A -> B -> C -> A (for size=3)

    Parameters
    ----------
    size
        Number of nodes in the cycle.

    Returns
    -------
    nx.DiGraph
        A directed cycle graph.

    Example
    -------
    >>> g = cyclic_graph(3)
    >>> list(g.edges())
    [('A', 'B'), ('B', 'C'), ('C', 'A')]
    """
    g = nx.DiGraph()
    if size < _MIN_CYCLE_SIZE:
        return g

    def node_label(i: int) -> str:
        return chr(ord("A") + i % _ALPHABET_SIZE) if i < _ALPHABET_SIZE else f"N{i}"

    nodes = [node_label(i) for i in range(size)]
    g.add_nodes_from(nodes)

    # Add cycle edges
    for i in range(size):
        g.add_edge(nodes[i], nodes[(i + 1) % size])

    return g


def disconnected_graph() -> nx.DiGraph:
    """Create a graph with disconnected components.

    Structure: Two separate chains:
    - Component 1: A -> B -> C
    - Component 2: X -> Y -> Z

    Returns
    -------
    nx.DiGraph
        A graph with two disconnected components.

    Example
    -------
    >>> g = disconnected_graph()
    >>> g.number_of_nodes()
    6
    >>> len(list(nx.weakly_connected_components(g)))
    2
    """
    g = nx.DiGraph()
    # Component 1
    g.add_edges_from([("A", "B"), ("B", "C")])
    # Component 2
    g.add_edges_from([("X", "Y"), ("Y", "Z")])
    return g


def complete_digraph(n: int = DEFAULT_COMPLETE_SIZE) -> nx.DiGraph:
    """Create a complete directed graph.

    Every node has an edge to every other node.

    Parameters
    ----------
    n
        Number of nodes.

    Returns
    -------
    nx.DiGraph
        A complete directed graph.

    Example
    -------
    >>> g = complete_digraph(3)
    >>> g.number_of_edges()
    6
    """
    return nx.complete_graph(n, create_using=nx.DiGraph())


def bipartite_graph(left: int = 3, right: int = 3) -> nx.DiGraph:
    """Create a complete bipartite directed graph.

    All nodes in the left set have edges to all nodes in the right set.

    Parameters
    ----------
    left
        Number of nodes in the left partition.
    right
        Number of nodes in the right partition.

    Returns
    -------
    nx.DiGraph
        A bipartite directed graph.

    Example
    -------
    >>> g = bipartite_graph(2, 3)
    >>> g.number_of_edges()
    6
    """
    g = nx.DiGraph()
    left_nodes = [f"L{i}" for i in range(left)]
    right_nodes = [f"R{i}" for i in range(right)]

    g.add_nodes_from(left_nodes)
    g.add_nodes_from(right_nodes)

    for lnode in left_nodes:
        for rnode in right_nodes:
            g.add_edge(lnode, rnode)

    return g


def tree_graph(depth: int = 3, branching: int = 2) -> nx.DiGraph:
    """Create a balanced tree graph.

    Parameters
    ----------
    depth
        Tree depth (root is depth 0).
    branching
        Number of children per node.

    Returns
    -------
    nx.DiGraph
        A tree graph with specified depth and branching factor.

    Example
    -------
    >>> g = tree_graph(2, 2)  # Binary tree of depth 2
    >>> g.number_of_nodes()
    7
    """
    g = nx.DiGraph()
    if depth < 0:
        return g

    node_counter = 0

    def add_children(parent: str, current_depth: int) -> None:
        nonlocal node_counter
        if current_depth >= depth:
            return

        for _ in range(branching):
            node_counter += 1
            child = f"N{node_counter}"
            g.add_node(child)
            g.add_edge(parent, child)
            add_children(child, current_depth + 1)

    root = "N0"
    g.add_node(root)
    add_children(root, 0)

    return g


def hub_and_spoke_graph(hubs: int = 2, spokes_per_hub: int = 3) -> nx.DiGraph:
    """Create a graph with multiple hubs connected to their own spokes.

    Parameters
    ----------
    hubs
        Number of hub nodes.
    spokes_per_hub
        Number of spokes connected to each hub.

    Returns
    -------
    nx.DiGraph
        A graph with multiple independent hub-spoke clusters.

    Example
    -------
    >>> g = hub_and_spoke_graph(2, 3)
    >>> g.number_of_nodes()
    8
    """
    g = nx.DiGraph()

    for h in range(hubs):
        hub = f"hub{h}"
        g.add_node(hub)

        for s in range(spokes_per_hub):
            spoke = f"h{h}_spoke{s}"
            g.add_node(spoke)
            g.add_edge(hub, spoke)

    return g


def layered_graph(layers: tuple[int, ...] = (2, 3, 2)) -> nx.DiGraph:
    """Create a layered DAG with edges between adjacent layers.

    Parameters
    ----------
    layers
        Tuple specifying number of nodes per layer.

    Returns
    -------
    nx.DiGraph
        A layered directed acyclic graph.

    Example
    -------
    >>> g = layered_graph((2, 3, 2))
    >>> g.number_of_nodes()
    7
    """
    g = nx.DiGraph()
    prev_layer_nodes: list[str] = []

    for layer_idx, size in enumerate(layers):
        current_nodes = [f"L{layer_idx}N{i}" for i in range(size)]
        g.add_nodes_from(current_nodes)

        # Connect all nodes in previous layer to all nodes in current layer
        for prev in prev_layer_nodes:
            for curr in current_nodes:
                g.add_edge(prev, curr)

        prev_layer_nodes = current_nodes

    return g


__all__ = [
    "DEFAULT_CHAIN_LENGTH",
    "DEFAULT_COMPLETE_SIZE",
    "DEFAULT_CYCLE_SIZE",
    "DEFAULT_SPOKES",
    "bipartite_graph",
    "chain_graph",
    "complete_digraph",
    "cyclic_graph",
    "diamond_graph",
    "disconnected_graph",
    "hub_and_spoke_graph",
    "layered_graph",
    "star_graph",
    "tree_graph",
]
