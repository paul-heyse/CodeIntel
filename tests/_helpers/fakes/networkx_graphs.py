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


def empty_graph() -> nx.Graph:
    """Create an empty undirected graph.

    Returns
    -------
    nx.Graph
        Empty undirected graph instance.
    """
    return nx.Graph()


def empty_digraph() -> nx.DiGraph:
    """Create an empty directed graph.

    Returns
    -------
    nx.DiGraph
        Empty directed graph instance.
    """
    return nx.DiGraph()


def barbell_graph_small(
    clique_size: int = 5,
    bridge_size: int = 1,
) -> nx.Graph:
    """Create a small barbell graph used across community tests.

    Parameters
    ----------
    clique_size
        Size of each clique on either end of the barbell.
    bridge_size
        Number of nodes in the path connecting the cliques.

    Returns
    -------
    nx.Graph
        Undirected barbell graph.
    """
    return nx.barbell_graph(clique_size, bridge_size)


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


def complete_graph(n: int = DEFAULT_COMPLETE_SIZE) -> nx.Graph:
    """Create a complete undirected graph.

    Parameters
    ----------
    n
        Number of nodes.

    Returns
    -------
    nx.Graph
        Complete undirected graph.
    """
    return nx.complete_graph(n)


def single_node_graph(node: str | int = "A") -> nx.Graph:
    """Create an undirected graph with a single node.

    Returns
    -------
    nx.Graph
        Graph containing the provided node.
    """
    graph = nx.Graph()
    graph.add_node(node)
    return graph


def single_node_digraph(node: str | int = "A") -> nx.DiGraph:
    """Create a directed graph with a single node.

    Returns
    -------
    nx.DiGraph
        Graph containing the provided node.
    """
    graph = nx.DiGraph()
    graph.add_node(node)
    return graph


def single_edge_graph(u: str | int = "A", v: str | int = "B") -> nx.Graph:
    """Create an undirected graph with a single edge.

    Returns
    -------
    nx.Graph
        Graph containing a single undirected edge.
    """
    graph = nx.Graph()
    graph.add_edge(u, v)
    return graph


def single_edge_digraph(u: str | int = "A", v: str | int = "B") -> nx.DiGraph:
    """Create a directed graph with a single edge.

    Returns
    -------
    nx.DiGraph
        Graph containing a single directed edge.
    """
    graph = nx.DiGraph()
    graph.add_edge(u, v)
    return graph


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


def hub_dependencies_graph() -> nx.DiGraph:
    """Create a hub dependency graph with many modules depending on core.

    Returns
    -------
    nx.DiGraph
        Graph where several modules depend on a single core node.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("module_a", "core"),
            ("module_b", "core"),
            ("module_c", "core"),
            ("module_d", "core"),
        ]
    )
    return g


def god_module_graph() -> nx.DiGraph:
    """Create a graph where a single module depends on many others.

    Returns
    -------
    nx.DiGraph
        Graph with a single highly dependent module.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("god", "module_a"),
            ("god", "module_b"),
            ("god", "module_c"),
            ("god", "module_d"),
        ]
    )
    return g


def bidirectional_deps_graph() -> nx.DiGraph:
    """Create a bidirectional dependency graph with a simple cycle.

    Returns
    -------
    nx.DiGraph
        Graph containing a two-node cycle.
    """
    g = nx.DiGraph()
    g.add_edges_from([("module_a", "module_b"), ("module_b", "module_a")])
    return g


def linear_dependency_graph() -> nx.DiGraph:
    """Create a simple linear dependency chain A -> B -> C.

    Returns
    -------
    nx.DiGraph
        Graph representing a simple chain.
    """
    g = nx.DiGraph()
    g.add_edges_from([("module_a", "module_b"), ("module_b", "module_c")])
    return g


def independent_modules_graph() -> nx.DiGraph:
    """Create a graph with independent modules and no edges.

    Returns
    -------
    nx.DiGraph
        Graph containing unconnected module nodes.
    """
    g = nx.DiGraph()
    g.add_nodes_from(["module_a", "module_b", "module_c"])
    return g


def two_sccs_graph() -> nx.DiGraph:
    """Create a graph with two strongly connected components connected by an edge.

    Returns
    -------
    nx.DiGraph
        Graph composed of two SCCs linked in sequence.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "A"),
            ("C", "D"),
            ("D", "C"),
            ("B", "C"),
        ]
    )
    return g


def complex_sccs_graph() -> nx.DiGraph:
    """Create a graph with SCCs of sizes 1, 2, and 3 connected linearly.

    Returns
    -------
    nx.DiGraph
        Graph combining singleton, pair, and triple SCCs.
    """
    g = nx.DiGraph()
    g.add_node("A")
    g.add_edges_from([("B", "C"), ("C", "D"), ("D", "B")])
    g.add_edges_from([("E", "F"), ("F", "E")])
    g.add_edge("A", "B")
    g.add_edge("D", "E")
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


def bridged_cliques_graph(
    clique1_size: int = 3,
    clique2_size: int = 3,
    *,
    bridge_from: str = "a0",
    bridge_to: str = "b0",
) -> nx.Graph:
    """Create two cliques connected by a single bridge edge.

    Parameters
    ----------
    clique1_size
        Number of nodes in the first clique (named a0, a1, ...).
    clique2_size
        Number of nodes in the second clique (named b0, b1, ...).
    bridge_from
        Node in the first clique to connect across the bridge.
    bridge_to
        Node in the second clique to connect across the bridge.

    Returns
    -------
    nx.Graph
        Undirected graph with two complete subgraphs joined by one edge.
    """
    g = nx.Graph()
    clique1_nodes = [f"a{i}" for i in range(clique1_size)]
    clique2_nodes = [f"b{i}" for i in range(clique2_size)]

    g.add_nodes_from(clique1_nodes + clique2_nodes)
    # Add full cliques
    for i, src in enumerate(clique1_nodes):
        for dst in clique1_nodes[i + 1 :]:
            g.add_edge(src, dst)
    for i, src in enumerate(clique2_nodes):
        for dst in clique2_nodes[i + 1 :]:
            g.add_edge(src, dst)
    # Bridge
    g.add_edge(bridge_from, bridge_to)
    return g


def self_loop_graph(node: str = "A") -> nx.DiGraph:
    """Create a graph containing a single self-loop.

    Returns
    -------
    nx.DiGraph
        Directed graph with one node that has a self-loop.
    """
    g = nx.DiGraph()
    g.add_edge(node, node)
    return g


def nested_loop_graph() -> nx.DiGraph:
    """Create a graph with nested loops (outer and inner).

    Returns
    -------
    nx.DiGraph
        Directed graph containing outer and inner cycles.
    """
    g = nx.DiGraph()
    # Outer loop: A -> B -> C -> A
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # Inner loop: B -> D -> B
    g.add_edges_from([("B", "D"), ("D", "B")])
    return g


def fork_join_cfg(
    *,
    entry: str = "entry",
    branch: str = "branch",
    left: str = "left",
    right: str = "right",
    join: str = "join",
) -> nx.DiGraph:
    r"""Create a simple fork/join control-flow graph.

    Structure:
    entry -> branch
            /     \
         left    right
            \\   //
             join

    Parameters
    ----------
    entry
        Name of the entry node.
    branch
        Node where control splits.
    left
        Name of the left branch node.
    right
        Name of the right branch node.
    join
        Node where the branches reconverge.

    Returns
    -------
    nx.DiGraph
        Directed graph with a fork and join.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (entry, branch),
            (branch, left),
            (branch, right),
            (left, join),
            (right, join),
        ]
    )
    return g


def while_loop_cfg(
    *,
    entry: str = "entry",
    condition: str = "condition",
    body: str = "body",
    exit_node: str = "exit",
) -> nx.DiGraph:
    r"""Create a simple while-loop control-flow graph.

    Structure:
    entry -> condition -> body -> condition (back edge)
                     \\-> exit

    Parameters
    ----------
    entry
        Entry node before the loop condition.
    condition
        Node representing the loop guard.
    body
        Node executed when the condition is true.
    exit_node
        Node reached when the condition is false.

    Returns
    -------
    nx.DiGraph
        Directed graph modeling a while loop with an exit edge.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (entry, condition),
            (condition, body),
            (condition, exit_node),
            (body, condition),
        ]
    )
    return g


def scc_with_tail_graph() -> nx.DiGraph:
    """Create a graph with a single SCC feeding a DAG tail.

    Returns
    -------
    nx.DiGraph
        Graph containing an SCC connected to a linear tail.
    """
    g = nx.DiGraph()
    # SCC: A -> B -> C -> A
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # Tail: C -> D -> E
    g.add_edges_from([("C", "D"), ("D", "E")])
    return g


def two_cycle_graph() -> nx.DiGraph:
    """Create a graph with two disjoint 2-cycles.

    Returns
    -------
    nx.DiGraph
        Directed graph containing two separate 2-cycles.
    """
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "A")])
    g.add_edges_from([("C", "D"), ("D", "C")])
    return g


def dag_to_cycle_graph(
    *,
    entry: str = "entry",
    exit_node: str = "exit",
    cycle_nodes: tuple[str, str, str] = ("A", "B", "C"),
) -> nx.DiGraph:
    """Create a DAG prefix feeding into a 3-node cycle with a tail exit.

    Structure:
        entry -> n1 -> n2 -> n3 -> n1 (cycle)
        entry -> exit
        n3 -> exit (tail out of the cycle)

    Parameters
    ----------
    entry
        Entry node before the cycle.
    exit_node
        Exit node after the cycle.
    cycle_nodes
        Names for the three cycle nodes in order.

    Returns
    -------
    nx.DiGraph
        Graph combining a line into a cycle with an exit edge.
    """
    n1, n2, n3 = cycle_nodes
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (entry, n1),
            (n1, n2),
            (n2, n3),
            (n3, n1),
            (entry, exit_node),
            (n3, exit_node),
        ]
    )
    return g


def shared_neighbors_graph(
    shared: int,
    *,
    primary: tuple[str, str] = ("p1", "p2"),
    unique_first: int = 0,
    unique_second: int = 0,
) -> nx.Graph:
    """Create a bipartite-style graph with two primary nodes sharing neighbors.

    Parameters
    ----------
    shared
        Number of shared secondary neighbors between the two primary nodes.
    primary
        Labels for the two primary nodes.
    unique_first
        Number of secondary neighbors connected only to the first primary node.
    unique_second
        Number of secondary neighbors connected only to the second primary node.

    Returns
    -------
    nx.Graph
        Undirected graph containing the specified shared and unique neighbors.
    """
    first, second = primary
    g = nx.Graph()
    g.add_nodes_from(primary)
    shared_nodes = [f"s{i}" for i in range(shared)]
    first_uniques = [f"{first}_u{i}" for i in range(unique_first)]
    second_uniques = [f"{second}_u{i}" for i in range(unique_second)]

    for node in shared_nodes:
        g.add_edge(first, node)
        g.add_edge(second, node)
    for node in first_uniques:
        g.add_edge(first, node)
    for node in second_uniques:
        g.add_edge(second, node)
    return g


__all__ = [
    "DEFAULT_CHAIN_LENGTH",
    "DEFAULT_COMPLETE_SIZE",
    "DEFAULT_CYCLE_SIZE",
    "DEFAULT_SPOKES",
    "barbell_graph_small",
    "bidirectional_deps_graph",
    "bipartite_graph",
    "bridged_cliques_graph",
    "chain_graph",
    "complete_digraph",
    "complete_graph",
    "complex_sccs_graph",
    "cyclic_graph",
    "dag_to_cycle_graph",
    "diamond_graph",
    "disconnected_graph",
    "empty_digraph",
    "empty_graph",
    "fork_join_cfg",
    "god_module_graph",
    "hub_and_spoke_graph",
    "hub_dependencies_graph",
    "independent_modules_graph",
    "layered_graph",
    "linear_dependency_graph",
    "nested_loop_graph",
    "scc_with_tail_graph",
    "self_loop_graph",
    "shared_neighbors_graph",
    "single_edge_digraph",
    "single_edge_graph",
    "single_node_digraph",
    "single_node_graph",
    "star_graph",
    "tree_graph",
    "two_cycle_graph",
    "two_sccs_graph",
    "while_loop_cfg",
]
