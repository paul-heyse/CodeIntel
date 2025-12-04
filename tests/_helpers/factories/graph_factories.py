"""Pre-built graph structures for testing graph algorithms.

This module provides factory functions for creating realistic NetworkX graphs
that simulate production patterns. These graphs are used to test graph metrics,
algorithms, and analytics without requiring actual production data.

Factory Functions
-----------------
build_layered_call_graph
    Build realistic call graph with hub functions, layers, and SCCs.
build_layered_import_graph
    Build import graph with intentional cycles.
build_star_graph
    Build star graph for hub detection tests.
build_chain_graph
    Build linear chain for layer/betweenness tests.
build_cycle_graph
    Build cycle for SCC detection tests.
build_dense_cluster
    Build a densely connected cluster for community detection tests.
"""

from __future__ import annotations

from typing import Final

import networkx as nx

# =============================================================================
# Constants for realistic graph validation
# =============================================================================

GOLDEN_MIN_NODES: Final[int] = 13
GOLDEN_MIN_EDGES: Final[int] = 30
GOLDEN_EXPECTED_COMMUNITIES: Final[int] = 2
GOLDEN_EXPECTED_SCC: Final[int] = 1


# =============================================================================
# Realistic Production-Like Graphs
# =============================================================================


def build_layered_call_graph() -> nx.DiGraph:
    """Build a realistic call graph simulating production patterns.

    Creates a directed graph with:
    - Layer 0: Core utilities (no internal deps)
    - Layer 1: Services (depend on core)
    - Layer 2: Handlers (depend on services, core)
    - Layer 3: API (depend on handlers)
    - Hub function (log_info) called by many
    - Small SCC for auth/cache interaction

    Returns
    -------
    nx.DiGraph
        A directed graph with hub functions, layered architecture, and SCCs.

    Examples
    --------
    >>> graph = build_layered_call_graph()
    >>> len(graph.nodes()) >= 13
    True
    >>> len(graph.edges()) >= 30
    True
    """
    g = nx.DiGraph()

    # Layer 0: Core utilities (no internal deps)
    core_funcs = ["format_string", "parse_json", "validate_input", "hash_value"]
    g.add_nodes_from(core_funcs)

    # Layer 1: Services (depend on core)
    services = ["authenticate", "query", "execute", "get_cached", "set_cached"]
    g.add_nodes_from(services)
    for s in services:
        g.add_edge(s, "validate_input")
        g.add_edge(s, "format_string")

    # Layer 2: Handlers (depend on services, core)
    handlers = ["create_user", "get_user", "update_user", "delete_user", "create_order"]
    g.add_nodes_from(handlers)
    for h in handlers:
        g.add_edge(h, "authenticate")
        g.add_edge(h, "query")
        g.add_edge(h, "get_cached")

    # Layer 3: API (depend on handlers)
    api = ["handle_request", "register_routes"]
    g.add_nodes_from(api)
    for a in api:
        for h in handlers:
            g.add_edge(a, h)

    # Hub function: log_info is called by many
    g.add_node("log_info")
    for node in services + handlers:
        g.add_edge(node, "log_info")

    # Small SCC: auth <-> cache interaction
    g.add_edge("authenticate", "get_cached")
    g.add_edge("get_cached", "authenticate")  # Cache validates with auth

    return g


def build_layered_import_graph() -> nx.DiGraph:
    """Build a realistic import graph with layered architecture.

    Creates a directed graph representing module imports with:
    - Core modules (types, errors, config, utils)
    - Service modules (auth, cache, database)
    - Handler modules (user, product, order)
    - API modules (routes, middleware)
    - Cross-cutting logging module
    - Intentional cycle between auth and cache

    Returns
    -------
    nx.DiGraph
        A directed graph representing module imports.

    Examples
    --------
    >>> graph = build_layered_import_graph()
    >>> "core.utils" in graph.nodes()
    True
    >>> "services.auth" in graph.nodes()
    True
    """
    g = nx.DiGraph()

    # Core modules
    core = ["core.utils", "core.types", "core.errors", "core.config"]
    g.add_nodes_from(core)

    # Service modules
    services = ["services.auth", "services.cache", "services.database"]
    g.add_nodes_from(services)
    for s in services:
        g.add_edge(s, "core.utils")
        g.add_edge(s, "core.errors")

    # Handler modules
    handlers = ["handlers.user", "handlers.product", "handlers.order"]
    g.add_nodes_from(handlers)
    for h in handlers:
        g.add_edge(h, "services.auth")
        g.add_edge(h, "services.database")
        g.add_edge(h, "core.errors")

    # API modules
    api = ["api.routes", "api.middleware"]
    g.add_nodes_from(api)
    for a in api:
        for h in handlers:
            g.add_edge(a, h)

    # Cross-cutting: utils.logging imported by many
    g.add_node("utils.logging")
    for node in services + handlers + api:
        g.add_edge(node, "utils.logging")

    # Intentional cycle: services.auth <-> services.cache
    g.add_edge("services.auth", "services.cache")
    g.add_edge("services.cache", "services.auth")

    return g


# =============================================================================
# Simple Structural Graphs for Unit Tests
# =============================================================================


def build_star_graph(center: str | int = 0, leaves: int = 10) -> nx.DiGraph:
    """Build star graph for hub detection tests.

    Creates a directed graph where all leaf nodes point to a central hub node.
    Useful for testing hub detection algorithms like PageRank.

    Parameters
    ----------
    center
        Identifier for the center (hub) node.
    leaves
        Number of leaf nodes pointing to the center.

    Returns
    -------
    nx.DiGraph
        A star graph with edges pointing inward to center.

    Examples
    --------
    >>> graph = build_star_graph("hub", leaves=5)
    >>> len(graph.nodes())
    6
    >>> graph.in_degree("hub")
    5
    """
    g = nx.DiGraph()
    g.add_node(center)
    for i in range(leaves):
        leaf = f"leaf_{i}" if isinstance(center, str) else i + 1
        g.add_node(leaf)
        g.add_edge(leaf, center)
    return g


def build_chain_graph(length: int = 10, prefix: str = "node") -> nx.DiGraph:
    """Build linear chain for layer/betweenness tests.

    Creates a directed graph forming a linear chain from first to last node.
    Useful for testing betweenness centrality and layer detection.

    Parameters
    ----------
    length
        Number of nodes in the chain.
    prefix
        Prefix for node names.

    Returns
    -------
    nx.DiGraph
        A linear chain graph.

    Examples
    --------
    >>> graph = build_chain_graph(length=5, prefix="func")
    >>> len(graph.nodes())
    5
    >>> len(graph.edges())
    4
    """
    g = nx.DiGraph()
    nodes = [f"{prefix}_{i}" for i in range(length)]
    g.add_nodes_from(nodes)
    for i in range(length - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def build_cycle_graph(size: int = 5, prefix: str = "node") -> nx.DiGraph:
    """Build cycle for SCC detection tests.

    Creates a directed graph forming a single cycle.
    The entire graph is one strongly connected component.

    Parameters
    ----------
    size
        Number of nodes in the cycle.
    prefix
        Prefix for node names.

    Returns
    -------
    nx.DiGraph
        A cycle graph.

    Examples
    --------
    >>> graph = build_cycle_graph(size=4, prefix="n")
    >>> len(graph.nodes())
    4
    >>> len(graph.edges())
    4
    """
    g = nx.DiGraph()
    nodes = [f"{prefix}_{i}" for i in range(size)]
    g.add_nodes_from(nodes)
    for i in range(size):
        g.add_edge(nodes[i], nodes[(i + 1) % size])
    return g


def build_dense_cluster(size: int = 5, prefix: str = "node") -> nx.DiGraph:
    """Build a densely connected cluster for community detection tests.

    Creates a complete directed graph where every node connects to every
    other node. Useful for testing community detection and clustering.

    Parameters
    ----------
    size
        Number of nodes in the cluster.
    prefix
        Prefix for node names.

    Returns
    -------
    nx.DiGraph
        A complete directed graph.

    Examples
    --------
    >>> graph = build_dense_cluster(size=4, prefix="c")
    >>> len(graph.nodes())
    4
    >>> len(graph.edges())
    12
    """
    g = nx.DiGraph()
    nodes = [f"{prefix}_{i}" for i in range(size)]
    g.add_nodes_from(nodes)
    for source in nodes:
        for target in nodes:
            if source != target:
                g.add_edge(source, target)
    return g


def build_two_communities_graph(
    community_a_size: int = 5,
    community_b_size: int = 5,
    inter_edges: int = 1,
) -> nx.DiGraph:
    """Build a graph with two distinct communities connected by few edges.

    Useful for testing community detection algorithms.

    Parameters
    ----------
    community_a_size
        Number of nodes in community A.
    community_b_size
        Number of nodes in community B.
    inter_edges
        Number of edges connecting the two communities.

    Returns
    -------
    nx.DiGraph
        A graph with two communities.

    Examples
    --------
    >>> graph = build_two_communities_graph(3, 3, 1)
    >>> len(graph.nodes())
    6
    """
    g = nx.DiGraph()

    # Community A (densely connected)
    a_nodes = [f"a_{i}" for i in range(community_a_size)]
    g.add_nodes_from(a_nodes)
    for i, source in enumerate(a_nodes):
        for target in a_nodes[i + 1 :]:
            g.add_edge(source, target)
            g.add_edge(target, source)

    # Community B (densely connected)
    b_nodes = [f"b_{i}" for i in range(community_b_size)]
    g.add_nodes_from(b_nodes)
    for i, source in enumerate(b_nodes):
        for target in b_nodes[i + 1 :]:
            g.add_edge(source, target)
            g.add_edge(target, source)

    # Inter-community edges
    for i in range(min(inter_edges, community_a_size, community_b_size)):
        g.add_edge(a_nodes[i], b_nodes[i])

    return g


def build_simple_call_graph() -> nx.DiGraph:
    """Build a simple call graph with cycle for basic testing.

    Creates a minimal call graph representing:
    - main -> process -> validate -> helper
    - process -> util
    - helper -> util (shared dependency)
    - validate -> process (creates cycle)

    Returns
    -------
    nx.DiGraph
        A simple directed call graph with 5 nodes and 6 edges.

    Examples
    --------
    >>> graph = build_simple_call_graph()
    >>> len(graph.nodes())
    5
    >>> len(graph.edges())
    6
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (1001, 1002),  # main -> process
            (1002, 1003),  # process -> validate
            (1003, 1004),  # validate -> helper
            (1002, 1005),  # process -> util
            (1004, 1005),  # helper -> util
            (1003, 1002),  # validate -> process (cycle)
        ]
    )
    return g


def build_simple_import_graph() -> nx.DiGraph:
    """Build a simple import graph for basic testing.

    Creates a minimal import graph representing:
    - main -> core -> utils
    - main -> helpers
    - helpers -> utils

    Returns
    -------
    nx.DiGraph
        A simple directed import graph with 4 modules.

    Examples
    --------
    >>> graph = build_simple_import_graph()
    >>> len(graph.nodes())
    4
    >>> len(graph.edges())
    4
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("mypackage.main", "mypackage.core"),
            ("mypackage.core", "mypackage.utils"),
            ("mypackage.main", "mypackage.helpers"),
            ("mypackage.helpers", "mypackage.utils"),
        ]
    )
    return g


def build_dag_with_bottleneck(
    layers: int = 4,
    width: int = 3,
    bottleneck_width: int = 1,
    bottleneck_layer: int = 2,
) -> nx.DiGraph:
    """Build a DAG with a bottleneck layer for betweenness testing.

    Creates a layered DAG where one layer is significantly narrower,
    forcing all paths through a bottleneck.

    Parameters
    ----------
    layers
        Total number of layers.
    width
        Default width of each layer.
    bottleneck_width
        Width of the bottleneck layer.
    bottleneck_layer
        Which layer is the bottleneck (0-indexed).

    Returns
    -------
    nx.DiGraph
        A DAG with a bottleneck.

    Examples
    --------
    >>> graph = build_dag_with_bottleneck(layers=4, width=3, bottleneck_width=1)
    >>> len(graph.nodes()) >= 4
    True
    """
    g = nx.DiGraph()

    prev_layer: list[str] = []
    for layer_idx in range(layers):
        layer_width = bottleneck_width if layer_idx == bottleneck_layer else width
        current_layer = [f"layer{layer_idx}_node{i}" for i in range(layer_width)]
        g.add_nodes_from(current_layer)

        # Connect from previous layer to current
        if prev_layer:
            for source in prev_layer:
                for target in current_layer:
                    g.add_edge(source, target)

        prev_layer = current_layer

    return g


__all__ = [
    "GOLDEN_EXPECTED_COMMUNITIES",
    "GOLDEN_EXPECTED_SCC",
    "GOLDEN_MIN_EDGES",
    "GOLDEN_MIN_NODES",
    "build_chain_graph",
    "build_cycle_graph",
    "build_dag_with_bottleneck",
    "build_dense_cluster",
    "build_layered_call_graph",
    "build_layered_import_graph",
    "build_simple_call_graph",
    "build_simple_import_graph",
    "build_star_graph",
    "build_two_communities_graph",
]
