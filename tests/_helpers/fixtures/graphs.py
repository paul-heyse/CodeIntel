"""Unified graph fixtures for test helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final, Literal

from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

GraphKind = Literal["chain", "star", "cycle", "layered", "golden", "custom"]

_ALPHABET_SIZE: Final[int] = 26
_GOLDEN_MIN_NODES: Final[int] = 5


@dataclass(frozen=True)
class GraphFixtureSpec:
    """Specification for building a test graph."""

    kind: GraphKind
    directed: bool = True
    nodes: int | None = None
    edges: int | None = None
    layers: tuple[int, ...] | None = None
    spokes: int | None = None
    cycle_size: int | None = None
    seed: int | None = None


class GraphFixtureFactory:
    """Factory for building standardized rustworkx stores."""

    @staticmethod
    def build(spec: GraphFixtureSpec) -> RxGraphStore:
        """Build a graph from a fixture specification.

        Returns
        -------
        RxGraphStore
            Graph store instance matching the fixture specification.
        """
        if spec.kind == "chain":
            return _build_chain(spec)
        if spec.kind == "star":
            return _build_star(spec)
        if spec.kind == "cycle":
            return _build_cycle(spec)
        if spec.kind == "layered":
            return _build_layered(spec)
        if spec.kind == "golden":
            return _build_golden(spec)
        return _build_custom(spec)


def _make_store(*, directed: bool) -> RxGraphStore:
    return RxGraphStore.directed() if directed else RxGraphStore.undirected()


def _add_edges(
    store: RxGraphStore,
    edges: Sequence[tuple[object, object]],
    *,
    weight: float = 1.0,
) -> None:
    for src, dst in edges:
        store.add_weighted_edge(src, dst, weight=weight)


def _reverse_store(store: RxGraphStore) -> RxGraphStore:
    if not store.is_directed:
        return store
    reversed_store = RxGraphStore.directed(
        node_hint=store.graph.num_nodes(),
        edge_hint=store.graph.num_edges(),
    )
    for node_id in store.node_ids():
        reversed_store.set_node_attrs(node_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        weight = edge_weight_from_payload(store.graph.get_edge_data(src_idx, dst_idx))
        reversed_store.add_weighted_edge(dst_id, src_id, weight=weight)
    return reversed_store


def _relabel_store(store: RxGraphStore, mapping: Mapping[object, object]) -> RxGraphStore:
    relabeled = RxGraphStore.directed() if store.is_directed else RxGraphStore.undirected()
    for node_id in store.node_ids():
        new_id = mapping.get(node_id, node_id)
        relabeled.set_node_attrs(new_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = mapping.get(store.index_to_id[src_idx], store.index_to_id[src_idx])
        dst_id = mapping.get(store.index_to_id[dst_idx], store.index_to_id[dst_idx])
        weight = edge_weight_from_payload(store.graph.get_edge_data(src_idx, dst_idx))
        relabeled.add_weighted_edge(src_id, dst_id, weight=weight)
    return relabeled


def _set_edge_weight(store: RxGraphStore, src_id: object, dst_id: object, weight: float) -> None:
    src_idx = store.id_to_index.get(src_id)
    dst_idx = store.id_to_index.get(dst_id)
    if src_idx is None or dst_idx is None:
        return
    if store.graph.has_edge(src_idx, dst_idx):
        store.graph.update_edge(src_idx, dst_idx, float(weight))


def _set_uniform_edge_weights(store: RxGraphStore, weight: float) -> None:
    for src_idx, dst_idx in store.graph.edge_list():
        store.graph.update_edge(src_idx, dst_idx, float(weight))


@dataclass
class GraphFixtures:
    """Bundled graph fixtures for analytics graph tests."""

    call_graph: RxGraphStore
    import_graph: RxGraphStore
    config_graph: RxGraphStore
    symbol_module_graph: RxGraphStore
    symbol_function_graph: RxGraphStore
    cfg_graph: RxGraphStore | None = None


STANDARD_CALL: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="chain", directed=True, nodes=4)
STANDARD_IMPORT: Final[GraphFixtureSpec] = GraphFixtureSpec(
    kind="cycle", directed=True, cycle_size=3
)
GOLDEN_CALL: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="golden", directed=True)
GOLDEN_IMPORT: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="golden", directed=True)


def _build_chain(spec: GraphFixtureSpec) -> RxGraphStore:
    graph = _make_store(directed=spec.directed)
    length = spec.nodes or 0
    if length < 1:
        return graph

    def node_label(i: int) -> str:
        return chr(ord("A") + i % _ALPHABET_SIZE) if i < _ALPHABET_SIZE else f"N{i}"

    nodes = [node_label(i) for i in range(length)]
    for node in nodes:
        graph.ensure_node(node)
    for i in range(length - 1):
        graph.add_weighted_edge(nodes[i], nodes[i + 1], weight=1.0)
    return graph


def _build_star(spec: GraphFixtureSpec) -> RxGraphStore:
    graph = _make_store(directed=spec.directed)
    spokes = spec.spokes or 0
    hub = "hub"
    graph.ensure_node(hub)
    for idx in range(spokes):
        leaf = f"spoke{idx + 1}"
        graph.ensure_node(leaf)
        graph.add_weighted_edge(hub, leaf, weight=1.0)
    return graph


def _build_cycle(spec: GraphFixtureSpec) -> RxGraphStore:
    size = spec.cycle_size or spec.nodes or 0
    graph = _build_chain(GraphFixtureSpec(kind="chain", directed=spec.directed, nodes=size))
    nodes = graph.node_ids()
    if len(nodes) > 1:
        graph.add_weighted_edge(nodes[-1], nodes[0], weight=1.0)
    return graph


def _build_layered(spec: GraphFixtureSpec) -> RxGraphStore:
    graph = _make_store(directed=spec.directed)
    layers = spec.layers or (4, 5, 3, 2)
    layer_nodes: list[list[str]] = []
    for layer_index, count in enumerate(layers):
        nodes = [f"L{layer_index}_{idx}" for idx in range(count)]
        for node in nodes:
            graph.ensure_node(node)
        layer_nodes.append(nodes)
    for idx in range(len(layer_nodes) - 1):
        for node in layer_nodes[idx]:
            for downstream in layer_nodes[idx + 1]:
                graph.add_weighted_edge(node, downstream, weight=1.0)
    return graph


def _build_golden(spec: GraphFixtureSpec) -> RxGraphStore:
    layered = GraphFixtureSpec(kind="layered", directed=spec.directed, layers=(4, 5, 5, 3))
    graph = _build_layered(layered)
    if spec.directed and graph.graph.num_nodes() >= _GOLDEN_MIN_NODES:
        nodes = graph.node_ids()
        graph.add_weighted_edge(nodes[1], nodes[2], weight=1.0)
        graph.add_weighted_edge(nodes[2], nodes[1], weight=1.0)
    return graph


def _build_custom(spec: GraphFixtureSpec) -> RxGraphStore:
    graph = _make_store(directed=spec.directed)
    if spec.nodes is not None:
        for node in range(spec.nodes):
            graph.ensure_node(node)
    return graph


DEFAULT_CHAIN_LENGTH: Final[int] = 4
DEFAULT_SPOKES: Final[int] = 3
DEFAULT_CYCLE_SIZE: Final[int] = 3
DEFAULT_COMPLETE_SIZE: Final[int] = 5
_MIN_CYCLE_SIZE: Final[int] = 2
_MIN_LAYER_SPAN: Final[int] = 3


def empty_graph() -> RxGraphStore:
    """Create an empty undirected graph.

    Returns
    -------
    RxGraphStore
        Empty undirected graph store instance.
    """
    return RxGraphStore.undirected()


def empty_digraph() -> RxGraphStore:
    """Create an empty directed graph.

    Returns
    -------
    RxGraphStore
        Empty directed graph store instance.
    """
    return RxGraphStore.directed()


def barbell_graph_small(
    clique_size: int = 5,
    bridge_size: int = 1,
) -> RxGraphStore:
    """Create a small barbell graph used across community tests.

    Parameters
    ----------
    clique_size
        Size of each clique on either end of the barbell.
    bridge_size
        Number of nodes in the path connecting the cliques.

    Returns
    -------
    RxGraphStore
        Undirected barbell graph store.
    """
    store = RxGraphStore.undirected()
    total_nodes = (clique_size * 2) + bridge_size
    for node in range(total_nodes):
        store.ensure_node(node)

    for left in range(clique_size):
        for right in range(left + 1, clique_size):
            store.add_weighted_edge(left, right, weight=1.0)

    right_start = clique_size + bridge_size
    for left in range(right_start, right_start + clique_size):
        for right in range(left + 1, right_start + clique_size):
            store.add_weighted_edge(left, right, weight=1.0)

    path_nodes = [clique_size - 1]
    path_nodes.extend(range(clique_size, clique_size + bridge_size))
    path_nodes.append(right_start)
    for left, right in pairwise(path_nodes):
        store.add_weighted_edge(left, right, weight=1.0)
    return store


def chain_graph(length: int = DEFAULT_CHAIN_LENGTH) -> RxGraphStore:
    """Create a simple chain (path) graph.

    Structure: A -> B -> C -> D (for length=4)

    Parameters
    ----------
    length
        Number of nodes in the chain.

    Returns
    -------
    RxGraphStore
        A directed chain graph store.

    Example
    -------
    >>> g = chain_graph(4)
    >>> edges = sorted(
    ...     [(g.index_to_id[src], g.index_to_id[dst]) for src, dst in g.graph.edge_list()]
    ... )
    >>> edges
    [('A', 'B'), ('B', 'C'), ('C', 'D')]
    """
    spec = GraphFixtureSpec(kind="chain", directed=True, nodes=length)
    return GraphFixtureFactory.build(spec)


def call_chain_graph(length: int = DEFAULT_CHAIN_LENGTH) -> RxGraphStore:
    """Build a call graph with a simple chain topology.

    Returns
    -------
    RxGraphStore
        Directed chain graph store.
    """
    return chain_graph(length)


def star_graph(spokes: int = DEFAULT_SPOKES, *, inward: bool = False) -> RxGraphStore:
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
    RxGraphStore
        A directed star graph store.

    Example
    -------
    >>> g = star_graph(3)
    >>> edges = sorted(
    ...     [(g.index_to_id[src], g.index_to_id[dst]) for src, dst in g.graph.edge_list()]
    ... )
    >>> edges
    [('hub', 'spoke1'), ('hub', 'spoke2'), ('hub', 'spoke3')]

    >>> g = star_graph(3, inward=True)
    >>> edges = sorted(
    ...     [(g.index_to_id[src], g.index_to_id[dst]) for src, dst in g.graph.edge_list()]
    ... )
    >>> edges
    [('spoke1', 'hub'), ('spoke2', 'hub'), ('spoke3', 'hub')]
    """
    spec = GraphFixtureSpec(kind="star", directed=True, spokes=spokes)
    graph = GraphFixtureFactory.build(spec)
    return _reverse_store(graph) if inward else graph


def call_star_graph(spokes: int = DEFAULT_SPOKES, *, inward: bool = False) -> RxGraphStore:
    """Build a call graph with a star topology.

    Returns
    -------
    RxGraphStore
        Directed star graph store.
    """
    return star_graph(spokes, inward=inward)


def weighted_star_graph(
    spokes: int = DEFAULT_SPOKES,
    *,
    weight: float = 1.0,
    inward: bool = False,
) -> RxGraphStore:
    """Create a weighted star graph with consistent edge attributes.

    Returns
    -------
    RxGraphStore
        Star graph store with weight attributes on each edge.
    """
    g = star_graph(spokes, inward=inward)
    _set_uniform_edge_weights(g, weight)
    return g


def diamond_graph() -> RxGraphStore:
    r"""Create a diamond-shaped graph.

    Structure: A -> B, A -> C, B -> D, C -> D

        A
       / \
      B   C
       \ /
        D

    Returns
    -------
    RxGraphStore
        A diamond graph store with 4 nodes.

    Example
    -------
    >>> g = diamond_graph()
    >>> g.graph.num_nodes()
    4
    >>> g.graph.num_edges()
    4
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return g


def cyclic_graph(size: int = DEFAULT_CYCLE_SIZE) -> RxGraphStore:
    """Create a simple cycle graph.

    Structure: A -> B -> C -> A (for size=3)

    Parameters
    ----------
    size
        Number of nodes in the cycle.

    Returns
    -------
    RxGraphStore
        A directed cycle graph store.

    Example
    -------
    >>> g = cyclic_graph(3)
    >>> edges = [(g.index_to_id[src], g.index_to_id[dst]) for src, dst in g.graph.edge_list()]
    >>> edges
    [('A', 'B'), ('B', 'C'), ('C', 'A')]
    """
    g = RxGraphStore.directed()
    if size < _MIN_CYCLE_SIZE:
        return g

    def node_label(i: int) -> str:
        return chr(ord("A") + i % _ALPHABET_SIZE) if i < _ALPHABET_SIZE else f"N{i}"

    nodes = [node_label(i) for i in range(size)]
    for node in nodes:
        g.ensure_node(node)

    for i in range(size):
        g.add_weighted_edge(nodes[i], nodes[(i + 1) % size], weight=1.0)

    return g


def import_cycle_graph(size: int = DEFAULT_CYCLE_SIZE) -> RxGraphStore:
    """Build an import graph with a directed cycle.

    Returns
    -------
    RxGraphStore
        Directed cycle graph store.
    """
    return cyclic_graph(size)


def symbol_star_graph(spokes: int = DEFAULT_SPOKES) -> RxGraphStore:
    """Build a symbol graph with a star topology (undirected).

    Returns
    -------
    RxGraphStore
        Undirected star graph store.
    """
    spec = GraphFixtureSpec(kind="star", directed=False, spokes=spokes)
    return GraphFixtureFactory.build(spec)


def call_graph_fixture(edges: Sequence[tuple[str, str]] | None = None) -> RxGraphStore:
    """Create a small call graph for tests, defaulting to a simple chain.

    Returns
    -------
    RxGraphStore
        Directed call graph store containing the provided edges.
    """
    if edges is None:
        edges = [("func_a", "func_b"), ("func_b", "func_c")]
    graph = RxGraphStore.directed()
    _add_edges(graph, list(edges))
    return graph


def standard_graph_fixtures(
    *,
    chain_length: int = DEFAULT_CHAIN_LENGTH,
    cycle_size: int = DEFAULT_CYCLE_SIZE,
    star_spokes: int = DEFAULT_SPOKES,
) -> GraphFixtures:
    """Build a consistent set of graph fixtures for tests.

    Returns
    -------
    GraphFixtures
        Fixture bundle with call/import/symbol/config graphs.
    """
    call_spec = GraphFixtureSpec(kind="chain", directed=True, nodes=chain_length)
    import_spec = GraphFixtureSpec(kind="cycle", directed=True, cycle_size=cycle_size)
    symbol_spec = GraphFixtureSpec(kind="star", directed=False, spokes=star_spokes)
    return GraphFixtures(
        call_graph=GraphFixtureFactory.build(call_spec),
        import_graph=GraphFixtureFactory.build(import_spec),
        config_graph=RxGraphStore.undirected(),
        symbol_module_graph=GraphFixtureFactory.build(symbol_spec),
        symbol_function_graph=GraphFixtureFactory.build(symbol_spec),
        cfg_graph=RxGraphStore.directed(),
    )


def build_sample_graphs(goids: Mapping[str, int]) -> GraphFixtures:
    """Construct sample graphs used across integration tests.

    Parameters
    ----------
    goids
        GOID mapping keyed by target name.

    Returns
    -------
    GraphFixtures
        Collection of seeded graph objects keyed by purpose.
    """
    call_graph = RxGraphStore.directed()
    _add_edges(
        call_graph,
        [
            (goids["func_a"], goids["func_b"]),
            (goids["func_b"], goids["func_c"]),
        ],
    )
    call_graph.add_weighted_edge(goids["func_a"], goids["func_c"], weight=0.5)

    import_graph = RxGraphStore.directed()
    _add_edges(
        import_graph,
        [
            ("pkg.mod_a", "pkg.mod_b"),
            ("pkg.mod_b", "pkg.mod_c"),
            ("pkg.mod_c", "pkg.mod_a"),
        ],
    )

    config_graph = RxGraphStore.undirected()
    config_graph.set_node_attrs(("config_key", "API_TOKEN"), {"bipartite": 0})
    config_graph.set_node_attrs(("config_key", "FEATURE_FLAG"), {"bipartite": 0})
    config_graph.set_node_attrs(("module", "pkg.mod_a"), {"bipartite": 1})
    config_graph.set_node_attrs(("module", "pkg.mod_b"), {"bipartite": 1})
    config_graph.add_weighted_edge(
        ("config_key", "API_TOKEN"),
        ("module", "pkg.mod_a"),
        weight=1.0,
    )
    config_graph.add_weighted_edge(
        ("config_key", "API_TOKEN"),
        ("module", "pkg.mod_b"),
        weight=1.0,
    )
    config_graph.add_weighted_edge(
        ("config_key", "FEATURE_FLAG"),
        ("module", "pkg.mod_b"),
        weight=2.0,
    )

    symbol_module_graph = RxGraphStore.undirected()
    _add_edges(
        symbol_module_graph,
        [
            ("pkg.mod_a", "pkg.mod_b"),
            ("pkg.mod_a", "pkg.mod_c"),
        ],
    )
    _set_uniform_edge_weights(symbol_module_graph, 1.0)

    symbol_function_graph = RxGraphStore.undirected()
    _add_edges(
        symbol_function_graph,
        [
            (goids["func_a"], goids["func_b"]),
            (goids["func_a"], goids["func_c"]),
        ],
    )
    _set_uniform_edge_weights(symbol_function_graph, 1.0)

    return GraphFixtures(
        call_graph=call_graph,
        import_graph=import_graph,
        config_graph=config_graph,
        symbol_module_graph=symbol_module_graph,
        symbol_function_graph=symbol_function_graph,
    )


def disconnected_graph() -> RxGraphStore:
    """Create a graph with disconnected components.

    Structure: Two separate chains:
    - Component 1: A -> B -> C
    - Component 2: X -> Y -> Z

    Returns
    -------
    RxGraphStore
        A graph store with two disconnected components.

    Example
    -------
    >>> from codeintel.build.graphs.compute.metrics.components import find_weakly_connected
    >>> g = disconnected_graph()
    >>> g.graph.num_nodes()
    6
    >>> len(find_weakly_connected(g))
    2
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("A", "B"), ("B", "C")])
    _add_edges(g, [("X", "Y"), ("Y", "Z")])
    return g


def complete_digraph(n: int = DEFAULT_COMPLETE_SIZE) -> RxGraphStore:
    """Create a complete directed graph.

    Every node has an edge to every other node.

    Parameters
    ----------
    n
        Number of nodes.

    Returns
    -------
    RxGraphStore
        A complete directed graph store.

    Example
    -------
    >>> g = complete_digraph(3)
    >>> g.graph.num_edges()
    6
    """
    graph = RxGraphStore.directed()
    for node in range(n):
        graph.ensure_node(node)
    for src in range(n):
        for dst in range(n):
            if src != dst:
                graph.add_weighted_edge(src, dst, weight=1.0)
    return graph


def complete_graph(n: int = DEFAULT_COMPLETE_SIZE) -> RxGraphStore:
    """Create a complete undirected graph.

    Parameters
    ----------
    n
        Number of nodes.

    Returns
    -------
    RxGraphStore
        Complete undirected graph store.
    """
    graph = RxGraphStore.undirected()
    for node in range(n):
        graph.ensure_node(node)
    for src in range(n):
        for dst in range(src + 1, n):
            graph.add_weighted_edge(src, dst, weight=1.0)
    return graph


def single_node_graph(node: str | int = "A") -> RxGraphStore:
    """Create an undirected graph with a single node.

    Returns
    -------
    RxGraphStore
        Graph store containing the provided node.
    """
    graph = RxGraphStore.undirected()
    graph.ensure_node(node)
    return graph


def single_node_digraph(node: str | int = "A") -> RxGraphStore:
    """Create a directed graph with a single node.

    Returns
    -------
    RxGraphStore
        Graph store containing the provided node.
    """
    graph = RxGraphStore.directed()
    graph.ensure_node(node)
    return graph


def single_edge_graph(u: str | int = "A", v: str | int = "B") -> RxGraphStore:
    """Create an undirected graph with a single edge.

    Returns
    -------
    RxGraphStore
        Graph store containing a single undirected edge.
    """
    graph = RxGraphStore.undirected()
    graph.add_weighted_edge(u, v, weight=1.0)
    return graph


def single_edge_digraph(u: str | int = "A", v: str | int = "B") -> RxGraphStore:
    """Create a directed graph with a single edge.

    Returns
    -------
    RxGraphStore
        Graph store containing a single directed edge.
    """
    graph = RxGraphStore.directed()
    graph.add_weighted_edge(u, v, weight=1.0)
    return graph


def bipartite_graph(left: int = 3, right: int = 3) -> RxGraphStore:
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
    RxGraphStore
        A bipartite directed graph store.

    Example
    -------
    >>> g = bipartite_graph(2, 3)
    >>> g.graph.num_edges()
    6
    """
    g = RxGraphStore.directed()
    left_nodes = [f"L{i}" for i in range(left)]
    right_nodes = [f"R{i}" for i in range(right)]

    for node in left_nodes:
        g.ensure_node(node)
    for node in right_nodes:
        g.ensure_node(node)

    for lnode in left_nodes:
        for rnode in right_nodes:
            g.add_weighted_edge(lnode, rnode, weight=1.0)

    return g


def acyclic_bipartite_flow(
    left: int = DEFAULT_SPOKES,
    right: int = DEFAULT_SPOKES,
    *,
    direction: str = "lr",
) -> RxGraphStore:
    """Create a directed bipartite graph flowing left-to-right or right-to-left.

    Returns
    -------
    RxGraphStore
        Directed bipartite graph store with optional direction reversal.
    """
    g = RxGraphStore.directed()
    left_nodes = [f"L{i}" for i in range(left)]
    right_nodes = [f"R{i}" for i in range(right)]
    for node in left_nodes:
        g.set_node_attrs(node, {"bipartite": 0})
    for node in right_nodes:
        g.set_node_attrs(node, {"bipartite": 1})

    if direction == "rl":
        sources, targets = right_nodes, left_nodes
    else:
        sources, targets = left_nodes, right_nodes

    for src in sources:
        for dst in targets:
            g.add_weighted_edge(src, dst, weight=1.0)
    return g


def hub_dependencies_graph() -> RxGraphStore:
    """Create a hub dependency graph with many modules depending on core.

    Returns
    -------
    RxGraphStore
        Graph store where several modules depend on a single core node.
    """
    g = RxGraphStore.directed()
    _add_edges(
        g,
        [
            ("module_a", "core"),
            ("module_b", "core"),
            ("module_c", "core"),
            ("module_d", "core"),
        ],
    )
    return g


def god_module_graph() -> RxGraphStore:
    """Create a graph where a single module depends on many others.

    Returns
    -------
    RxGraphStore
        Graph store with a single highly dependent module.
    """
    g = RxGraphStore.directed()
    _add_edges(
        g,
        [
            ("god", "module_a"),
            ("god", "module_b"),
            ("god", "module_c"),
            ("god", "module_d"),
        ],
    )
    return g


def bidirectional_deps_graph() -> RxGraphStore:
    """Create a bidirectional dependency graph with a simple cycle.

    Returns
    -------
    RxGraphStore
        Graph store containing a two-node cycle.
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("module_a", "module_b"), ("module_b", "module_a")])
    return g


def linear_dependency_graph() -> RxGraphStore:
    """Create a simple linear dependency chain A -> B -> C.

    Returns
    -------
    RxGraphStore
        Graph store representing a simple chain.
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("module_a", "module_b"), ("module_b", "module_c")])
    return g


def independent_modules_graph() -> RxGraphStore:
    """Create a graph with independent modules and no edges.

    Returns
    -------
    RxGraphStore
        Graph store containing unconnected module nodes.
    """
    g = RxGraphStore.directed()
    for node in ["module_a", "module_b", "module_c"]:
        g.ensure_node(node)
    return g


def two_sccs_graph() -> RxGraphStore:
    """Create a graph with two strongly connected components connected by an edge.

    Returns
    -------
    RxGraphStore
        Graph store composed of two SCCs linked in sequence.
    """
    g = RxGraphStore.directed()
    _add_edges(
        g,
        [
            ("A", "B"),
            ("B", "A"),
            ("C", "D"),
            ("D", "C"),
            ("B", "C"),
        ],
    )
    return g


def complex_sccs_graph() -> RxGraphStore:
    """Create a graph with SCCs of sizes 1, 2, and 3 connected linearly.

    Returns
    -------
    RxGraphStore
        Graph store combining singleton, pair, and triple SCCs.
    """
    g = RxGraphStore.directed()
    g.ensure_node("A")
    _add_edges(g, [("B", "C"), ("C", "D"), ("D", "B")])
    _add_edges(g, [("E", "F"), ("F", "E")])
    g.add_weighted_edge("A", "B", weight=1.0)
    g.add_weighted_edge("D", "E", weight=1.0)
    return g


def tree_graph(depth: int = 3, branching: int = 2) -> RxGraphStore:
    """Create a balanced tree graph.

    Parameters
    ----------
    depth
        Tree depth (root is depth 0).
    branching
        Number of children per node.

    Returns
    -------
    RxGraphStore
        A tree graph store with specified depth and branching factor.

    Example
    -------
    >>> g = tree_graph(2, 2)
    >>> g.graph.num_nodes()
    7
    """
    g = RxGraphStore.directed()
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
            g.ensure_node(child)
            g.add_weighted_edge(parent, child, weight=1.0)
            add_children(child, current_depth + 1)

    root = "N0"
    g.ensure_node(root)
    add_children(root, 0)

    return g


def hub_and_spoke_graph(hubs: int = 2, spokes_per_hub: int = 3) -> RxGraphStore:
    """Create a graph with multiple hubs connected to their own spokes.

    Parameters
    ----------
    hubs
        Number of hub nodes.
    spokes_per_hub
        Number of spokes connected to each hub.

    Returns
    -------
    RxGraphStore
        A graph store with multiple independent hub-spoke clusters.

    Example
    -------
    >>> g = hub_and_spoke_graph(2, 3)
    >>> g.graph.num_nodes()
    8
    """
    g = RxGraphStore.directed()

    for h in range(hubs):
        hub = f"hub{h}"
        g.ensure_node(hub)

        for s in range(spokes_per_hub):
            spoke = f"h{h}_spoke{s}"
            g.ensure_node(spoke)
            g.add_weighted_edge(hub, spoke, weight=1.0)

    return g


def layered_graph(layers: tuple[int, ...] = (2, 3, 2)) -> RxGraphStore:
    """Create a layered DAG with edges between adjacent layers.

    Parameters
    ----------
    layers
        Tuple specifying number of nodes per layer.

    Returns
    -------
    RxGraphStore
        A layered directed acyclic graph store.

    Example
    -------
    >>> g = layered_graph((2, 3, 2))
    >>> g.graph.num_nodes()
    7
    """
    g = RxGraphStore.directed()
    prev_layer_nodes: list[str] = []

    for layer_idx, size in enumerate(layers):
        current_nodes = [f"L{layer_idx}N{i}" for i in range(size)]
        for node in current_nodes:
            g.ensure_node(node)

        for prev in prev_layer_nodes:
            for curr in current_nodes:
                g.add_weighted_edge(prev, curr, weight=1.0)

        prev_layer_nodes = current_nodes

    return g


def layered_dag_graph(
    layers: tuple[int, ...] = (2, 3, 2), *, cross_layer_edges: bool = False
) -> RxGraphStore:
    """Create a layered DAG with optional skip-level edges.

    Parameters
    ----------
    layers
        Tuple specifying number of nodes per layer.
    cross_layer_edges
        When True, connect each layer to the next layer and the following layer
        to model wider fan-out without manual edge wiring.

    Returns
    -------
    RxGraphStore
        Directed acyclic graph store with layered structure.
    """
    g = layered_graph(layers)
    if not cross_layer_edges or len(layers) < _MIN_LAYER_SPAN:
        return g

    nodes_by_layer: list[list[str]] = []
    for layer_idx, size in enumerate(layers):
        layer_nodes = [f"L{layer_idx}N{i}" for i in range(size)]
        nodes_by_layer.append(layer_nodes)

    for layer_idx in range(len(nodes_by_layer) - 2):
        current_layer = nodes_by_layer[layer_idx]
        skip_layer = nodes_by_layer[layer_idx + 2]
        for src in current_layer:
            for dst in skip_layer:
                g.add_weighted_edge(src, dst, weight=1.0)

    return g


def bridged_cliques_graph(
    clique1_size: int = 3,
    clique2_size: int = 3,
    *,
    bridge_from: str = "a0",
    bridge_to: str = "b0",
) -> RxGraphStore:
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
    RxGraphStore
        Undirected graph store with two complete subgraphs joined by one edge.
    """
    g = RxGraphStore.undirected()
    clique1_nodes = [f"a{i}" for i in range(clique1_size)]
    clique2_nodes = [f"b{i}" for i in range(clique2_size)]

    for node in clique1_nodes + clique2_nodes:
        g.ensure_node(node)

    for i, src in enumerate(clique1_nodes):
        for dst in clique1_nodes[i + 1 :]:
            g.add_weighted_edge(src, dst, weight=1.0)
    for i, src in enumerate(clique2_nodes):
        for dst in clique2_nodes[i + 1 :]:
            g.add_weighted_edge(src, dst, weight=1.0)

    g.add_weighted_edge(bridge_from, bridge_to, weight=1.0)
    return g


def bridge_chain_graph(
    segments: int = 3,
    *,
    segment_size: int = 3,
    prefix: str = "seg",
) -> RxGraphStore:
    """Create a chain of clique segments connected by single bridge edges.

    Each segment is a clique of ``segment_size`` nodes (``seg0_0``, ``seg0_1``, ...).
    The last node of each segment connects to the first node of the next,
    producing articulation edges useful for resilience/bridge testing.

    Parameters
    ----------
    segments
        Number of clique segments in the chain.
    segment_size
        Number of nodes inside each clique segment.
    prefix
        Prefix used when naming nodes.

    Returns
    -------
    RxGraphStore
        Graph store of clique segments joined by bridge edges.
    """
    g = RxGraphStore.undirected()
    if segments <= 0 or segment_size <= 0:
        return g

    previous_nodes: list[str] | None = None
    for segment_index in range(segments):
        nodes = [f"{prefix}{segment_index}_{i}" for i in range(segment_size)]
        for node in nodes:
            g.ensure_node(node)
        for i, src in enumerate(nodes):
            for dst in nodes[i + 1 :]:
                g.add_weighted_edge(src, dst, weight=1.0)
        if previous_nodes:
            g.add_weighted_edge(previous_nodes[-1], nodes[0], weight=1.0)
        previous_nodes = nodes

    return g


def fan_in_fan_out_graph(
    sources: tuple[str, ...] = ("in1", "in2"),
    sinks: tuple[str, ...] = ("out1", "out2"),
    *,
    center: str = "core",
) -> RxGraphStore:
    """Create a directed graph with multiple inputs converging and diverging.

    Parameters
    ----------
    sources
        Nodes that feed into the central node.
    sinks
        Nodes that receive edges from the central node.
    center
        Name of the central fan-in/fan-out node.

    Returns
    -------
    RxGraphStore
        Directed graph store modelling fan-in and fan-out structure.
    """
    g = RxGraphStore.directed()
    g.ensure_node(center)
    for src in sources:
        g.ensure_node(src)
    for dst in sinks:
        g.ensure_node(dst)

    for src in sources:
        g.add_weighted_edge(src, center, weight=1.0)
    for dst in sinks:
        g.add_weighted_edge(center, dst, weight=1.0)

    return g


def self_loop_graph(node: str = "A") -> RxGraphStore:
    """Create a graph containing a single self-loop.

    Returns
    -------
    RxGraphStore
        Directed graph store with one node that has a self-loop.
    """
    g = RxGraphStore.directed()
    g.add_weighted_edge(node, node, weight=1.0)
    return g


def nested_loop_graph() -> RxGraphStore:
    """Create a graph with nested loops (outer and inner).

    Returns
    -------
    RxGraphStore
        Directed graph store containing outer and inner cycles.
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("A", "B"), ("B", "C"), ("C", "A")])
    _add_edges(g, [("B", "D"), ("D", "B")])
    return g


def fork_join_cfg(
    *,
    entry: str = "entry",
    branch: str = "branch",
    left: str = "left",
    right: str = "right",
    join: str = "join",
) -> RxGraphStore:
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
    RxGraphStore
        Directed graph store with a fork and join.
    """
    g = RxGraphStore.directed()
    _add_edges(
        g,
        [
            (entry, branch),
            (branch, left),
            (branch, right),
            (left, join),
            (right, join),
        ],
    )
    return g


def while_loop_cfg(
    *,
    entry: str = "entry",
    condition: str = "condition",
    body: str = "body",
    exit_node: str = "exit",
) -> RxGraphStore:
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
    RxGraphStore
        Directed graph store modeling a while loop with an exit edge.
    """
    g = RxGraphStore.directed()
    _add_edges(
        g,
        [
            (entry, condition),
            (condition, body),
            (condition, exit_node),
            (body, condition),
        ],
    )
    return g


def scc_with_tail_graph() -> RxGraphStore:
    """Create a graph with a single SCC feeding a DAG tail.

    Returns
    -------
    RxGraphStore
        Graph store containing an SCC connected to a linear tail.
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("A", "B"), ("B", "C"), ("C", "A")])
    _add_edges(g, [("C", "D"), ("D", "E")])
    return g


def two_cycle_graph() -> RxGraphStore:
    """Create a graph with two disjoint 2-cycles.

    Returns
    -------
    RxGraphStore
        Directed graph store containing two separate 2-cycles.
    """
    g = RxGraphStore.directed()
    _add_edges(g, [("A", "B"), ("B", "A")])
    _add_edges(g, [("C", "D"), ("D", "C")])
    return g


def dag_to_cycle_graph(
    *,
    entry: str = "entry",
    exit_node: str = "exit",
    cycle_nodes: tuple[str, str, str] = ("A", "B", "C"),
) -> RxGraphStore:
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
    RxGraphStore
        Graph store combining a line into a cycle with an exit edge.
    """
    n1, n2, n3 = cycle_nodes
    g = RxGraphStore.directed()
    _add_edges(
        g,
        [
            (entry, n1),
            (n1, n2),
            (n2, n3),
            (n3, n1),
            (entry, exit_node),
            (n3, exit_node),
        ],
    )
    return g


def shared_neighbors_graph(
    shared: int,
    *,
    primary: tuple[str, str] = ("p1", "p2"),
    unique_first: int = 0,
    unique_second: int = 0,
) -> RxGraphStore:
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
    RxGraphStore
        Undirected graph store containing the specified shared and unique neighbors.
    """
    first, second = primary
    g = RxGraphStore.undirected()
    for node in primary:
        g.ensure_node(node)
    shared_nodes = [f"s{i}" for i in range(shared)]
    first_uniques = [f"{first}_u{i}" for i in range(unique_first)]
    second_uniques = [f"{second}_u{i}" for i in range(unique_second)]

    for node in shared_nodes:
        g.add_weighted_edge(first, node, weight=1.0)
        g.add_weighted_edge(second, node, weight=1.0)
    for node in first_uniques:
        g.add_weighted_edge(first, node, weight=1.0)
    for node in second_uniques:
        g.add_weighted_edge(second, node, weight=1.0)
    return g


__all__ = [
    "DEFAULT_CHAIN_LENGTH",
    "DEFAULT_COMPLETE_SIZE",
    "DEFAULT_CYCLE_SIZE",
    "DEFAULT_SPOKES",
    "GOLDEN_CALL",
    "GOLDEN_IMPORT",
    "STANDARD_CALL",
    "STANDARD_IMPORT",
    "GraphFixtureFactory",
    "GraphFixtureSpec",
    "GraphFixtures",
    "GraphKind",
    "acyclic_bipartite_flow",
    "barbell_graph_small",
    "bidirectional_deps_graph",
    "bipartite_graph",
    "bridge_chain_graph",
    "bridged_cliques_graph",
    "build_sample_graphs",
    "call_chain_graph",
    "call_graph_fixture",
    "call_star_graph",
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
    "fan_in_fan_out_graph",
    "fork_join_cfg",
    "god_module_graph",
    "hub_and_spoke_graph",
    "hub_dependencies_graph",
    "import_cycle_graph",
    "independent_modules_graph",
    "layered_dag_graph",
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
    "standard_graph_fixtures",
    "star_graph",
    "symbol_star_graph",
    "tree_graph",
    "two_cycle_graph",
    "two_sccs_graph",
    "weighted_star_graph",
    "while_loop_cfg",
]
