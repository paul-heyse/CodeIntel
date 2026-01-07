"""Pure control flow graph metric computation functions.

This module provides stateless functions for computing CFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import rustworkx as rx

from codeintel.build.graphs.compute.metrics.centrality import centrality_directed
from codeintel.build.graphs.compute.metrics.paths import (
    compute_avg_shortest_path_from_source,
    compute_reachable_nodes,
)
from codeintel.build.graphs.compute.metrics.types import (
    CentralityBundle,
)
from codeintel.build.graphs.compute.metrics.types import (
    DominanceMetrics as DominanceSummary,
)
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, stable_key
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from codeintel.build.graphs.runtime.context import GraphContext

NodeT = TypeVar("NodeT", bound=Hashable)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DominanceMetrics:
    """Dominance metrics for control flow graphs.

    Attributes
    ----------
    depth
        Depth in dominator tree (root = 0).
    frontier_size
        Size of dominance frontier.
    is_loop_header
        Whether node is a natural loop header.
    """

    depth: int
    frontier_size: int
    is_loop_header: bool


def _ensure_directed_store(graph: GraphInput) -> RxGraphStore:
    store = ensure_store(graph)
    if store.is_directed:
        return store
    directed = RxGraphStore.directed(
        node_hint=store.graph.num_nodes(),
        edge_hint=store.graph.num_edges() * 2,
    )
    for node_id in store.node_ids():
        directed.set_node_attrs(node_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        directed.add_weighted_edge(src_id, dst_id, weight=weight)
        if src_id != dst_id:
            directed.add_weighted_edge(dst_id, src_id, weight=weight)
    return directed


def _directed_graph(store: RxGraphStore) -> rx.PyDiGraph:
    if not store.is_directed:
        message = "Expected a directed graph store"
        raise ValueError(message)
    return cast("rx.PyDiGraph", store.graph)


def _component_sort_key(store: RxGraphStore, component: set[int]) -> tuple[str, str]:
    if not component:
        return ("", "")
    smallest = min((store.index_to_id[idx] for idx in component), key=stable_key)
    return stable_key(smallest)


def _sorted_components(store: RxGraphStore, components: list[set[int]]) -> list[set[int]]:
    return sorted(components, key=lambda comp: _component_sort_key(store, comp))


def _condensation_store(store: RxGraphStore) -> RxGraphStore:
    directed_graph = _directed_graph(store)
    components = [set(comp) for comp in rx.strongly_connected_components(directed_graph)]
    if not components:
        return RxGraphStore.directed()
    sorted_components = _sorted_components(store, components)
    index_to_component: dict[int, int] = {}
    for comp_id, comp in enumerate(sorted_components):
        for node_idx in comp:
            index_to_component[node_idx] = comp_id
    condensed = RxGraphStore.directed(node_hint=len(sorted_components))
    for comp_id in range(len(sorted_components)):
        condensed.ensure_node(comp_id)
    for src_idx, dst_idx in store.graph.edge_list():
        src_comp = index_to_component.get(src_idx)
        dst_comp = index_to_component.get(dst_idx)
        if src_comp is None or dst_comp is None or src_comp == dst_comp:
            continue
        condensed.add_weighted_edge(src_comp, dst_comp, weight=1.0)
    return condensed


def compute_dominator_tree(
    graph: GraphInput,
    entry: Hashable,
) -> dict[Hashable, Hashable | None]:
    """Compute immediate dominators for all nodes.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry
        Entry node (typically function start).

    Returns
    -------
    dict[Hashable, Hashable | None]
        Node to immediate dominator mapping. Entry node maps to None.
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    entry_idx = store.id_to_index.get(entry)
    if entry_idx is None:
        return {}
    directed_graph = _directed_graph(store)
    try:
        idoms = rx.immediate_dominators(directed_graph, entry_idx)
    except (rx.InvalidNode, rx.NullGraph) as exc:
        log.warning("Dominator computation failed: %s", exc)
        return {}

    result: dict[Hashable, Hashable | None] = {}
    for node_idx, idom_idx in idoms.items():
        node_id = store.index_to_id[node_idx]
        if node_idx == entry_idx:
            result[node_id] = None
        else:
            result[node_id] = store.index_to_id[idom_idx]
    return {node: result[node] for node in sorted(result, key=stable_key)}


def compute_dominance_frontier(
    graph: GraphInput,
    entry: Hashable,
) -> dict[Hashable, frozenset[Hashable]]:
    """Compute dominance frontier for all nodes.

    The dominance frontier of node n is the set of nodes where n's
    dominance ends.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry
        Entry node.

    Returns
    -------
    dict[Hashable, frozenset[Hashable]]
        Node to dominance frontier mapping.
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    entry_idx = store.id_to_index.get(entry)
    if entry_idx is None:
        return {}
    directed_graph = _directed_graph(store)
    try:
        frontiers = rx.dominance_frontiers(directed_graph, entry_idx)
    except (rx.InvalidNode, rx.NullGraph) as exc:
        log.warning("Dominance frontier computation failed: %s", exc)
        return {}
    mapped = {
        store.index_to_id[node_idx]: frozenset(store.index_to_id[idx] for idx in frontier)
        for node_idx, frontier in frontiers.items()
    }
    return {node: mapped[node] for node in sorted(mapped, key=stable_key)}


def compute_dominator_depths(
    idoms: dict[NodeT, NodeT | None],
) -> dict[NodeT, int]:
    """Compute depth in dominator tree for all nodes.

    Parameters
    ----------
    idoms
        Immediate dominator mapping from compute_dominator_tree.

    Returns
    -------
    dict[NodeT, int]
        Node to depth mapping (root = 0).
    """
    if not idoms:
        return {}

    depths: dict[NodeT, int] = {}

    def get_depth(node: NodeT) -> int:
        if node in depths:
            return depths[node]
        idom = idoms.get(node)
        if idom is None:
            depths[node] = 0
        else:
            depths[node] = get_depth(idom) + 1
        return depths[node]

    for node in idoms:
        get_depth(node)

    return depths


def find_natural_loop_headers(
    graph: GraphInput,
    entry: Hashable,
) -> set[Hashable]:
    """Find natural loop headers in a control flow graph.

    A natural loop header is a node that dominates a predecessor
    (forming a back edge).

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry
        Entry node.

    Returns
    -------
    set[Hashable]
        Set of loop header nodes.
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return set()
    if entry not in store.id_to_index:
        return set()
    idoms = compute_dominator_tree(store, entry)
    if not idoms:
        return set()

    dominates: dict[Hashable, set[Hashable]] = {node: set() for node in idoms}
    for node in idoms:
        current: Hashable | None = node
        while current is not None:
            dominates[current].add(node)
            current = idoms.get(current)
            if current == node:
                break

    headers: set[Hashable] = set()
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        if src_id in dominates.get(dst_id, set()):
            headers.add(dst_id)

    return headers


def compute_cfg_longest_path(
    graph: GraphInput,
) -> int:
    """Compute longest path length in a CFG.

    For cyclic graphs, this computes on the DAG after condensation.

    Parameters
    ----------
    graph
        Control flow graph (directed).

    Returns
    -------
    int
        Longest path length (number of edges).
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0

    directed_graph = _directed_graph(store)
    try:
        if rx.is_directed_acyclic_graph(directed_graph):
            return int(rx.dag_longest_path_length(directed_graph))
    except rx.NullGraph:
        return 0

    condensed = _condensation_store(store)
    if condensed.graph.num_nodes() == 0:
        return 0
    condensed_graph = cast("rx.PyDiGraph", condensed.graph)
    try:
        return int(rx.dag_longest_path_length(condensed_graph))
    except (rx.DAGHasCycle, rx.NullGraph):
        return 0


def compute_all_dominance(
    graph: GraphInput,
    entry: Hashable,
) -> dict[Hashable, DominanceMetrics]:
    """Compute all dominance-related metrics for CFG nodes.

    Parameters
    ----------
    graph
        Control flow graph.
    entry
        Entry node.

    Returns
    -------
    dict[Hashable, DominanceMetrics]
        Node to dominance metrics mapping.
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}

    idoms = compute_dominator_tree(store, entry)
    frontiers = compute_dominance_frontier(store, entry)
    depths = compute_dominator_depths(idoms)
    loop_headers = find_natural_loop_headers(store, entry)

    return {
        node: DominanceMetrics(
            depth=depths.get(node, 0),
            frontier_size=len(frontiers.get(node, frozenset())),
            is_loop_header=node in loop_headers,
        )
        for node in store.node_ids()
    }


def cfg_dominance_metrics(graph: GraphInput, entry_idx: int) -> DominanceSummary:
    """Compute dominator tree depth and frontier sizes for a CFG.

    Returns
    -------
    DominanceSummary
        Dominance depth and frontier sizes for CFG nodes.
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return DominanceSummary(depth={}, frontier_sizes={}, tree_height=None)

    idoms = compute_dominator_tree(store, entry_idx)
    dom_depth = compute_dominator_depths(idoms)
    frontiers = compute_dominance_frontier(store, entry_idx)
    frontier_sizes = {node: len(frontiers.get(node, frozenset())) for node in store.node_ids()}
    height = max(dom_depth.values()) if dom_depth else None

    return DominanceSummary(
        depth=dom_depth,
        frontier_sizes=frontier_sizes,
        tree_height=height,
    )


def cfg_centralities(
    graph: GraphInput,
    entry_idx: int,
    *,
    ctx: GraphContext,
) -> tuple[CentralityBundle, DominanceSummary]:
    """Compute CFG centralities and dominance metrics.

    Returns
    -------
    tuple[CentralityBundle, DominanceSummary]
        Centrality bundle and dominance summary for the CFG.
    """
    dominance = cfg_dominance_metrics(graph, entry_idx)
    centrality = centrality_directed(
        graph,
        ctx,
        weight=None,
        include_eigen=True,
    )
    return centrality, dominance


def cfg_longest_path_length(
    graph: GraphInput,
    entry_idx: int,
    *,
    is_dag: bool | None = None,
) -> int:
    """Compute the longest path length for a CFG.

    Returns
    -------
    int
        Longest path length from the entry node.
    """
    store = _ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0
    entry_node = store.id_to_index.get(entry_idx)
    if entry_node is None:
        return 0

    directed_graph = _directed_graph(store)
    if is_dag is None:
        try:
            is_dag = rx.is_directed_acyclic_graph(directed_graph)
        except rx.NullGraph:
            return 0

    if not is_dag:
        return compute_cfg_longest_path(store)

    try:
        reachable = set(rx.descendants(directed_graph, entry_node))
    except (rx.InvalidNode, rx.NullGraph):
        return 0
    reachable.add(entry_node)
    subgraph = directed_graph.subgraph(sorted(reachable))
    try:
        longest = int(rx.dag_longest_path_length(subgraph))
    except (rx.DAGHasCycle, rx.NullGraph):
        longest = 0
    return longest


def cfg_avg_shortest_path_length(graph: GraphInput, entry_idx: int) -> float:
    """Return the average shortest path length from the entry block.

    Returns
    -------
    float
        Average shortest path length from the entry node.
    """
    return compute_avg_shortest_path_from_source(graph, entry_idx)


def cfg_reachable_nodes(graph: GraphInput, entry_idx: int) -> set[Any]:
    """Return the set of nodes reachable from the entry node.

    Returns
    -------
    set[Any]
        Reachable nodes in the CFG.
    """
    return set(compute_reachable_nodes(graph, entry_idx))


def build_cfg_graph(
    blocks: list[tuple[int, str, int, int]],
    edges: list[tuple[int, int, str]],
) -> tuple[RxGraphStore, int, int]:
    """Build a control-flow graph from block and edge tuples.

    Returns
    -------
    tuple[RxGraphStore, int, int]
        Graph, entry node id, and exit node id.
    """
    graph = RxGraphStore.directed()
    entry_idx = None
    exit_idx = None
    out_deg_map: dict[int, int] = {}
    for idx, kind, in_deg, out_deg in blocks:
        graph.set_node_attrs(
            idx,
            {"kind": kind, "in_degree": in_deg, "out_degree": out_deg},
        )
        if kind == "entry":
            entry_idx = idx
        if kind == "exit":
            exit_idx = idx
        out_deg_map[idx] = out_deg
    for src, dst, _edge_type in edges:
        graph.add_weighted_edge(src, dst, weight=1.0)
    if entry_idx is None and graph.graph.num_nodes() > 0:
        entry_idx = min(int(str(node)) for node in graph.node_ids())
    if exit_idx is None:
        exits = [node for node, deg in out_deg_map.items() if deg == 0]
        exit_idx = exits[0] if exits else (entry_idx if entry_idx is not None else 0)
    return graph, entry_idx or 0, exit_idx or 0


__all__ = [
    "DominanceMetrics",
    "build_cfg_graph",
    "cfg_avg_shortest_path_length",
    "cfg_centralities",
    "cfg_dominance_metrics",
    "cfg_longest_path_length",
    "cfg_reachable_nodes",
    "compute_all_dominance",
    "compute_cfg_longest_path",
    "compute_dominance_frontier",
    "compute_dominator_depths",
    "compute_dominator_tree",
    "find_natural_loop_headers",
]
