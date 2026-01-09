"""Pure control flow graph metric computation functions.

This module provides stateless functions for computing CFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

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
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    dag_longest_path_length,
    descendants_by_id,
    dominance_frontiers_by_id,
    ensure_directed_store,
    immediate_dominators_by_id,
    insert_node_on_out_edges_by_id,
    is_directed_acyclic,
    remove_node_retain_edges_by_id,
)
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.iterators import iter_edge_id_payloads, iter_edge_id_weights
from codeintel.build.graphs.rx.metadata import apply_graph_metadata, metadata_from_graph
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, weight_policy_for_kind
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from codeintel.build.graphs.runtime.context import GraphContext

NodeT = TypeVar("NodeT", bound=Hashable)


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
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return immediate_dominators_by_id(store, entry)


def _clone_cfg_store(store: RxGraphStore) -> RxGraphStore:
    edge_rows = [
        (src_id, dst_id, weight) for src_id, dst_id, weight in iter_edge_id_weights(store)
    ]
    node_attrs = {node_id: dict(attrs) for node_id, attrs in store.node_attrs.items()}
    spec = EdgeBuildSpec(
        directed=True,
        weight_policy=store.weight_policy,
        numeric_policy=store.numeric_policy,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=True,
        node_ids=store.node_ids(),
        node_attrs=node_attrs or None,
        node_hint=len(store.id_to_index),
        edge_hint=len(edge_rows),
    )
    cloned = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
    metadata = metadata_from_graph(store.graph)
    if metadata is not None:
        apply_graph_metadata(cloned.graph, metadata)
    return cloned


def _as_int_id(value: Hashable) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _next_synthetic_id(store: RxGraphStore) -> int:
    numeric_ids = [_as_int_id(node_id) for node_id in store.node_ids()]
    resolved = [node_id for node_id in numeric_ids if node_id is not None]
    candidate = (min(resolved) - 1) if resolved else -1
    while candidate in store.id_to_index:
        candidate -= 1
    return candidate


def _insert_entry_fanout(store: RxGraphStore, entry_id: Hashable) -> int | None:
    entry_idx = store.id_to_index.get(entry_id)
    if entry_idx is None:
        return None
    directed_graph = cast("rx.PyDiGraph", store.graph)
    if directed_graph.out_degree(entry_idx) <= 1:
        return None
    synthetic_id = _next_synthetic_id(store)
    return insert_node_on_out_edges_by_id(
        store,
        synthetic_id,
        entry_id,
        attrs={"kind": "entry_fanout", "synthetic": True},
    )


def _prune_isolated_nodes(store: RxGraphStore, *, protected: set[Hashable]) -> None:
    directed_graph = cast("rx.PyDiGraph", store.graph)
    for node_id in store.node_ids():
        if node_id in protected:
            continue
        node_idx = store.id_to_index.get(node_id)
        if node_idx is None:
            continue
        if directed_graph.in_degree(node_idx) == 0 and directed_graph.out_degree(node_idx) == 0:
            remove_node_retain_edges_by_id(store, node_id)


def normalize_cfg_graph(
    graph: GraphInput,
    *,
    entry_idx: Hashable,
    exit_idx: Hashable,
) -> RxGraphStore:
    """Normalize a CFG for analysis using rustworkx mutation helpers.

    Returns
    -------
    RxGraphStore
        Normalized CFG graph for analysis metrics.
    """
    store = ensure_directed_store(graph)
    normalized = _clone_cfg_store(store)
    synthetic_id = _insert_entry_fanout(normalized, entry_idx)
    protected: set[Hashable] = {entry_idx, exit_idx}
    if synthetic_id is not None:
        protected.add(synthetic_id)
    _prune_isolated_nodes(normalized, protected=protected)
    return normalized


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
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return dominance_frontiers_by_id(store, entry)


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
    store = ensure_directed_store(graph)
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
    for src_id, dst_id, _payload in iter_edge_id_payloads(store):
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
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0
    return dag_longest_path_length(store)


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
    store = ensure_directed_store(graph)
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
    store = ensure_directed_store(graph)
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
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0
    entry_node = store.id_to_index.get(entry_idx)
    if entry_node is None:
        return 0

    if is_dag is None:
        is_dag = is_directed_acyclic(store)

    if not is_dag:
        return compute_cfg_longest_path(store)

    reachable_ids = descendants_by_id(store, entry_idx, include_source=True)
    ordered = sorted(
        (store.id_to_index[node_id] for node_id in reachable_ids if node_id in store.id_to_index),
        key=lambda idx: stable_key(store.index_to_id[idx]),
    )
    subgraph, _node_map = store.graph.subgraph_with_nodemap(ordered, preserve_attrs=True)
    return dag_longest_path_length(subgraph, allow_condensation=False)


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
    entry_idx: int | None = None
    exit_idx: int | None = None
    out_deg_map: dict[int, int] = {}
    node_attrs: dict[Hashable, dict[str, object]] = {}
    node_ids: list[int] = []
    for idx, kind, in_deg, out_deg in blocks:
        node_ids.append(idx)
        node_attrs[idx] = {"kind": kind, "in_degree": in_deg, "out_degree": out_deg}
        if kind == "entry":
            entry_idx = idx
        if kind == "exit":
            exit_idx = idx
        out_deg_map[idx] = out_deg
    edge_rows = [(src, dst, 1.0) for src, dst, _edge_type in edges]
    spec = EdgeBuildSpec(
        directed=True,
        weight_policy=weight_policy_for_kind(GraphKind.CFG_GRAPH),
        numeric_policy=DEFAULT_NUMERIC_POLICY,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=True,
        node_ids=node_ids or None,
        node_attrs=node_attrs or None,
        node_hint=len(node_ids) if node_ids else None,
        edge_hint=len(edge_rows),
    )
    graph = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
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
    "normalize_cfg_graph",
]
