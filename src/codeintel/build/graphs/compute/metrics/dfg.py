"""Pure data flow graph metric computation functions.

This module provides stateless functions for computing DFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.graphs.compute.metrics.centrality import centrality_directed
from codeintel.build.graphs.compute.metrics.components import (
    find_strongly_connected,
    find_weakly_connected,
)
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    bfs_distances_by_id,
    digraph_all_pairs_shortest_path_lengths_by_id,
    ensure_directed_store,
    in_degree_by_id,
    insert_node_on_out_edges_by_id,
    out_degree_by_id,
    predecessors_by_id,
    remove_node_retain_edges_by_id,
    simple_cycles_by_id,
    successors_by_id,
)
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.iterators import iter_edge_id_weights
from codeintel.build.graphs.rx.metadata import apply_graph_metadata, metadata_from_graph
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, DEFAULT_WEIGHT_POLICY
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from codeintel.build.graphs.compute.metrics.components import (
        ComponentInfo,
    )
    from codeintel.build.graphs.runtime.context import GraphContext


@dataclass(frozen=True)
class DFGPathStats:
    """Path length statistics for a DFG node.

    Attributes
    ----------
    max_def_use_distance
        Maximum distance from definition to use.
    avg_def_use_distance
        Average distance from definition to uses.
    reach_count
        Number of nodes reachable from this node.
    """

    max_def_use_distance: int
    avg_def_use_distance: float
    reach_count: int


def compute_dfg_path_lengths(
    graph: GraphInput,
    *,
    max_depth: int = 100,
) -> dict[Any, DFGPathStats]:
    """Compute path length statistics for DFG nodes.

    Parameters
    ----------
    graph
        Data flow graph (directed).
    max_depth
        Maximum search depth (for performance).

    Returns
    -------
    dict[Any, DFGPathStats]
        Node to path statistics mapping.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}

    result: dict[Any, DFGPathStats] = {}
    for node_id in store.node_ids():
        distances = bfs_distances_by_id(store, node_id, max_depth=max_depth + 1)
        limit = max_depth + 1
        bounded = [
            distance
            for node, distance in distances.items()
            if node != node_id and 0 < distance <= limit
        ]
        if bounded:
            result[node_id] = DFGPathStats(
                max_def_use_distance=max(bounded),
                avg_def_use_distance=sum(bounded) / len(bounded),
                reach_count=len(bounded),
            )
            continue
        result[node_id] = DFGPathStats(
            max_def_use_distance=0,
            avg_def_use_distance=0.0,
            reach_count=0,
        )
    return result


def _clone_dfg_store(store: RxGraphStore) -> RxGraphStore:
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


def _as_int_id(value: object) -> int | None:
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


def _insert_phi_fanout(store: RxGraphStore, node_id: int) -> int | None:
    if node_id not in store.id_to_index:
        return None
    out_degrees = out_degree_by_id(store)
    if out_degrees.get(node_id, 0) <= 1:
        return None
    synthetic_id = _next_synthetic_id(store)
    return insert_node_on_out_edges_by_id(
        store,
        synthetic_id,
        node_id,
        attrs={"kind": "phi_fanout", "synthetic": True},
    )


def _prune_isolated_nodes(store: RxGraphStore) -> None:
    in_degrees = in_degree_by_id(store)
    out_degrees = out_degree_by_id(store)
    for node_id in store.node_ids():
        if in_degrees.get(node_id, 0) == 0 and out_degrees.get(node_id, 0) == 0:
            remove_node_retain_edges_by_id(store, node_id)


def normalize_dfg_graph(
    graph: GraphInput,
    edges: list[tuple[int, int, str, str, bool, str]],
) -> RxGraphStore:
    """Normalize a DFG for analysis using rustworkx mutation helpers.

    Returns
    -------
    RxGraphStore
        Normalized DFG graph for analysis metrics.
    """
    store = ensure_directed_store(graph)
    normalized = _clone_dfg_store(store)
    totals: dict[int, tuple[int, int]] = {}
    for src, _dst, _src_sym, _dst_sym, via_phi, _use_kind in edges:
        total, phi = totals.get(src, (0, 0))
        totals[src] = (total + 1, phi + (1 if via_phi else 0))
    for src_id, (total, phi) in sorted(totals.items()):
        if total > 1 and total == phi:
            _insert_phi_fanout(normalized, src_id)
    _prune_isolated_nodes(normalized)
    return normalized


def compute_dfg_components(
    graph: GraphInput,
) -> tuple[list[ComponentInfo], list[ComponentInfo]]:
    """Compute connected components for a DFG.

    Returns both strongly and weakly connected components.

    Parameters
    ----------
    graph
        Data flow graph (directed).

    Returns
    -------
    tuple[list[ComponentInfo], list[ComponentInfo]]
        (strongly_connected, weakly_connected) component lists.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return ([], [])

    scc_result = find_strongly_connected(graph)
    wccs = find_weakly_connected(graph)

    return (list(scc_result.components), wccs)


def compute_def_use_chains(
    graph: GraphInput,
) -> dict[Any, list[Any]]:
    """Compute def-use chains for each node.

    A def-use chain is the list of nodes that use a definition.

    Parameters
    ----------
    graph
        Data flow graph where edges represent def-use relationships.

    Returns
    -------
    dict[Any, list[Any]]
        Node to list of users mapping.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    result: dict[Any, list[Any]] = {}
    for node_id in store.node_ids():
        result[node_id] = successors_by_id(store, node_id)
    return result


def compute_use_def_chains(
    graph: GraphInput,
) -> dict[Any, list[Any]]:
    """Compute use-def chains for each node.

    A use-def chain is the list of definitions that reach a use.

    Parameters
    ----------
    graph
        Data flow graph where edges represent def-use relationships.

    Returns
    -------
    dict[Any, list[Any]]
        Node to list of definitions mapping.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    result: dict[Any, list[Any]] = {}
    for node_id in store.node_ids():
        result[node_id] = predecessors_by_id(store, node_id)
    return result


def compute_dfg_density(graph: GraphInput) -> float:
    """Compute edge density of a DFG.

    Parameters
    ----------
    graph
        Data flow graph.

    Returns
    -------
    float
        Edge density (0.0 to 1.0).
    """
    store = ensure_directed_store(graph)
    node_count = store.graph.num_nodes()
    if node_count <= 1:
        return 0.0
    max_edges = node_count * (node_count - 1)
    return store.graph.num_edges() / max_edges


def find_dfg_cycles(
    graph: GraphInput,
    *,
    limit: int = 100,
) -> list[list[Any]]:
    """Find cycles in a DFG (may indicate recursive data flow).

    Parameters
    ----------
    graph
        Data flow graph.
    limit
        Maximum number of cycles to return.

    Returns
    -------
    list[list[Any]]
        List of cycles as node sequences.
    """
    return simple_cycles_by_id(graph, limit=limit)


def dfg_component_stats(graph: GraphInput) -> tuple[int, list[set[Any]], bool]:
    """Return connected component stats for DFG graphs.

    Returns
    -------
    tuple[int, list[set[int]], bool]
        Component count, components, and whether cycles are present.
    """
    sccs, wccs = compute_dfg_components(graph)
    components: list[set[Any]] = [set(wcc.nodes) for wcc in wccs]
    has_cycles = any(scc.size > 1 for scc in sccs)
    return len(components), components, has_cycles


def dfg_path_lengths(graph: GraphInput) -> tuple[int, float]:
    """Return longest path length and average shortest path length for DFGs.

    Returns
    -------
    tuple[int, float]
        Longest path length and average shortest path length.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0, 0.0
    lengths = digraph_all_pairs_shortest_path_lengths_by_id(store)
    longest = 0.0
    total = 0.0
    count = 0
    for targets in lengths.values():
        if targets:
            longest = max(longest, float(max(targets.values(), default=0)))
            total += sum(targets.values())
            count += len(targets)
    avg = total / count if count else 0.0
    return int(longest), avg


def dfg_centralities(
    graph: GraphInput, ctx: GraphContext
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Compute DFG betweenness and eigenvector centralities.

    Returns
    -------
    tuple[dict[Any, float], dict[Any, float]]
        Betweenness and eigenvector centrality mappings.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}, {}
    centrality = centrality_directed(
        store,
        ctx,
        weight=None,
        include_eigen=True,
    )
    return centrality.betweenness, centrality.eigenvector


def build_dfg_graph(
    edges: list[tuple[int, int, str, str, bool, str]],
) -> tuple[RxGraphStore, int, int]:
    """Build a data-flow graph from edge tuples.

    Returns
    -------
    tuple[RxGraphStore, int, int]
        Graph, phi edge count, and symbol count.
    """
    phi_edges = 0
    symbols: set[str] = set()
    node_ids: set[int] = set()
    edge_rows: list[tuple[int, int, float]] = []
    for src, dst, src_sym, dst_sym, via_phi, _use_kind in edges:
        edge_rows.append((src, dst, 1.0))
        node_ids.update((src, dst))
        symbols.add(src_sym)
        symbols.add(dst_sym)
        if via_phi:
            phi_edges += 1
    spec = EdgeBuildSpec(
        directed=True,
        weight_policy=DEFAULT_WEIGHT_POLICY,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        aggregate_edges=True,
        node_ids=node_ids or None,
        node_hint=len(node_ids) if node_ids else None,
        edge_hint=len(edge_rows),
    )
    graph = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
    return graph, phi_edges, len(symbols)


__all__ = [
    "DFGPathStats",
    "build_dfg_graph",
    "compute_def_use_chains",
    "compute_dfg_components",
    "compute_dfg_density",
    "compute_dfg_path_lengths",
    "compute_use_def_chains",
    "dfg_centralities",
    "dfg_component_stats",
    "dfg_path_lengths",
    "find_dfg_cycles",
    "normalize_dfg_graph",
]
