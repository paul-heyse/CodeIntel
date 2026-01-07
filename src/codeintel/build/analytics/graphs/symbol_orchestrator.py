"""Orchestrator for undirected symbol graph metrics computation.

This module provides a generic orchestrator pattern for computing
symbol coupling metrics on undirected graphs, reducing code duplication
between module-level and function-level metrics computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.graphs import (
    centrality_undirected,
    component_ids_undirected,
    log_empty_graph,
    structural_metrics,
)
from codeintel.build.analytics.compute.row_builders import RowBuildContext, SymbolMetricInputs
from codeintel.build.analytics.graphs.constants import MAX_BETWEENNESS_NODES, MAX_COMMUNITY_NODES
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class UndirectedMetricsConfig[TNode]:
    """Configuration for undirected symbol metrics computation.

    Type Parameters
    ---------------
    TNode
        Node type for the graph (str for modules, int for functions).

    Attributes
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.symbol_graph_metrics_modules").
    graph_name
        Name for logging empty graph warnings.
    filter_node
        Function to check if a node should be included.
    build_rows
        Function to build rows from metric inputs.
    """

    table_key: str
    graph_name: str
    filter_node: Callable[[object, set[TNode]], bool]
    build_rows: Callable[[SymbolMetricInputs[TNode]], Sequence[tuple[object, ...]]]


@dataclass(frozen=True)
class UndirectedMetricInputs[TNode]:
    """Inputs for undirected symbol metrics computation."""

    repo: str
    commit: str
    graph: GraphInput
    known_nodes: set[TNode] | None = None
    runtime: GraphRuntimeOptions | None = None


def _filter_nodes(graph: GraphInput, allowed: set[object]) -> RxGraphStore:
    store = ensure_store(graph)
    if store.is_directed:
        filtered = RxGraphStore.directed(
            node_hint=store.graph.num_nodes(),
            edge_hint=store.graph.num_edges(),
        )
    else:
        filtered = RxGraphStore.undirected(
            node_hint=store.graph.num_nodes(),
            edge_hint=store.graph.num_edges(),
        )
    for node_id in store.node_ids():
        if node_id in allowed:
            filtered.set_node_attrs(node_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        if src_id not in allowed or dst_id not in allowed:
            continue
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        filtered.add_weighted_edge(src_id, dst_id, weight=weight)
    return filtered


def build_undirected_symbol_metric_rows[TNode](
    *,
    inputs: UndirectedMetricInputs[TNode],
    config: UndirectedMetricsConfig[TNode],
) -> list[tuple[object, ...]]:
    """Compute undirected symbol graph metrics rows.

    This orchestrator handles the common pattern for computing symbol coupling
    metrics on undirected graphs, including centrality, structural metrics,
    and component information.

    Parameters
    ----------
    inputs
        Metric inputs including repo/commit identifiers, graph, and runtime options.
    config
        Configuration specifying table name, node filtering, and row building.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for the configured analytics table.
    """
    runtime_opts = inputs.runtime or GraphRuntimeOptions()
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=inputs.repo,
            commit=inputs.commit,
            use_gpu=runtime_opts.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
            community_detection_limit=runtime_opts.features.community_detection_limit,
        )
    )

    graph: GraphInput = inputs.graph
    if inputs.known_nodes is not None:
        store = ensure_store(graph)
        allowed = {
            node for node in store.node_ids() if config.filter_node(node, inputs.known_nodes)
        }
        graph = _filter_nodes(graph, allowed)

    if graph_node_count(graph) == 0:
        log_empty_graph(config.graph_name, graph)
        return []

    centrality = centrality_undirected(graph, ctx)
    structure = structural_metrics(
        graph,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
    comp_id, comp_size = component_ids_undirected(graph)
    node_count = graph_node_count(graph)

    row_context = RowBuildContext.from_repo_commit(
        inputs.repo,
        inputs.commit,
        created_at=ctx.resolved_now(),
    )
    metric_inputs = SymbolMetricInputs[TNode](
        row_context=row_context,
        centrality={
            "betweenness": centrality.betweenness,
            "closeness": centrality.closeness,
            "eigenvector": centrality.eigenvector,
            "harmonic": centrality.harmonic,
        },
        structure={
            "core_number": structure.core_number,
            "constraint": structure.constraint,
            "effective_size": structure.effective_size,
            "community_id": (structure.community_id if node_count <= MAX_COMMUNITY_NODES else {}),
        },
        comp_id=comp_id,
        comp_size=comp_size,
    )
    return list(config.build_rows(metric_inputs))


__all__ = [
    "UndirectedMetricInputs",
    "UndirectedMetricsConfig",
    "build_undirected_symbol_metric_rows",
]
