"""Orchestrator for undirected symbol graph metrics computation.

This module provides a generic orchestrator pattern for computing
symbol coupling metrics on undirected graphs, reducing code duplication
between module-level and function-level metrics computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.build.analytics.compute.graphs import (
    centrality_undirected,
    component_ids_undirected,
    log_empty_graph,
    structural_metrics,
)
from codeintel.build.analytics.compute.row_builders import SymbolMetricInputs
from codeintel.build.analytics.graphs.constants import MAX_BETWEENNESS_NODES, MAX_COMMUNITY_NODES
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context

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
    graph: nx.Graph
    known_nodes: set[TNode] | None = None
    runtime: GraphRuntimeOptions | None = None


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

    graph = inputs.graph
    if inputs.known_nodes is not None:
        graph = graph.subgraph(
            [node for node in graph.nodes if config.filter_node(node, inputs.known_nodes)]
        ).copy()

    if graph.number_of_nodes() == 0:
        log_empty_graph(config.graph_name, graph)
        return []

    centrality = centrality_undirected(graph, ctx)
    structure = structural_metrics(
        graph,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
    comp_id, comp_size = component_ids_undirected(graph)

    metric_inputs = SymbolMetricInputs[TNode](
        repo=inputs.repo,
        commit=inputs.commit,
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
            "community_id": (
                structure.community_id if graph.number_of_nodes() <= MAX_COMMUNITY_NODES else {}
            ),
        },
        comp_id=comp_id,
        comp_size=comp_size,
        created_at=ctx.resolved_now(),
    )
    return list(config.build_rows(metric_inputs))


__all__ = [
    "UndirectedMetricInputs",
    "UndirectedMetricsConfig",
    "build_undirected_symbol_metric_rows",
]
