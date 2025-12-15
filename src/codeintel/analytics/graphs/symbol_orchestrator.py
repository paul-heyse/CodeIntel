"""Orchestrator for undirected symbol graph metrics computation.

This module provides a generic orchestrator pattern for computing
symbol coupling metrics on undirected graphs, reducing code duplication
between module-level and function-level metrics computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.compute.graphs import (
    centrality_undirected,
    component_ids_undirected,
    log_empty_graph,
    structural_metrics,
)
from codeintel.analytics.compute.row_builders import SymbolMetricInputs
from codeintel.analytics.graphs.constants import MAX_BETWEENNESS_NODES, MAX_COMMUNITY_NODES
from codeintel.analytics.utilities.datasets import validate_tuple_rows
from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.graphs.runtime.context import GraphContextSpec, resolve_graph_context

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import networkx as nx

    from codeintel.storage.gateway import StorageGateway


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
    get_graph
        Function to extract the graph from resolved runtime.
    get_known_nodes
        Function to get known nodes from the database.
    filter_node
        Function to check if a node should be included.
    build_rows
        Function to build rows from metric inputs.
    """

    table_key: str
    graph_name: str
    get_graph: Callable[[GraphRuntime], nx.Graph]
    get_known_nodes: Callable[[StorageGateway, str, str], set[TNode]]
    filter_node: Callable[[object, set[TNode]], bool]
    build_rows: Callable[[SymbolMetricInputs[TNode]], Sequence[tuple[object, ...]]]


def compute_undirected_symbol_metrics[TNode](
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    config: UndirectedMetricsConfig[TNode],
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Compute and persist undirected symbol graph metrics.

    This orchestrator handles the common pattern for computing symbol coupling
    metrics on undirected graphs, including centrality, structural metrics,
    and component information.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit identifier.
    config
        Configuration specifying table, graph extraction, and row building.
    runtime
        Optional graph runtime or options.
    """
    runtime_opts = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    snapshot = runtime_opts.snapshot or SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_opts,
    )
    backend = gateway.policy
    backend.ensure_table(config.table_key)
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=resolved_runtime.backend.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
            community_detection_limit=runtime_opts.features.community_detection_limit,
        )
    )

    graph = config.get_graph(resolved_runtime)
    known_nodes = config.get_known_nodes(gateway, repo, commit)
    if known_nodes:
        graph = graph.subgraph(
            [node for node in graph.nodes if config.filter_node(node, known_nodes)]
        ).copy()
    if graph.number_of_nodes() == 0:
        log_empty_graph(config.graph_name, graph)
        backend.delete_for_snapshot(config.table_key, repo=repo, commit=commit)
        return

    centrality = centrality_undirected(graph, ctx)
    structure = structural_metrics(
        graph,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
    comp_id, comp_size = component_ids_undirected(graph)

    inputs = SymbolMetricInputs[TNode](
        repo=repo,
        commit=commit,
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
    rows = config.build_rows(inputs)
    contract = DATASET_CONTRACTS_BY_TABLE_KEY[config.table_key]
    validated_rows = validate_tuple_rows(
        contract.table_key,
        list(rows),
        schema=contract.schema,
    )
    backend.delete_for_snapshot(config.table_key, repo=repo, commit=commit)
    if validated_rows:
        backend.bulk_insert(config.table_key, validated_rows)


__all__ = [
    "UndirectedMetricsConfig",
    "compute_undirected_symbol_metrics",
]
