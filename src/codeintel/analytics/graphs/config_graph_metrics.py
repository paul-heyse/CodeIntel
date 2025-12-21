"""Config bipartite/projection graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.compute.graphs import (
    build_projection_graph,
    log_empty_graph,
    log_projection_skipped,
    projection_metrics,
)
from codeintel.analytics.graphs.constants import MAX_BETWEENNESS_NODES
from codeintel.analytics.utilities.datasets import validate_contract_rows
from codeintel.build.schemas import get_contract_for_table_key
from codeintel.config.primitives import SnapshotRef
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.graphs.runtime.context import GraphContextSpec, resolve_graph_context

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.analytics.compute.graphs import (
        ProjectionMetrics,
    )
    from codeintel.graphs.runtime.context import GraphContext
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
    from codeintel.storage.gateway import StorageGateway


CONFIG_GRAPH_METRICS_KEYS_COLS = (
    "repo",
    "commit",
    "config_key",
    "degree",
    "weighted_degree",
    "betweenness",
    "closeness",
    "community_id",
    "created_at",
)

CONFIG_GRAPH_METRICS_MODULES_COLS = (
    "repo",
    "commit",
    "module_path",
    "degree",
    "weighted_degree",
    "betweenness",
    "closeness",
    "community_id",
    "created_at",
)

CONFIG_PROJECTION_KEY_EDGES_COLS = (
    "repo",
    "commit",
    "src_config_key",
    "dst_config_key",
    "weight",
    "created_at",
)

CONFIG_PROJECTION_MODULE_EDGES_COLS = (
    "repo",
    "commit",
    "src_module_path",
    "dst_module_path",
    "weight",
    "created_at",
)

NODE_ID_INDEX = 2


@dataclass(frozen=True)
class ProjectionContext:
    """Projection execution context."""

    repo: str
    commit: str
    created_at: datetime
    graph_ctx: GraphContext


@dataclass(frozen=True)
class ProjectionTargets:
    """Dataset targets for projection metrics."""

    node_table_key: str
    edge_table_key: str


def _clear_config_tables(backend: DuckDBPolicyBackend, repo: str, commit: str) -> None:
    backend.delete_for_snapshot("analytics.config_graph_metrics_keys", repo=repo, commit=commit)
    backend.delete_for_snapshot("analytics.config_graph_metrics_modules", repo=repo, commit=commit)
    backend.delete_for_snapshot("analytics.config_projection_key_edges", repo=repo, commit=commit)
    backend.delete_for_snapshot(
        "analytics.config_projection_module_edges", repo=repo, commit=commit
    )


def _projection_rows(
    *,
    proj: nx.Graph,
    metrics: ProjectionMetrics,
    context: ProjectionContext,
    targets: ProjectionTargets,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    node_contract = get_contract_for_table_key(targets.node_table_key)
    edge_contract = get_contract_for_table_key(targets.edge_table_key)
    node_columns = node_contract.schema.column_names() if node_contract.schema else []
    edge_columns = edge_contract.schema.column_names() if edge_contract.schema else []
    node_id_col = node_columns[NODE_ID_INDEX] if len(node_columns) > NODE_ID_INDEX else "node"
    src_col = next((col for col in edge_columns if col.startswith("src_")), "src")
    dst_col = next((col for col in edge_columns if col.startswith("dst_")), "dst")

    node_dicts = [
        {
            "repo": context.repo,
            "commit": context.commit,
            node_id_col: node[1],
            "degree": metrics.degree.get(node, 0),
            "weighted_degree": metrics.weighted_degree.get(node, 0.0),
            "betweenness": metrics.betweenness.get(node, 0.0),
            "closeness": metrics.closeness.get(node, 0.0),
            "community_id": metrics.community_id.get(node),
            "created_at": context.created_at,
        }
        for node in proj.nodes
    ]
    edge_dicts = [
        {
            "repo": context.repo,
            "commit": context.commit,
            src_col: src[1],
            dst_col: dst[1],
            "weight": float(data.get("weight", 1.0)),
            "created_at": context.created_at,
        }
        for src, dst, data in proj.edges(data=True)
    ]
    node_rows = validate_contract_rows(node_contract.table_key, node_dicts)
    edge_rows = validate_contract_rows(edge_contract.table_key, edge_dicts)
    node_serializer = row_serializer_for_table_key(node_contract.table_key)
    edge_serializer = row_serializer_for_table_key(edge_contract.table_key)
    return (
        [node_serializer(row) for row in node_rows],
        [edge_serializer(row) for row in edge_rows],
    )


def _projection_payload(
    *,
    graph: nx.Graph,
    nodes: set[tuple[str, str]],
    context: ProjectionContext,
    label: str,
    targets: ProjectionTargets,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    proj = build_projection_graph(
        graph,
        nodes,
        label=label,
    )
    metrics = projection_metrics(
        graph,
        nodes,
        context.graph_ctx,
        projection=proj,
        label=label,
    )
    return _projection_rows(
        proj=proj,
        metrics=metrics,
        context=context,
        targets=targets,
    )


@dataclass(frozen=True)
class ConfigGraphMetricsResult:
    """Result from config graph metrics computation.

    Attributes
    ----------
    key_rows
        Tuple rows for analytics.config_graph_metrics_keys table, or None.
    module_rows
        Tuple rows for analytics.config_graph_metrics_modules table, or None.
    key_edge_rows
        Tuple rows for analytics.config_projection_key_edges table, or None.
    module_edge_rows
        Tuple rows for analytics.config_projection_module_edges table, or None.
    """

    key_rows: tuple[tuple[object, ...], ...] | None
    module_rows: tuple[tuple[object, ...], ...] | None
    key_edge_rows: tuple[tuple[object, ...], ...] | None
    module_edge_rows: tuple[tuple[object, ...], ...] | None


def compute_config_graph_metrics_result(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> ConfigGraphMetricsResult:
    """Compute config graph metrics rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/DuckDBRowsSaver.

    Parameters
    ----------
    gateway
        Storage gateway used for reading graphs.
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    runtime
        Optional runtime supplying cached graphs and backend selection.

    Returns
    -------
    ConfigGraphMetricsResult
        Container with rows for all four config graph metrics tables.
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

    graph = resolved_runtime.ensure_config_module_bipartite()
    if graph.number_of_nodes() == 0:
        log_empty_graph("config_module_bipartite", graph)
        return ConfigGraphMetricsResult(
            key_rows=None, module_rows=None, key_edge_rows=None, module_edge_rows=None
        )
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=resolved_runtime.backend.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
        )
    )
    keys = {node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0}
    modules = set(graph) - keys
    if len(keys) == 0 or len(modules) == 0:
        log_projection_skipped(
            "config_projection",
            "missing partition",
            nodes=0,
            graph_nodes=graph.number_of_nodes(),
        )
        return ConfigGraphMetricsResult(
            key_rows=None, module_rows=None, key_edge_rows=None, module_edge_rows=None
        )

    projection_ctx = ProjectionContext(
        repo=repo,
        commit=commit,
        created_at=ctx.resolved_now(),
        graph_ctx=ctx,
    )
    key_targets = ProjectionTargets(
        node_table_key="analytics.config_graph_metrics_keys",
        edge_table_key="analytics.config_projection_key_edges",
    )
    module_targets = ProjectionTargets(
        node_table_key="analytics.config_graph_metrics_modules",
        edge_table_key="analytics.config_projection_module_edges",
    )

    key_rows, key_edges = _projection_payload(
        graph=graph,
        nodes=keys,
        context=projection_ctx,
        label="config_keys",
        targets=key_targets,
    )
    module_rows, module_edges = _projection_payload(
        graph=graph,
        nodes=modules,
        context=projection_ctx,
        label="config_modules",
        targets=module_targets,
    )

    return ConfigGraphMetricsResult(
        key_rows=tuple(key_rows) if key_rows else None,
        module_rows=tuple(module_rows) if module_rows else None,
        key_edge_rows=tuple(key_edges) if key_edges else None,
        module_edge_rows=tuple(module_edges) if module_edges else None,
    )


def compute_config_graph_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Compute metrics for config keys/modules and their projections.

    Parameters
    ----------
    gateway
        Storage gateway used for reading graphs and writing metrics.
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    runtime
        Optional runtime supplying cached graphs and backend selection.
    """
    backend = gateway.policy
    backend.ensure_table("analytics.config_graph_metrics_keys")
    backend.ensure_table("analytics.config_graph_metrics_modules")
    backend.ensure_table("analytics.config_projection_key_edges")
    backend.ensure_table("analytics.config_projection_module_edges")

    result = compute_config_graph_metrics_result(
        gateway,
        repo=repo,
        commit=commit,
        runtime=runtime,
    )

    _clear_config_tables(backend, repo, commit)

    if result.key_rows:
        backend.bulk_insert("analytics.config_graph_metrics_keys", list(result.key_rows))
    if result.module_rows:
        backend.bulk_insert("analytics.config_graph_metrics_modules", list(result.module_rows))
    if result.key_edge_rows:
        backend.bulk_insert("analytics.config_projection_key_edges", list(result.key_edge_rows))
    if result.module_edge_rows:
        backend.bulk_insert(
            "analytics.config_projection_module_edges", list(result.module_edge_rows)
        )
