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
from codeintel.analytics.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.runtime.context import (
    GraphContextSpec,
    resolve_graph_context,
)
from codeintel.analytics.utilities.datasets import validate_tuple_rows
from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.sql.builder import ensure_schema

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.analytics.compute.graphs import (
        ProjectionMetrics,
    )
    from codeintel.analytics.runtime.context import (
        GraphContext,
    )
    from codeintel.storage.gateway import StorageGateway

MAX_BETWEENNESS_NODES = 1000
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


def _clear_config_tables(gateway: StorageGateway, repo: str, commit: str) -> None:
    con = gateway.con
    con.execute(
        "DELETE FROM analytics.config_graph_metrics_keys WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_graph_metrics_modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_projection_key_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_projection_module_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )


def _projection_rows(
    *,
    proj: nx.Graph,
    metrics: ProjectionMetrics,
    context: ProjectionContext,
    targets: ProjectionTargets,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    node_contract = DATASET_CONTRACTS_BY_TABLE_KEY[targets.node_table_key]
    edge_contract = DATASET_CONTRACTS_BY_TABLE_KEY[targets.edge_table_key]
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
    node_rows = validate_tuple_rows(
        node_contract.table_key,
        node_dicts,
        schema=node_contract.schema,
    )
    edge_rows = validate_tuple_rows(
        edge_contract.table_key,
        edge_dicts,
        schema=edge_contract.schema,
    )
    return node_rows, edge_rows


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


def compute_config_graph_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """
    Compute metrics for config keys/modules and their projections.

    Parameters
    ----------
    gateway :
        Storage gateway used for reading graphs and writing metrics.
    repo : str
        Repository identifier anchoring the metrics.
    commit : str
        Commit hash anchoring the metrics snapshot.
    runtime : GraphRuntime | GraphRuntimeOptions | None
        Optional runtime supplying cached graphs and backend selection.
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
    ensure_schema(gateway.con, "analytics.config_graph_metrics_keys")
    ensure_schema(gateway.con, "analytics.config_graph_metrics_modules")
    ensure_schema(gateway.con, "analytics.config_projection_key_edges")
    ensure_schema(gateway.con, "analytics.config_projection_module_edges")

    graph = resolved_runtime.ensure_config_module_bipartite()
    if graph.number_of_nodes() == 0:
        log_empty_graph("config_module_bipartite", graph)
        _clear_config_tables(gateway, repo, commit)
        return
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
    created_at = ctx.resolved_now()

    keys = {node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0}
    modules = set(graph) - keys
    if len(keys) == 0 or len(modules) == 0:
        log_projection_skipped(
            "config_projection",
            "missing partition",
            nodes=0,
            graph_nodes=graph.number_of_nodes(),
        )
        _clear_config_tables(gateway, repo, commit)
        return

    projection_ctx = ProjectionContext(
        repo=repo,
        commit=commit,
        created_at=created_at,
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

    _clear_config_tables(gateway, repo, commit)

    if key_rows:
        gateway.ibis.write(
            "analytics.config_graph_metrics_keys",
            key_rows,
            columns=[
                "repo",
                "commit",
                "config_key",
                "degree",
                "weighted_degree",
                "betweenness",
                "closeness",
                "community_id",
                "created_at",
            ],
        )
    if module_rows:
        gateway.ibis.write(
            "analytics.config_graph_metrics_modules",
            module_rows,
            columns=[
                "repo",
                "commit",
                "module",
                "degree",
                "weighted_degree",
                "betweenness",
                "closeness",
                "community_id",
                "created_at",
            ],
        )
    if key_edges:
        gateway.ibis.write(
            "analytics.config_projection_key_edges",
            key_edges,
            columns=["repo", "commit", "src_key", "dst_key", "weight", "created_at"],
        )
    if module_edges:
        gateway.ibis.write(
            "analytics.config_projection_module_edges",
            module_edges,
            columns=["repo", "commit", "src_module", "dst_module", "weight", "created_at"],
        )
