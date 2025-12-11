"""Global graph statistics for core graphs.

This module computes and persists global graph statistics using
DuckDBPolicyBackend for bulk insert operations.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.compute.graphs import (
    build_projection_graph,
    global_graph_stats,
)
from codeintel.analytics.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

# Column definitions for graph_stats table
_GRAPH_STATS_COLUMNS: tuple[str, ...] = (
    "graph_name",
    "repo",
    "commit",
    "node_count",
    "edge_count",
    "weak_component_count",
    "scc_count",
    "component_layers",
    "avg_clustering",
    "diameter_estimate",
    "avg_shortest_path_estimate",
    "created_at",
)


def compute_graph_stats(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """
    Populate analytics.graph_stats for call/import and related graphs.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection for source reads and destination writes.
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
    use_gpu = resolved_runtime.backend.use_gpu
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=use_gpu,
            now=datetime.now(UTC),
        )
    )

    config_bipartite = resolved_runtime.ensure_config_module_bipartite()
    graphs: dict[str, nx.Graph | nx.DiGraph] = {
        "call_graph": resolved_runtime.ensure_call_graph(),
        "import_graph": resolved_runtime.ensure_import_graph(),
        "symbol_module_graph": resolved_runtime.ensure_symbol_module_graph(),
        "symbol_function_graph": resolved_runtime.ensure_symbol_function_graph(),
    }

    if config_bipartite.number_of_nodes() > 0:
        keys = {n for n, d in config_bipartite.nodes(data=True) if d.get("bipartite") == 0}
        modules = set(config_bipartite) - keys
        if keys and modules:
            graphs["config_key_projection"] = build_projection_graph(
                config_bipartite,
                keys,
                label="config_keys",
            )
        if keys and modules and len(modules) > 1:
            graphs["config_module_projection"] = build_projection_graph(
                config_bipartite,
                modules,
                label="config_modules",
            )

    now = ctx.resolved_now()
    rows: list[tuple[object, ...]] = []

    for name, graph in graphs.items():
        stats = global_graph_stats(graph)
        rows.append(
            (
                name,
                repo,
                commit,
                stats.node_count,
                stats.edge_count,
                stats.weak_component_count,
                stats.scc_count,
                stats.component_layers,
                stats.avg_clustering,
                stats.diameter_estimate,
                stats.avg_shortest_path_estimate,
                now,
            )
        )

    # Use policy backend for delete and bulk insert
    backend = DuckDBPolicyBackend(gateway)
    backend.delete_for_snapshot("analytics.graph_stats", repo=repo, commit=commit)
    if rows:
        backend.bulk_insert("analytics.graph_stats", rows, columns=list(_GRAPH_STATS_COLUMNS))
