"""Global graph statistics for core graphs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.compute.graphs import (
    build_projection_graph,
    global_graph_stats,
)
from codeintel.analytics.utilities.datasets import write_analytics_tuple_rows
from codeintel.analytics.utilities.persistence import DeleteScope
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.graphs.runtime.context import GraphContextSpec, resolve_graph_context

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.storage.gateway import StorageGateway


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

    delete_scope = DeleteScope(repo=repo, commit=commit)
    write_analytics_tuple_rows(
        gateway,
        "analytics.graph_stats",
        rows,
        delete_scope=delete_scope,
        columns=_GRAPH_STATS_COLUMNS,
    )
