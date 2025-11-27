"""Global graph statistics for core graphs."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    GraphContext,
    build_projection_graph,
    global_graph_stats,
)
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

log = logging.getLogger(__name__)


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
    runtime_opts = runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    use_gpu = runtime.use_gpu if isinstance(runtime, GraphRuntime) else runtime_opts.use_gpu
    con = gateway.con
    ensure_schema(con, "analytics.graph_stats")
    ctx = runtime_opts.graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        use_gpu=use_gpu,
    )
    if ctx.use_gpu != use_gpu:
        ctx = replace(ctx, use_gpu=use_gpu)

    if isinstance(runtime, GraphRuntime):
        call_graph = runtime.ensure_call_graph()
        import_graph = runtime.ensure_import_graph()
        symbol_module_graph = runtime.ensure_symbol_module_graph()
        symbol_function_graph = runtime.ensure_symbol_function_graph()
        config_bipartite = runtime.ensure_config_module_bipartite()
    else:
        engine: GraphEngine = runtime_opts.build_engine(gateway, repo, commit)
        call_graph = engine.call_graph()
        import_graph = engine.import_graph()
        symbol_module_graph = engine.symbol_module_graph()
        symbol_function_graph = engine.symbol_function_graph()
        config_bipartite = engine.config_module_bipartite()
    graphs: dict[str, nx.Graph | nx.DiGraph] = {
        "call_graph": call_graph,
        "import_graph": import_graph,
        "symbol_module_graph": symbol_module_graph,
        "symbol_function_graph": symbol_function_graph,
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

    con.execute(
        "DELETE FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.graph_stats (
                graph_name, repo, commit, node_count, edge_count,
                weak_component_count, scc_count, component_layers, avg_clustering,
                diameter_estimate, avg_shortest_path_estimate, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
