"""Global graph statistics for core graphs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import networkx as nx

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_service import (
    GraphContext,
    build_projection_graph,
    global_graph_stats,
)
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import (
    load_call_graph,
    load_config_module_bipartite,
    load_import_graph,
    load_symbol_function_graph,
    load_symbol_module_graph,
)
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def compute_graph_stats(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    context: AnalyticsContext | None = None,
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
    context : AnalyticsContext | None
        Optional shared context to reuse cached call/import graphs.
    """
    con = gateway.con
    ensure_schema(con, "analytics.graph_stats")
    ctx = GraphContext(repo=repo, commit=commit, now=datetime.now(UTC))

    call_graph = (
        context.call_graph
        if context is not None and context.call_graph is not None
        else load_call_graph(gateway, repo, commit)
    )
    import_graph = (
        context.import_graph
        if context is not None and context.import_graph is not None
        else load_import_graph(gateway, repo, commit)
    )
    graphs: dict[str, nx.Graph | nx.DiGraph] = {
        "call_graph": call_graph,
        "import_graph": import_graph,
        "symbol_module_graph": load_symbol_module_graph(gateway, repo, commit),
        "symbol_function_graph": load_symbol_function_graph(gateway, repo, commit),
    }

    config_bipartite = load_config_module_bipartite(gateway, repo, commit)
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
