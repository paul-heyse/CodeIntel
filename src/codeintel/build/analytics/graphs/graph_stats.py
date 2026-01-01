"""Global graph statistics for core graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import networkx as nx

from codeintel.build.analytics.compute.graphs import (
    build_projection_graph,
    global_graph_stats,
)
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context

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


@dataclass(frozen=True)
class GraphStatsInputs:
    """Inputs required to compute global graph stats rows."""

    repo: str
    commit: str
    call_graph: nx.DiGraph
    import_graph: nx.DiGraph
    symbol_module_graph: nx.Graph
    symbol_function_graph: nx.Graph
    config_module_bipartite: nx.Graph | None = None
    use_gpu: bool = False


def build_graph_stats_rows(inputs: GraphStatsInputs) -> list[tuple[object, ...]]:
    """
    Build analytics.graph_stats rows for call/import and related graphs.

    Parameters
    ----------
    inputs
        Graph stats inputs with repo/commit metadata and graph sources.

    Returns
    -------
    list[tuple[object, ...]]
        Rows ready for insertion into analytics.graph_stats.
    """
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=inputs.repo,
            commit=inputs.commit,
            use_gpu=inputs.use_gpu,
            now=datetime.now(UTC),
        )
    )

    graphs: dict[str, nx.Graph | nx.DiGraph] = {
        "call_graph": inputs.call_graph,
        "import_graph": inputs.import_graph,
        "symbol_module_graph": inputs.symbol_module_graph,
        "symbol_function_graph": inputs.symbol_function_graph,
    }

    config_graph = inputs.config_module_bipartite
    if config_graph is not None and config_graph.number_of_nodes() > 0:
        keys = {n for n, d in config_graph.nodes(data=True) if d.get("bipartite") == 0}
        modules = set(config_graph) - keys
        if keys and modules:
            graphs["config_key_projection"] = build_projection_graph(
                config_graph,
                keys,
                label="config_keys",
            )
        if keys and modules and len(modules) > 1:
            graphs["config_module_projection"] = build_projection_graph(
                config_graph,
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
                inputs.repo,
                inputs.commit,
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

    return rows
