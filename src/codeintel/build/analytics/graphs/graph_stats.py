"""Global graph statistics for core graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.build.analytics.compute.graphs import (
    build_projection_graph,
    global_graph_stats,
)
from codeintel.build.analytics.compute.row_builders import row_tuple_for_table
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count

GRAPH_STATS_TABLE_KEY = "analytics.graph_stats"


@dataclass(frozen=True)
class GraphStatsInputs:
    """Inputs required to compute global graph stats rows."""

    repo: str
    commit: str
    call_graph: GraphInput
    import_graph: GraphInput
    symbol_module_graph: GraphInput
    symbol_function_graph: GraphInput
    config_module_bipartite: GraphInput | None = None
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

    graphs: dict[str, GraphInput] = {
        "call_graph": inputs.call_graph,
        "import_graph": inputs.import_graph,
        "symbol_module_graph": inputs.symbol_module_graph,
        "symbol_function_graph": inputs.symbol_function_graph,
    }

    config_graph = inputs.config_module_bipartite
    if config_graph is not None and graph_node_count(config_graph) > 0:
        store = ensure_store(config_graph)
        keys = {
            node for node in store.node_ids() if store.get_node_attrs(node).get("bipartite") == 0
        }
        modules = set(store.node_ids()) - keys
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
            row_tuple_for_table(
                GRAPH_STATS_TABLE_KEY,
                {
                    "graph_name": name,
                    "repo": inputs.repo,
                    "commit": inputs.commit,
                    "node_count": stats.node_count,
                    "edge_count": stats.edge_count,
                    "weak_component_count": stats.weak_component_count,
                    "scc_count": stats.scc_count,
                    "component_layers": stats.component_layers,
                    "avg_clustering": stats.avg_clustering,
                    "diameter_estimate": stats.diameter_estimate,
                    "avg_shortest_path_estimate": stats.avg_shortest_path_estimate,
                    "created_at": now,
                },
            )
        )

    return rows
