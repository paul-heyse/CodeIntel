"""Global graph statistics for core graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.graphs import (
    build_projection_graph,
    global_graph_stats,
)
from codeintel.build.analytics.compute.row_builders import row_tuple_for_table
from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
from codeintel.build.analytics.graphs.orchestrator import (
    MetricsPipelineConfig,
    MetricsPipelineRequest,
    build_metrics_pipeline_rows,
    build_store_views,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.analytics.compute.graphs import GlobalGraphStats
    from codeintel.build.analytics.graphs.orchestrator import GraphViews
    from codeintel.build.graphs.runtime.context import GraphContext

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
    now = datetime.now(UTC)

    def _build_context(
        _runtime_opts: GraphRuntimeOptions,
        repo: str,
        commit: str,
    ) -> GraphContext:
        return resolve_graph_context(
            GraphContextSpec(
                repo=repo,
                commit=commit,
                use_gpu=inputs.use_gpu,
                now=now,
                runtime_profile=_runtime_opts.runtime_profile,
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

    runtime = GraphRuntimeOptions()
    filters = GraphMetricFilters()
    rows: list[tuple[object, ...]] = []

    def _stats_slices(views: GraphViews, _ctx: GraphContext) -> GlobalGraphStats:
        return global_graph_stats(views.graph)

    def _stats_rows(
        repo: str,
        commit: str,
        ctx: GraphContext,
        _views: GraphViews,
        stats: GlobalGraphStats,
        *,
        graph_name: str,
    ) -> list[tuple[object, ...]]:
        return [
            row_tuple_for_table(
                GRAPH_STATS_TABLE_KEY,
                {
                    "graph_name": graph_name,
                    "repo": repo,
                    "commit": commit,
                    "node_count": stats.node_count,
                    "edge_count": stats.edge_count,
                    "weak_component_count": stats.weak_component_count,
                    "scc_count": stats.scc_count,
                    "component_layers": stats.component_layers,
                    "avg_clustering": stats.avg_clustering,
                    "diameter_estimate": stats.diameter_estimate,
                    "avg_shortest_path_estimate": stats.avg_shortest_path_estimate,
                    "created_at": ctx.resolved_now(),
                },
            )
        ]

    def _stats_rows_builder(
        *,
        graph_name: str,
    ) -> Callable[[str, str, GraphContext, GraphViews, GlobalGraphStats], list[tuple[object, ...]]]:
        def _build_rows(
            repo: str,
            commit: str,
            ctx: GraphContext,
            views: GraphViews,
            stats: GlobalGraphStats,
        ) -> list[tuple[object, ...]]:
            return _stats_rows(
                repo,
                commit,
                ctx,
                views,
                stats,
                graph_name=graph_name,
            )

        return _build_rows

    for name, graph in graphs.items():
        config = MetricsPipelineConfig(
            table_key=GRAPH_STATS_TABLE_KEY,
            filter_graph=lambda _filters, target: target,
            build_context=_build_context,
            build_views=build_store_views,
            build_slices=_stats_slices,
            build_rows=_stats_rows_builder(graph_name=name),
        )
        request = MetricsPipelineRequest(
            repo=inputs.repo,
            commit=inputs.commit,
            graph=graph,
            runtime=runtime,
            filters=filters,
        )
        rows.extend(build_metrics_pipeline_rows(config, request))

    return rows
