"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural hotspots and coupling signals.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.graph_rows import (
    FunctionGraphMetricInputs,
    ModuleGraphMetricInputs,
    build_function_graph_metric_rows,
    build_module_graph_metric_rows,
    component_metadata_from_import_table,
    load_symbol_module_edges,
    merge_component_metadata,
)
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
from codeintel.analytics.graph_service_runtime import GraphContextSpec, resolve_graph_context
from codeintel.config import GraphMetricsStepConfig
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.sql_helpers import ensure_schema

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext

log = logging.getLogger(__name__)


def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
    analytics_context: AnalyticsContext | None = None,
) -> None:
    """
    Populate analytics graph metrics tables for the provided repo/commit.

    Parameters
    ----------
    gateway :
        Storage gateway used for graph reads and metric writes.
    cfg :
        Graph metrics configuration for the current repository snapshot.
    catalog_provider :
        Optional catalog provider reused for module lookups.
    runtime : GraphRuntime | GraphRuntimeOptions | None
        Shared graph runtime supplying cached graphs and backend selection.
    analytics_context :
        Optional analytics context reused for module lookups.
    """
    runtime_opts: GraphRuntimeOptions = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    resolved_runtime = resolve_graph_runtime(
        gateway,
        cfg.snapshot,
        runtime_opts,
        context=analytics_context or runtime_opts.context,
    )
    use_gpu = resolved_runtime.backend.use_gpu

    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions")
    ensure_schema(con, "analytics.graph_metrics_modules")
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=cfg.repo,
            commit=cfg.commit,
            use_gpu=use_gpu,
            metrics_cfg=cfg,
            now=datetime.now(tz=UTC),
        )
    )
    _compute_function_graph_metrics(gateway, cfg, ctx=ctx, runtime=resolved_runtime)
    module_by_path = None
    active_context = analytics_context or runtime_opts.context
    if active_context is not None:
        module_by_path = active_context.module_map
    elif catalog_provider is not None:
        module_by_path = catalog_provider.catalog().module_by_path
    _compute_module_graph_metrics(
        gateway,
        cfg,
        ctx=ctx,
        runtime=resolved_runtime,
        module_by_path=module_by_path,
    )


def _compute_function_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    ctx: GraphContext,
    runtime: GraphRuntime,
) -> None:
    con = gateway.con
    graph = runtime.ensure_call_graph()
    stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, ctx)
    components = component_metadata(graph)
    created_at = ctx.resolved_now()

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }

    con.execute(
        "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    rows = build_function_graph_metric_rows(
        FunctionGraphMetricInputs(
            cfg=cfg,
            stats=stats,
            centrality=centrality,
            components=components,
            graph_nodes=sorted(graph.nodes),
            created_at=created_at,
        )
    )

    if rows:
        gateway.analytics.insert_graph_metrics_functions(rows)
        log.info(
            "graph_metrics_functions populated: %d rows for %s@%s",
            len(rows),
            cfg.repo,
            cfg.commit,
        )


def _compute_module_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    ctx: GraphContext,
    runtime: GraphRuntime,
    module_by_path: dict[str, str] | None,
) -> None:
    con = gateway.con
    graph = runtime.ensure_import_graph()
    symbol_modules, symbol_inbound, symbol_outbound = load_symbol_module_edges(
        gateway, module_by_path
    )
    modules = set(graph.nodes) | symbol_modules
    module_repo = ModuleRepository(gateway=gateway, repo=cfg.repo, commit=cfg.commit)
    modules.update(module_repo.list_modules())
    if modules:
        graph.add_nodes_from(modules)

    import_stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, ctx)
    component_raw = component_metadata(graph)
    cached_component_meta = component_metadata_from_import_table(gateway, cfg.repo, cfg.commit)
    component_meta = merge_component_metadata(
        modules,
        {
            "component_id": dict(component_raw.component_id),
            "in_cycle": dict(component_raw.in_cycle),
            "layer": dict(component_raw.layer),
        },
        cached_component_meta,
    )

    con.execute(
        "DELETE FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }
    rows_to_insert = build_module_graph_metric_rows(
        ModuleGraphMetricInputs(
            cfg=cfg,
            modules=modules,
            import_stats=import_stats,
            centrality=centrality,
            component_meta=component_meta,
            symbol_inbound=symbol_inbound,
            symbol_outbound=symbol_outbound,
            created_at=ctx.resolved_now(),
        )
    )

    if rows_to_insert:
        gateway.analytics.insert_graph_metrics_modules(rows_to_insert)
        log.info(
            "graph_metrics_modules populated: %d rows for %s@%s",
            len(rows_to_insert),
            cfg.repo,
            cfg.commit,
        )
