"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural hotspots and coupling signals.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    GraphContext,
    build_graph_context,
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
from codeintel.config import GraphMetricsStepConfig
from codeintel.graphs.engine import GraphEngine
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

log = logging.getLogger(__name__)


def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntimeOptions | None = None,
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
    runtime : GraphRuntimeOptions | None
        Optional runtime options supplying cached graphs and backend selection.
    """
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    graph_ctx = runtime.graph_ctx
    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions")
    ensure_schema(con, "analytics.graph_metrics_modules")
    use_gpu = runtime.use_gpu
    ctx = graph_ctx or build_graph_context(
        cfg,
        now=datetime.now(tz=UTC),
        use_gpu=use_gpu,
    )
    if ctx.repo != cfg.repo or ctx.commit != cfg.commit or ctx.use_gpu != use_gpu:
        ctx = replace(ctx, repo=cfg.repo, commit=cfg.commit, use_gpu=use_gpu)
    engine: GraphEngine = runtime.build_engine(gateway, cfg.repo, cfg.commit)
    _compute_function_graph_metrics(gateway, cfg, ctx=ctx, engine=engine)
    module_by_path = None
    if context is not None:
        module_by_path = context.module_map
    elif catalog_provider is not None:
        module_by_path = catalog_provider.catalog().module_by_path
    _compute_module_graph_metrics(
        gateway,
        cfg,
        ctx=ctx,
        engine=engine,
        module_by_path=module_by_path,
    )


def _compute_function_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    ctx: GraphContext,
    engine: GraphEngine,
) -> None:
    con = gateway.con
    graph = engine.call_graph()
    stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality = centrality_directed(graph, ctx)
    components = component_metadata(graph)
    created_at = ctx.resolved_now()

    con.execute(
        "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    rows: list[tuple[object, ...]] = [
        (
            cfg.repo,
            cfg.commit,
            node,
            len(stats.in_neighbors.get(node, ())),
            len(stats.out_neighbors.get(node, ())),
            stats.in_counts.get(node, 0),
            stats.out_counts.get(node, 0),
            centrality.pagerank.get(node),
            centrality.betweenness.get(node),
            centrality.closeness.get(node),
            components.in_cycle.get(node, False),
            components.scc_id.get(node),
            components.layer.get(node),
            created_at,
        )
        for node in sorted(graph.nodes)
    ]

    if rows:
        con.executemany(
            """
            INSERT INTO analytics.graph_metrics_functions (
                repo, commit, function_goid_h128,
                call_fan_in, call_fan_out, call_in_degree, call_out_degree,
                call_pagerank, call_betweenness, call_closeness,
                call_cycle_member, call_cycle_id, call_layer, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        log.info(
            "graph_metrics_functions populated: %d rows for %s@%s",
            len(rows),
            cfg.repo,
            cfg.commit,
        )


def _component_metadata_from_import_table(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, dict[str, int | bool]] | None:
    try:
        rows = gateway.con.execute(
            """
            SELECT module, scc_id, component_size, layer
            FROM graph.import_modules
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except DuckDBError:
        return None
    if not rows:
        return None

    comp_id: dict[str, int] = {}
    in_cycle: dict[str, bool] = {}
    layer_by_module: dict[str, int] = {}
    for module, scc_id, component_size, layer in rows:
        name = str(module)
        comp_id[name] = int(scc_id) if scc_id is not None else -1
        size = int(component_size) if component_size is not None else 0
        in_cycle[name] = size > 1
        if layer is not None:
            layer_by_module[name] = int(layer)
    in_cycle_cast: dict[str, int | bool] = {node: bool(flag) for node, flag in in_cycle.items()}
    component_id_cast: dict[str, int | bool] = {node: int(val) for node, val in comp_id.items()}
    layer_cast: dict[str, int | bool] = {node: int(val) for node, val in layer_by_module.items()}
    return {
        "component_id": component_id_cast,
        "in_cycle": in_cycle_cast,
        "layer": layer_cast,
    }


def _merge_component_metadata(
    graph_nodes: set[Any],
    computed: Mapping[str, Mapping[Any, int | bool]],
    cached: Mapping[str, Mapping[Any, int | bool]] | None,
) -> dict[str, dict[Any, int | bool]]:
    if cached is None:
        return {
            "component_id": dict(computed["component_id"]),
            "in_cycle": dict(computed["in_cycle"]),
            "layer": dict(computed["layer"]),
        }
    ids = dict(computed["component_id"])
    in_cycle = dict(computed["in_cycle"])
    layer = dict(computed["layer"])
    for node in graph_nodes:
        if node in cached.get("component_id", {}):
            ids[node] = cached["component_id"][node]
            in_cycle[node] = bool(cached["in_cycle"].get(node, False))
            layer[node] = int(cached["layer"].get(node, layer.get(node, 0)))
    return {"component_id": ids, "in_cycle": in_cycle, "layer": layer}


def _load_symbol_module_edges(
    gateway: StorageGateway,
    module_by_path: dict[str, str] | None,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    modules: set[str] = set()
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)

    if module_by_path is None:
        rows = gateway.con.execute(
            """
            SELECT m_use.module, m_def.module
            FROM graph.symbol_use_edges su
            LEFT JOIN core.modules m_def ON m_def.path = su.def_path
            LEFT JOIN core.modules m_use ON m_use.path = su.use_path
            WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
            """
        ).fetchall()

        for use_module, def_module in rows:
            src = str(use_module)
            dst = str(def_module)
            modules.update((src, dst))
            outbound[src].add(dst)
            inbound[dst].add(src)
        return modules, inbound, outbound

    path_rows = gateway.con.execute(
        "SELECT def_path, use_path FROM graph.symbol_use_edges"
    ).fetchall()
    for def_path, use_path in path_rows:
        def_module = module_by_path.get(str(def_path))
        use_module = module_by_path.get(str(use_path))
        if def_module is None or use_module is None:
            continue
        modules.update((use_module, def_module))
        outbound[use_module].add(def_module)
        inbound[def_module].add(use_module)

    return modules, inbound, outbound


def _compute_module_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    ctx: GraphContext,
    engine: GraphEngine,
    module_by_path: dict[str, str] | None,
) -> None:
    con = gateway.con
    graph = engine.import_graph()
    symbol_modules, symbol_inbound, symbol_outbound = _load_symbol_module_edges(
        gateway, module_by_path
    )
    modules = set(graph.nodes) | symbol_modules
    module_rows = con.execute(
        "SELECT module FROM core.modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    for (module,) in module_rows:
        if module is not None:
            modules.add(str(module))
    if modules:
        graph.add_nodes_from(modules)

    import_stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality = centrality_directed(graph, ctx)
    component_raw = component_metadata(graph)
    cached_component_meta = _component_metadata_from_import_table(gateway, cfg.repo, cfg.commit)
    component_meta = _merge_component_metadata(
        modules,
        {
            "component_id": component_raw.component_id,
            "in_cycle": component_raw.in_cycle,
            "layer": component_raw.layer,
        },
        cached_component_meta,
    )

    con.execute(
        "DELETE FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    now = ctx.resolved_now()
    rows_to_insert: list[tuple[object, ...]] = [
        (
            cfg.repo,
            cfg.commit,
            module,
            len(import_stats.in_neighbors.get(module, ())),
            len(import_stats.out_neighbors.get(module, ())),
            import_stats.in_counts.get(module, 0),
            import_stats.out_counts.get(module, 0),
            centrality.pagerank.get(module),
            centrality.betweenness.get(module),
            centrality.closeness.get(module),
            component_meta["in_cycle"].get(module, False),
            component_meta["component_id"].get(module),
            component_meta["layer"].get(module),
            len(symbol_inbound.get(module, ())),
            len(symbol_outbound.get(module, ())),
            now,
        )
        for module in sorted(modules)
    ]

    if rows_to_insert:
        con.executemany(
            """
            INSERT INTO analytics.graph_metrics_modules (
                repo, commit, module,
                import_fan_in, import_fan_out, import_in_degree, import_out_degree,
                import_pagerank, import_betweenness, import_closeness,
                import_cycle_member, import_cycle_id, import_layer,
                symbol_fan_in, symbol_fan_out, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        log.info(
            "graph_metrics_modules populated: %d rows for %s@%s",
            len(rows_to_insert),
            cfg.repo,
            cfg.commit,
        )
