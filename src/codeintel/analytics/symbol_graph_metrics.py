"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    GraphBundle,
    GraphContext,
    centrality_undirected,
    component_ids_undirected,
    log_empty_graph,
    structural_metrics,
    to_decimal_id,
)
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_symbol_function_graph, load_symbol_module_graph
from codeintel.storage.gateway import StorageGateway

MAX_BETWEENNESS_NODES = 1000
MAX_COMMUNITY_NODES = 5000


def compute_symbol_graph_metrics_modules(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.symbol_graph_metrics_modules from module symbol coupling."""
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    graph_ctx = runtime.graph_ctx
    use_gpu = runtime.use_gpu
    con = gateway.con
    ensure_schema(con, "analytics.symbol_graph_metrics_modules")
    ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        betweenness_sample=MAX_BETWEENNESS_NODES,
        pagerank_weight="weight",
        betweenness_weight="weight",
        use_gpu=use_gpu,
    )
    if ctx.betweenness_sample > MAX_BETWEENNESS_NODES:
        ctx = replace(ctx, betweenness_sample=MAX_BETWEENNESS_NODES)
    if ctx.use_gpu != use_gpu:
        ctx = replace(ctx, use_gpu=use_gpu)
    if context is not None and (context.repo != repo or context.commit != commit):
        return

    bundle: GraphBundle[nx.Graph] = GraphBundle(
        ctx=ctx,
        loaders={
            "symbol_module_graph": lambda: load_symbol_module_graph(
                gateway,
                repo,
                commit,
                use_gpu=use_gpu,
            )
        },
    )
    graph = bundle.get("symbol_module_graph")
    if graph.number_of_nodes() == 0:
        log_empty_graph("symbol_module_graph", graph)
        con.execute(
            "DELETE FROM analytics.symbol_graph_metrics_modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        return

    centrality = centrality_undirected(graph, ctx)
    structure = structural_metrics(graph, weight=ctx.pagerank_weight)
    community_map = structure.community_id if graph.number_of_nodes() <= MAX_COMMUNITY_NODES else {}
    comp_id, comp_size = component_ids_undirected(graph)

    now = ctx.resolved_now()
    rows = [
        (
            repo,
            commit,
            module,
            centrality.betweenness.get(module, 0.0),
            centrality.closeness.get(module, 0.0),
            centrality.eigenvector.get(module, 0.0),
            centrality.harmonic.get(module, 0.0),
            structure.core_number.get(module),
            structure.constraint.get(module, 0.0),
            structure.effective_size.get(module, 0.0),
            community_map.get(module),
            comp_id.get(module),
            comp_size.get(module),
            now,
        )
        for module in graph.nodes
    ]
    con.execute(
        "DELETE FROM analytics.symbol_graph_metrics_modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.symbol_graph_metrics_modules (
                repo, commit, module,
                symbol_betweenness, symbol_closeness, symbol_eigenvector, symbol_harmonic,
                symbol_k_core, symbol_constraint, symbol_effective_size,
                symbol_community_id, symbol_component_id, symbol_component_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def compute_symbol_graph_metrics_functions(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.symbol_graph_metrics_functions from function symbol coupling."""
    runtime = runtime or GraphRuntimeOptions()
    context = runtime.context
    graph_ctx = runtime.graph_ctx
    use_gpu = runtime.use_gpu
    con = gateway.con
    ensure_schema(con, "analytics.symbol_graph_metrics_functions")
    ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        betweenness_sample=MAX_BETWEENNESS_NODES,
        pagerank_weight="weight",
        betweenness_weight="weight",
        use_gpu=use_gpu,
    )
    if ctx.betweenness_sample > MAX_BETWEENNESS_NODES:
        ctx = replace(ctx, betweenness_sample=MAX_BETWEENNESS_NODES)
    if ctx.use_gpu != use_gpu:
        ctx = replace(ctx, use_gpu=use_gpu)
    if context is not None and (context.repo != repo or context.commit != commit):
        return

    bundle: GraphBundle[nx.Graph] = GraphBundle(
        ctx=ctx,
        loaders={
            "symbol_function_graph": lambda: load_symbol_function_graph(
                gateway,
                repo,
                commit,
                use_gpu=use_gpu,
            )
        },
    )
    graph = bundle.get("symbol_function_graph")
    if graph.number_of_nodes() == 0:
        log_empty_graph("symbol_function_graph", graph)
        con.execute(
            "DELETE FROM analytics.symbol_graph_metrics_functions WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        return

    centrality = centrality_undirected(graph, ctx)
    structure = structural_metrics(graph, weight=ctx.pagerank_weight)
    community_map = structure.community_id if graph.number_of_nodes() <= MAX_COMMUNITY_NODES else {}
    comp_id, comp_size = component_ids_undirected(graph)

    now = ctx.resolved_now()
    rows = [
        (
            repo,
            commit,
            to_decimal_id(goid),
            centrality.betweenness.get(goid, 0.0),
            centrality.closeness.get(goid, 0.0),
            centrality.eigenvector.get(goid, 0.0),
            centrality.harmonic.get(goid, 0.0),
            structure.core_number.get(goid),
            structure.constraint.get(goid, 0.0),
            structure.effective_size.get(goid, 0.0),
            community_map.get(goid),
            comp_id.get(goid),
            comp_size.get(goid),
            now,
        )
        for goid in graph.nodes
    ]
    con.execute(
        "DELETE FROM analytics.symbol_graph_metrics_functions WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.symbol_graph_metrics_functions (
                repo, commit, function_goid_h128,
                symbol_betweenness, symbol_closeness, symbol_eigenvector, symbol_harmonic,
                symbol_k_core, symbol_constraint, symbol_effective_size,
                symbol_community_id, symbol_component_id, symbol_component_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
