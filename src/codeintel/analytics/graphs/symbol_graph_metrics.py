"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.graph_rows import (
    SymbolFunctionMetricInputs,
    SymbolModuleMetricInputs,
    build_symbol_function_rows,
    build_symbol_module_rows,
)
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graph_service import (
    centrality_undirected,
    component_ids_undirected,
    log_empty_graph,
    structural_metrics,
)
from codeintel.analytics.graph_service_runtime import GraphContextSpec, resolve_graph_context
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.sql_helpers import ensure_schema

MAX_BETWEENNESS_NODES = 1000
MAX_COMMUNITY_NODES = 5000


def compute_symbol_graph_metrics_modules(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.symbol_graph_metrics_modules from module symbol coupling."""
    runtime_opts = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    snapshot = runtime_opts.snapshot or SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_opts,
        context=runtime_opts.context,
    )
    con = gateway.con
    ensure_schema(con, "analytics.symbol_graph_metrics_modules")
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=resolved_runtime.backend.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
        )
    )
    if runtime_opts.context is not None and (
        runtime_opts.context.repo != repo or runtime_opts.context.commit != commit
    ):
        return

    graph = resolved_runtime.ensure_symbol_module_graph()
    module_repo = ModuleRepository(gateway=gateway, repo=repo, commit=commit)
    known_modules = set(module_repo.list_modules())
    if known_modules:
        graph = graph.subgraph([module for module in graph.nodes if module in known_modules]).copy()
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

    rows = build_symbol_module_rows(
        SymbolModuleMetricInputs(
            repo=repo,
            commit=commit,
            centrality={
                "betweenness": centrality.betweenness,
                "closeness": centrality.closeness,
                "eigenvector": centrality.eigenvector,
                "harmonic": centrality.harmonic,
            },
            structure={
                "core_number": structure.core_number,
                "constraint": structure.constraint,
                "effective_size": structure.effective_size,
                "community_id": community_map,
            },
            comp_id=comp_id,
            comp_size=comp_size,
            created_at=ctx.resolved_now(),
        )
    )
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
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.symbol_graph_metrics_functions from function symbol coupling."""
    runtime_opts = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    snapshot = runtime_opts.snapshot or SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_opts,
        context=runtime_opts.context,
    )
    con = gateway.con
    ensure_schema(con, "analytics.symbol_graph_metrics_functions")
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=resolved_runtime.backend.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=MAX_BETWEENNESS_NODES,
            pagerank_weight="weight",
            betweenness_weight="weight",
        )
    )
    if runtime_opts.context is not None and (
        runtime_opts.context.repo != repo or runtime_opts.context.commit != commit
    ):
        return

    graph = resolved_runtime.ensure_symbol_function_graph()
    function_repo = FunctionRepository(gateway=gateway, repo=repo, commit=commit)
    known_goids = set(function_repo.list_function_goids())
    if known_goids:
        graph = graph.subgraph([goid for goid in graph.nodes if int(goid) in known_goids]).copy()
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

    rows = build_symbol_function_rows(
        SymbolFunctionMetricInputs(
            repo=repo,
            commit=commit,
            centrality={
                "betweenness": centrality.betweenness,
                "closeness": centrality.closeness,
                "eigenvector": centrality.eigenvector,
                "harmonic": centrality.harmonic,
            },
            structure={
                "core_number": structure.core_number,
                "constraint": structure.constraint,
                "effective_size": structure.effective_size,
                "community_id": community_map,
            },
            comp_id=comp_id,
            comp_size=comp_size,
            created_at=ctx.resolved_now(),
        )
    )
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
