"""Core graph metrics plugins using factory pattern.

This module provides the core graph metrics plugins using the hexagonal
architecture's compute layer for pure metric calculations.

Uses resource injection pattern via ctx.require() with fallback
to direct context properties for backward compatibility.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from codeintel.graphs.compute.metrics import centrality, components, coupling
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_metric_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import GraphResource, StorageResource

log = logging.getLogger(__name__)

# =============================================================================
# Computation Functions (using hexagonal compute layer)
# =============================================================================


def _compute_core_graph_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute core function/module graph metrics (centrality, neighbors, components).

    Uses the pure compute layer for metric calculations and persists results
    via the storage gateway.

    Returns
    -------
    ComputationResult
        Success result after computing core metrics.
    """
    # Get the engine - try resources first, fall back to context
    if ctx.resources is not None and ctx.resources.has(GraphResource.RESOURCE_NAME):
        graph_resource = ctx.require(GraphResource)
        call_graph = graph_resource.call_graph()
        import_graph = graph_resource.import_graph()
    elif ctx.engine is not None:
        call_graph = ctx.engine.call_graph()
        import_graph = ctx.engine.import_graph()
    else:
        log.warning("No graph engine available for metrics computation")
        return ComputationResult(success=False, message="No graph engine available")

    # Compute centrality metrics using pure compute functions
    func_centralities = centrality.compute_all_centralities(call_graph)
    module_centralities = centrality.compute_all_centralities(import_graph)

    # Compute SCC for both graphs
    func_sccs = components.find_strongly_connected(call_graph)
    module_sccs = components.find_strongly_connected(import_graph)

    # Persist function metrics
    now = datetime.now(tz=UTC)
    func_rows = [
        (
            goid,
            ctx.repo,
            ctx.commit,
            metrics.pagerank,
            metrics.betweenness,
            metrics.closeness,
            metrics.in_degree,
            metrics.out_degree,
            metrics.degree,
            func_sccs.node_to_component.get(goid, -1),
            now,
        )
        for goid, metrics in func_centralities.items()
    ]

    if func_rows:
        _persist_function_metrics(ctx, func_rows)

    # Persist module metrics
    module_rows = [
        (
            node,
            ctx.repo,
            ctx.commit,
            metrics.pagerank,
            metrics.betweenness,
            metrics.closeness,
            metrics.in_degree,
            metrics.out_degree,
            metrics.degree,
            module_sccs.node_to_component.get(node, -1),
            now,
        )
        for node, metrics in module_centralities.items()
    ]

    if module_rows:
        _persist_module_metrics(ctx, module_rows)

    return ComputationResult.ok(
        row_counts={
            "analytics.graph_metrics_functions": len(func_rows),
            "analytics.graph_metrics_modules": len(module_rows),
        }
    )


def _persist_function_metrics(
    ctx: GraphExecutionContext,
    rows: Sequence[tuple[object, ...]],
) -> None:
    """Persist function metrics to database.

    Uses resource injection with fallback to ctx.gateway.
    """
    from codeintel.ingestion.common import run_batch  # noqa: PLC0415

    # Get gateway via resource injection or fallback
    if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.require(StorageResource)
        gateway = storage.gateway
    else:
        gateway = ctx.gateway

    run_batch(
        gateway,
        "analytics.graph_metrics_functions",
        list(rows),
        delete_params=[ctx.repo, ctx.commit],
        scope="graph_metrics_functions",
    )


def _persist_module_metrics(
    ctx: GraphExecutionContext,
    rows: Sequence[tuple[object, ...]],
) -> None:
    """Persist module metrics to database.

    Uses resource injection with fallback to ctx.gateway.
    """
    from codeintel.ingestion.common import run_batch  # noqa: PLC0415

    # Get gateway via resource injection or fallback
    if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.require(StorageResource)
        gateway = storage.gateway
    else:
        gateway = ctx.gateway

    run_batch(
        gateway,
        "analytics.graph_metrics_modules",
        list(rows),
        delete_params=[ctx.repo, ctx.commit],
        scope="graph_metrics_modules",
    )


def _compute_function_ext_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute extended call graph metrics for functions.

    Returns
    -------
    ComputationResult
        Success result after computing extended function metrics.
    """
    # Get call graph
    if ctx.resources is not None and ctx.resources.has(GraphResource.RESOURCE_NAME):
        graph_resource = ctx.require(GraphResource)
        call_graph = graph_resource.call_graph()
    elif ctx.engine is not None:
        call_graph = ctx.engine.call_graph()
    else:
        return ComputationResult(success=False, message="No graph engine available")

    # Compute coupling metrics
    coupling_metrics = coupling.compute_coupling(call_graph)

    # Compute SCC with condensation for layers
    scc_result = components.find_strongly_connected(call_graph, compute_condensation=True)
    layers = components.condensation_layers(call_graph, scc_result)

    # Build and persist rows
    now = datetime.now(tz=UTC)
    rows = [
        (
            goid,
            ctx.repo,
            ctx.commit,
            metrics.afferent,
            metrics.efferent,
            metrics.instability,
            layers.get(goid, 0),
            scc_result.node_to_component.get(goid, -1),
            now,
        )
        for goid, metrics in coupling_metrics.items()
    ]

    if rows:
        from codeintel.ingestion.common import run_batch  # noqa: PLC0415

        # Get gateway via resource injection or fallback
        if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
            storage = ctx.require(StorageResource)
            gateway = storage.gateway
        else:
            gateway = ctx.gateway

        run_batch(
            gateway,
            "analytics.graph_metrics_functions_ext",
            list(rows),
            delete_params=[ctx.repo, ctx.commit],
            scope="graph_metrics_functions_ext",
        )

    return ComputationResult.ok(row_counts={"analytics.graph_metrics_functions_ext": len(rows)})


def _compute_module_ext_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute extended import graph metrics for modules.

    Returns
    -------
    ComputationResult
        Success result after computing extended module metrics.
    """
    # Get import graph
    if ctx.resources is not None and ctx.resources.has(GraphResource.RESOURCE_NAME):
        graph_resource = ctx.require(GraphResource)
        import_graph = graph_resource.import_graph()
    elif ctx.engine is not None:
        import_graph = ctx.engine.import_graph()
    else:
        return ComputationResult(success=False, message="No graph engine available")

    # Compute coupling metrics
    coupling_metrics = coupling.compute_coupling(import_graph)

    # Compute SCC with condensation for layers
    scc_result = components.find_strongly_connected(import_graph, compute_condensation=True)
    layers = components.condensation_layers(import_graph, scc_result)

    # Build and persist rows
    now = datetime.now(tz=UTC)
    rows = [
        (
            module,
            ctx.repo,
            ctx.commit,
            metrics.afferent,
            metrics.efferent,
            metrics.instability,
            layers.get(module, 0),
            scc_result.node_to_component.get(module, -1),
            now,
        )
        for module, metrics in coupling_metrics.items()
    ]

    if rows:
        from codeintel.ingestion.common import run_batch  # noqa: PLC0415

        # Get gateway via resource injection or fallback
        if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
            storage = ctx.require(StorageResource)
            gateway = storage.gateway
        else:
            gateway = ctx.gateway

        run_batch(
            gateway,
            "analytics.graph_metrics_modules_ext",
            list(rows),
            delete_params=[ctx.repo, ctx.commit],
            scope="graph_metrics_modules_ext",
        )

    return ComputationResult.ok(row_counts={"analytics.graph_metrics_modules_ext": len(rows)})


# =============================================================================
# Plugin Definitions (factory pattern - ~5 lines each)
# =============================================================================

core_graph_metrics_plugin = make_metric_plugin(
    name="core_graph_metrics",
    computation=_compute_core_graph_metrics,
    stage="core",
    depends_on=("callgraph_builder", "import_graph_builder"),
    provides=("core_metrics",),
    produces_tables=(
        "analytics.graph_metrics_functions",
        "analytics.graph_metrics_modules",
    ),
    requires_graphs=(GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH),
)

function_ext_metrics_plugin = make_metric_plugin(
    name="graph_metrics_functions_ext",
    computation=_compute_function_ext_metrics,
    stage="core",
    depends_on=("callgraph_builder",),
    provides=("function_ext_metrics",),
    produces_tables=("analytics.graph_metrics_functions_ext",),
    requires_graphs=(GraphKind.CALL_GRAPH,),
)

module_ext_metrics_plugin = make_metric_plugin(
    name="graph_metrics_modules_ext",
    computation=_compute_module_ext_metrics,
    stage="core",
    depends_on=("import_graph_builder",),
    provides=("module_ext_metrics",),
    produces_tables=("analytics.graph_metrics_modules_ext",),
    requires_graphs=(GraphKind.IMPORT_GRAPH,),
)


# =============================================================================
# Backward-compatible getters
# =============================================================================


def get_core_graph_metrics_plugin() -> GraphPluginProtocol:
    """Return the core graph metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured core graph metrics plugin.
    """
    return core_graph_metrics_plugin


def get_function_ext_metrics_plugin() -> GraphPluginProtocol:
    """Return the function ext metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured extended function metrics plugin.
    """
    return function_ext_metrics_plugin


def get_module_ext_metrics_plugin() -> GraphPluginProtocol:
    """Return the module ext metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured extended module metrics plugin.
    """
    return module_ext_metrics_plugin


__all__ = [
    "core_graph_metrics_plugin",
    "function_ext_metrics_plugin",
    "get_core_graph_metrics_plugin",
    "get_function_ext_metrics_plugin",
    "get_module_ext_metrics_plugin",
    "module_ext_metrics_plugin",
]
