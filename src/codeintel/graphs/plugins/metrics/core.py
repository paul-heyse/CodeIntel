"""Core graph metrics plugins using factory pattern.

This module provides the core graph metrics plugins using the hexagonal
architecture's compute layer for pure metric calculations.

Uses resource injection pattern via ctx.require_graphs() for
graph access.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from codeintel.graphs.compute.metrics import centrality, components
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_metric_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import StorageResource
from codeintel.ingestion.services.storage import IngestStorageService

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
    # Get graphs via resource injection
    try:
        graph_resource = ctx.require_graphs()
    except RuntimeError as exc:
        log.warning("No graph engine available for metrics computation: %s", exc)
        return ComputationResult(success=False, message=str(exc))

    call_graph = graph_resource.call_graph()
    import_graph = graph_resource.import_graph()

    # Compute centrality metrics using pure compute functions
    func_centralities = centrality.compute_all_centralities(call_graph)
    module_centralities = centrality.compute_all_centralities(import_graph)

    # Compute SCC for both graphs (need condensation for layer computation)
    func_sccs = components.find_strongly_connected(call_graph, compute_condensation=True)
    module_sccs = components.find_strongly_connected(import_graph, compute_condensation=True)

    # Compute topological layers based on condensation
    func_layers = components.condensation_layers(call_graph, func_sccs)
    module_layers = components.condensation_layers(import_graph, module_sccs)

    # Build set of nodes in non-trivial cycles (SCC size > 1)
    func_cycle_nodes: set[object] = set()
    for comp in func_sccs.components:
        if comp.size > 1:
            func_cycle_nodes.update(comp.nodes)

    module_cycle_nodes: set[object] = set()
    for comp in module_sccs.components:
        if comp.size > 1:
            module_cycle_nodes.update(comp.nodes)

    # Persist function metrics
    # Columns: repo, commit, function_goid_h128, call_fan_in, call_fan_out,
    #          call_in_degree, call_out_degree, call_pagerank, call_betweenness,
    #          call_closeness, call_cycle_member, call_cycle_id, call_layer, created_at
    now = datetime.now(tz=UTC)
    func_rows = [
        (
            ctx.repo,
            ctx.commit,
            goid,
            metrics.in_degree,  # call_fan_in
            metrics.out_degree,  # call_fan_out
            metrics.in_degree,  # call_in_degree
            metrics.out_degree,  # call_out_degree
            metrics.pagerank,  # call_pagerank
            metrics.betweenness,  # call_betweenness
            metrics.closeness,  # call_closeness
            goid in func_cycle_nodes,  # call_cycle_member
            func_sccs.node_to_component.get(goid),  # call_cycle_id
            func_layers.get(goid),  # call_layer
            now,  # created_at
        )
        for goid, metrics in func_centralities.items()
    ]

    if func_rows:
        _persist_function_metrics(ctx, func_rows)

    # Persist module metrics (see GRAPH_METRICS_MODULES_COLUMNS for column order)
    module_rows = [
        (
            ctx.repo,
            ctx.commit,
            node,
            metrics.in_degree,  # import_fan_in
            metrics.out_degree,  # import_fan_out
            metrics.in_degree,  # import_in_degree
            metrics.out_degree,  # import_out_degree
            metrics.pagerank,  # import_pagerank
            metrics.betweenness,  # import_betweenness
            metrics.closeness,  # import_closeness
            node in module_cycle_nodes,  # import_cycle_member
            module_sccs.node_to_component.get(node),  # import_cycle_id
            module_layers.get(node),  # import_layer
            0,  # symbol_fan_in (computed elsewhere, defaulting to 0)
            0,  # symbol_fan_out (computed elsewhere, defaulting to 0)
            now,  # created_at
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
    # Get gateway via resource injection or fallback
    if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.require(StorageResource)
        gateway = storage.gateway
    else:
        gateway = ctx.gateway

    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
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
    # Get gateway via resource injection or fallback
    if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.require(StorageResource)
        gateway = storage.gateway
    else:
        gateway = ctx.gateway

    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
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
    # Get call graph via resource injection
    try:
        graph_resource = ctx.require_graphs()
    except RuntimeError as exc:
        return ComputationResult(success=False, message=str(exc))

    call_graph = graph_resource.call_graph()

    # Compute centrality metrics for extended metrics
    func_centralities = centrality.compute_all_centralities(call_graph)

    # Compute SCC with condensation
    scc_result = components.find_strongly_connected(call_graph, compute_condensation=True)

    # Build SCC size lookup
    scc_sizes = {comp.component_id: comp.size for comp in scc_result.components}

    # Find weakly connected components for component ID/size
    wcc_result = components.find_weakly_connected(call_graph)
    wcc_node_to_comp: dict[object, int] = {}
    wcc_comp_sizes: dict[int, int] = {}
    for comp in wcc_result:
        wcc_comp_sizes[comp.component_id] = comp.size
        for node in comp.nodes:
            wcc_node_to_comp[node] = comp.component_id

    # Build and persist rows
    # Columns: repo, commit, function_goid_h128, call_betweenness, call_closeness,
    #          call_eigenvector, call_harmonic, call_core_number, call_clustering_coeff,
    #          call_triangle_count, call_is_articulation, call_articulation_impact,
    #          call_is_bridge_endpoint, call_component_id, call_component_size,
    #          call_scc_id, call_scc_size, call_ancestor_count, call_descendant_count,
    #          call_community_id, created_at
    now = datetime.now(tz=UTC)
    rows = [
        (
            ctx.repo,  # repo
            ctx.commit,  # commit
            goid,  # function_goid_h128
            metrics.betweenness,  # call_betweenness
            metrics.closeness,  # call_closeness
            0.0,  # call_eigenvector (placeholder)
            0.0,  # call_harmonic (placeholder)
            0,  # call_core_number (placeholder)
            0.0,  # call_clustering_coeff (placeholder)
            0,  # call_triangle_count (placeholder)
            False,  # call_is_articulation (placeholder)
            0.0,  # call_articulation_impact (placeholder)
            False,  # call_is_bridge_endpoint (placeholder)
            wcc_node_to_comp.get(goid, 0),  # call_component_id
            wcc_comp_sizes.get(wcc_node_to_comp.get(goid, 0), 0),  # call_component_size
            scc_result.node_to_component.get(goid, 0),  # call_scc_id
            scc_sizes.get(scc_result.node_to_component.get(goid, 0), 0),  # call_scc_size
            0,  # call_ancestor_count (placeholder)
            0,  # call_descendant_count (placeholder)
            0,  # call_community_id (placeholder)
            now,  # created_at
        )
        for goid, metrics in func_centralities.items()
    ]

    if rows:
        # Get gateway via resource injection or fallback
        if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
            storage = ctx.require(StorageResource)
            gateway = storage.gateway
        else:
            gateway = ctx.gateway

        storage_service = IngestStorageService.from_gateway(gateway)
        storage_service.run_batch(
            "analytics.graph_metrics_functions_ext",
            list(rows),
            delete_params=[ctx.repo, ctx.commit],
            scope="graph_metrics_functions_ext",
        )

    return ComputationResult.ok(row_counts={"analytics.graph_metrics_functions_ext": len(rows)})


def _compute_module_ext_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """Compute extended import graph metrics for modules.

    Returns
    -------
    ComputationResult
        Success result after computing extended module metrics.
    """
    # Get import graph via resource injection
    try:
        graph_resource = ctx.require_graphs()
    except RuntimeError as exc:
        return ComputationResult(success=False, message=str(exc))

    import_graph = graph_resource.import_graph()

    # Compute centrality metrics
    module_centralities = centrality.compute_all_centralities(import_graph)

    # Compute SCC with condensation
    scc_result = components.find_strongly_connected(import_graph, compute_condensation=True)

    # Build SCC size lookup
    scc_sizes = {comp.component_id: comp.size for comp in scc_result.components}

    # Find weakly connected components
    wcc_result = components.find_weakly_connected(import_graph)
    wcc_node_to_comp: dict[object, int] = {}
    wcc_comp_sizes: dict[int, int] = {}
    for comp in wcc_result:
        wcc_comp_sizes[comp.component_id] = comp.size
        for node in comp.nodes:
            wcc_node_to_comp[node] = comp.component_id

    # Build and persist rows (see GRAPH_METRICS_MODULES_EXT_COLUMNS for column order)
    now = datetime.now(tz=UTC)
    rows = [
        (
            ctx.repo,  # repo
            ctx.commit,  # commit
            module,  # module
            metrics.betweenness,  # import_betweenness
            metrics.closeness,  # import_closeness
            0.0,  # import_eigenvector (placeholder)
            0.0,  # import_harmonic (placeholder)
            0,  # import_k_core (placeholder)
            0.0,  # import_constraint (placeholder)
            0.0,  # import_effective_size (placeholder)
            0.0,  # import_rich_club (placeholder)
            0,  # import_shell_index (placeholder)
            0,  # import_community_id (placeholder)
            wcc_node_to_comp.get(module, 0),  # import_component_id
            wcc_comp_sizes.get(wcc_node_to_comp.get(module, 0), 0),  # import_component_size
            scc_result.node_to_component.get(module, 0),  # import_scc_id
            scc_sizes.get(scc_result.node_to_component.get(module, 0), 0),  # import_scc_size
            now,  # created_at
        )
        for module, metrics in module_centralities.items()
    ]

    if rows:
        # Get gateway via resource injection or fallback
        if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
            storage = ctx.require(StorageResource)
            gateway = storage.gateway
        else:
            gateway = ctx.gateway

        storage_service = IngestStorageService.from_gateway(gateway)
        storage_service.run_batch(
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
