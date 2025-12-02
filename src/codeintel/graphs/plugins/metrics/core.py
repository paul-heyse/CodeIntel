"""Core graph metrics plugins using factory pattern.

This module provides the core graph metrics plugins, wrapping the existing
analytics graph_metrics functionality using the factory pattern for minimal
boilerplate.
"""

from __future__ import annotations

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps, compute_graph_metrics
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.config.primitives import GraphBackendConfig
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    make_metric_plugin,
)
from codeintel.graphs.engine import GraphKind

# =============================================================================
# Computation Functions (standardized signature)
# =============================================================================


def _compute_core_graph_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """Compute core function/module graph metrics (centrality, neighbors, components)."""
    cfg = GraphMetricsStepConfig(snapshot=ctx.snapshot)

    runtime = resolve_graph_runtime(
        ctx.gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )

    deps = GraphMetricsDeps(
        catalog_provider=ctx.catalog_provider,
        runtime=runtime,
        analytics_context=None,
        filters=None,
    )
    compute_graph_metrics(ctx.gateway, cfg, deps=deps)
    return ComputationResult.ok()


def _compute_function_ext_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """Compute extended call graph metrics for functions."""
    runtime = resolve_graph_runtime(
        ctx.gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )

    compute_graph_metrics_functions_ext(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=runtime,
        filters=None,
    )
    return ComputationResult.ok()


def _compute_module_ext_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """Compute extended import graph metrics for modules."""
    runtime = resolve_graph_runtime(
        ctx.gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )

    compute_graph_metrics_modules_ext(
        ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        runtime=runtime,
        filters=None,
    )
    return ComputationResult.ok()


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


def get_core_graph_metrics_plugin() -> object:
    """Return the core graph metrics plugin instance."""
    return core_graph_metrics_plugin


def get_function_ext_metrics_plugin() -> object:
    """Return the function ext metrics plugin instance."""
    return function_ext_metrics_plugin


def get_module_ext_metrics_plugin() -> object:
    """Return the module ext metrics plugin instance."""
    return module_ext_metrics_plugin


# Legacy class aliases for backward compatibility
CoreGraphMetricsPlugin = type(core_graph_metrics_plugin)
FunctionExtMetricsPlugin = type(function_ext_metrics_plugin)
ModuleExtMetricsPlugin = type(module_ext_metrics_plugin)


__all__ = [
    "CoreGraphMetricsPlugin",
    "FunctionExtMetricsPlugin",
    "ModuleExtMetricsPlugin",
    "core_graph_metrics_plugin",
    "function_ext_metrics_plugin",
    "get_core_graph_metrics_plugin",
    "get_function_ext_metrics_plugin",
    "get_module_ext_metrics_plugin",
    "module_ext_metrics_plugin",
]
