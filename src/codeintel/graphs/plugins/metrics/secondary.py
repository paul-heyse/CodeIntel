"""Secondary graph metrics plugins using factory pattern.

This module provides additional graph metric plugins for CFG/DFG, test,
symbol, subsystem, config, and stats metrics using the factory pattern
for minimal boilerplate.

These plugins delegate computation to the analytics package (per architecture
decision Option B) and use resource injection pattern via ctx.require()
with fallback to direct context properties for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.graphs.graph_stats import compute_graph_stats
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.tests.graph_metrics import compute_test_graph_metrics
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_metric_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import StorageResource

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# =============================================================================
# Computation Functions (standardized signature)
# =============================================================================


def _get_gateway(ctx: GraphExecutionContext) -> StorageGateway:
    """Get storage gateway via resource injection or fallback.

    Parameters
    ----------
    ctx
        Graph execution context.

    Returns
    -------
    StorageGateway
        Storage gateway instance.
    """
    if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.require(StorageResource)
        return storage.gateway
    return ctx.gateway


def _compute_cfg_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute control-flow graph metrics for functions and blocks.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing CFG metrics.
    """
    gateway = _get_gateway(ctx)
    compute_cfg_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
    return ComputationResult.ok()


def _compute_dfg_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute data-flow graph metrics for functions and blocks.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing DFG metrics.
    """
    gateway = _get_gateway(ctx)
    compute_dfg_metrics(gateway, repo=ctx.repo, commit=ctx.commit)
    return ComputationResult.ok()


def _compute_test_graph_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute metrics over the test <-> function bipartite graph.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing test graph metrics.
    """
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

    gateway = _get_gateway(ctx)
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )
    compute_test_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit, runtime=runtime)
    return ComputationResult.ok()


def _compute_subsystem_graph_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute subsystem-level condensed import graph metrics.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing subsystem metrics.
    """
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

    gateway = _get_gateway(ctx)
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )
    compute_subsystem_graph_metrics(
        gateway, repo=ctx.repo, commit=ctx.commit, runtime=runtime, filters=None
    )
    return ComputationResult.ok()


def _compute_symbol_graph_metrics_modules(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute symbol graph metrics at the module level.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing module symbol metrics.
    """
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

    gateway = _get_gateway(ctx)
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )
    compute_symbol_graph_metrics_modules(gateway, repo=ctx.repo, commit=ctx.commit, runtime=runtime)
    return ComputationResult.ok()


def _compute_symbol_graph_metrics_functions(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute symbol graph metrics at the function level.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing function symbol metrics.
    """
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

    gateway = _get_gateway(ctx)
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )
    compute_symbol_graph_metrics_functions(
        gateway, repo=ctx.repo, commit=ctx.commit, runtime=runtime
    )
    return ComputationResult.ok()


def _compute_config_graph_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute config bipartite/projection graph metrics.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing config graph metrics.
    """
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

    gateway = _get_gateway(ctx)
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )
    compute_config_graph_metrics(gateway, repo=ctx.repo, commit=ctx.commit, runtime=runtime)
    return ComputationResult.ok()


def _compute_subsystem_agreement(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Check agreement between subsystem labels and import communities.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing subsystem agreement metrics.
    """
    gateway = _get_gateway(ctx)
    compute_subsystem_agreement(gateway, repo=ctx.repo, commit=ctx.commit)
    return ComputationResult.ok()


def _compute_graph_stats(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Compute global graph statistics for core graphs.

    Uses resource injection to access storage, with fallback to ctx.gateway.

    Returns
    -------
    ComputationResult
        Success result after computing graph statistics.
    """
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415

    gateway = _get_gateway(ctx)
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )
    compute_graph_stats(gateway, repo=ctx.repo, commit=ctx.commit, runtime=runtime)
    return ComputationResult.ok()


# =============================================================================
# Plugin Definitions (factory pattern - ~5 lines each)
# =============================================================================

cfg_metrics_plugin = make_metric_plugin(
    name="cfg_metrics",
    computation=_compute_cfg_metrics,
    stage="cfg",
    depends_on=("cfg_dfg_builder",),
    provides=("cfg_metrics",),
    requires_graphs=(GraphKind.CFG_GRAPH,),
)

dfg_metrics_plugin = make_metric_plugin(
    name="dfg_metrics",
    computation=_compute_dfg_metrics,
    stage="dfg",
    depends_on=("cfg_dfg_builder",),
    provides=("dfg_metrics",),
    requires_graphs=(GraphKind.CFG_GRAPH,),
)

test_graph_metrics_plugin = make_metric_plugin(
    name="test_graph_metrics",
    computation=_compute_test_graph_metrics,
    stage="test",
    depends_on=("callgraph_builder",),
    provides=("test_metrics",),
    produces_tables=(
        "analytics.test_graph_metrics_tests",
        "analytics.test_graph_metrics_functions",
    ),
    requires_graphs=(GraphKind.CALL_GRAPH,),
)

subsystem_graph_metrics_plugin = make_metric_plugin(
    name="subsystem_graph_metrics",
    computation=_compute_subsystem_graph_metrics,
    stage="subsystem",
    depends_on=("import_graph_builder",),
    provides=("subsystem_metrics",),
    produces_tables=("analytics.subsystem_graph_metrics",),
    requires_graphs=(GraphKind.IMPORT_GRAPH,),
)

symbol_graph_metrics_modules_plugin = make_metric_plugin(
    name="symbol_graph_metrics_modules",
    computation=_compute_symbol_graph_metrics_modules,
    stage="symbol",
    depends_on=("import_graph_builder",),
    provides=("symbol_metrics_modules",),
    produces_tables=("analytics.symbol_graph_metrics_modules",),
    requires_graphs=(GraphKind.IMPORT_GRAPH,),
)

symbol_graph_metrics_functions_plugin = make_metric_plugin(
    name="symbol_graph_metrics_functions",
    computation=_compute_symbol_graph_metrics_functions,
    stage="symbol",
    depends_on=("callgraph_builder",),
    provides=("symbol_metrics_functions",),
    produces_tables=("analytics.symbol_graph_metrics_functions",),
    requires_graphs=(GraphKind.CALL_GRAPH,),
)

config_graph_metrics_plugin = make_metric_plugin(
    name="config_graph_metrics",
    computation=_compute_config_graph_metrics,
    stage="config",
    depends_on=("import_graph_builder",),
    provides=("config_metrics",),
    produces_tables=(
        "analytics.config_graph_metrics_keys",
        "analytics.config_graph_metrics_modules",
        "analytics.config_projection_key_edges",
        "analytics.config_projection_module_edges",
    ),
    requires_graphs=(GraphKind.IMPORT_GRAPH,),
)

subsystem_agreement_plugin = make_metric_plugin(
    name="subsystem_agreement",
    computation=_compute_subsystem_agreement,
    stage="subsystem",
    depends_on=("subsystem_graph_metrics", "graph_metrics_modules_ext"),
    provides=("subsystem_agreement",),
    produces_tables=("analytics.subsystem_agreement",),
    requires_graphs=(GraphKind.IMPORT_GRAPH,),
)

graph_stats_plugin = make_metric_plugin(
    name="graph_stats",
    computation=_compute_graph_stats,
    stage="stats",
    depends_on=("callgraph_builder", "import_graph_builder"),
    provides=("graph_stats",),
    produces_tables=("analytics.graph_stats",),
    requires_graphs=(GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH),
)


# =============================================================================
# Plugin Getters
# =============================================================================


def get_cfg_metrics_plugin() -> GraphPluginProtocol:
    """Return the CFG metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured CFG metrics plugin.
    """
    return cfg_metrics_plugin


def get_dfg_metrics_plugin() -> GraphPluginProtocol:
    """Return the DFG metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured DFG metrics plugin.
    """
    return dfg_metrics_plugin


def get_test_graph_metrics_plugin() -> GraphPluginProtocol:
    """Return the test graph metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured test graph metrics plugin.
    """
    return test_graph_metrics_plugin


def get_subsystem_graph_metrics_plugin() -> GraphPluginProtocol:
    """Return the subsystem graph metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured subsystem graph metrics plugin.
    """
    return subsystem_graph_metrics_plugin


def get_symbol_graph_metrics_modules_plugin() -> GraphPluginProtocol:
    """Return the symbol graph metrics modules plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured symbol graph metrics modules plugin.
    """
    return symbol_graph_metrics_modules_plugin


def get_symbol_graph_metrics_functions_plugin() -> GraphPluginProtocol:
    """Return the symbol graph metrics functions plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured symbol graph metrics functions plugin.
    """
    return symbol_graph_metrics_functions_plugin


def get_config_graph_metrics_plugin() -> GraphPluginProtocol:
    """Return the config graph metrics plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured config graph metrics plugin.
    """
    return config_graph_metrics_plugin


def get_subsystem_agreement_plugin() -> GraphPluginProtocol:
    """Return the subsystem agreement plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured subsystem agreement plugin.
    """
    return subsystem_agreement_plugin


def get_graph_stats_plugin() -> GraphPluginProtocol:
    """Return the graph stats plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured graph stats plugin.
    """
    return graph_stats_plugin


__all__ = [
    "cfg_metrics_plugin",
    "config_graph_metrics_plugin",
    "dfg_metrics_plugin",
    "get_cfg_metrics_plugin",
    "get_config_graph_metrics_plugin",
    "get_dfg_metrics_plugin",
    "get_graph_stats_plugin",
    "get_subsystem_agreement_plugin",
    "get_subsystem_graph_metrics_plugin",
    "get_symbol_graph_metrics_functions_plugin",
    "get_symbol_graph_metrics_modules_plugin",
    "get_test_graph_metrics_plugin",
    "graph_stats_plugin",
    "subsystem_agreement_plugin",
    "subsystem_graph_metrics_plugin",
    "symbol_graph_metrics_functions_plugin",
    "symbol_graph_metrics_modules_plugin",
    "test_graph_metrics_plugin",
]
