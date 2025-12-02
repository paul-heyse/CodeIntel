"""Graph plugin core infrastructure.

This package provides the unified plugin protocol, execution context,
and registry for graph plugins, supporting both graph builders and
graph metric plugins without any dependency on the analytics subsystem.

Key Components
--------------
- GraphPluginProtocol: Unified interface for all graph plugins
- GraphExecutionContext: Execution context providing storage and engine access
- GraphPluginResult: Standard result type for plugin execution
- GraphPluginRegistry: Central registry with dependency resolution
- graph_plugin: Decorator for defining graph plugins from functions
- make_metric_plugin, make_builder_plugin: Factory functions for minimal boilerplate

Factory Pattern Example
-----------------------
```python
from codeintel.graphs.core import make_metric_plugin, ComputationResult
from codeintel.graphs.core.context import GraphExecutionContext


def compute_my_metrics(ctx: GraphExecutionContext) -> ComputationResult:
    # Compute metrics
    return ComputationResult.ok(row_counts={"analytics.my_metrics": 100})


my_metrics = make_metric_plugin(
    name="my_metrics",
    computation=compute_my_metrics,
    stage="core",
    depends_on=("callgraph_builder",),
)
```

Decorator Example
-----------------
```python
from codeintel.graphs.core import graph_plugin, GraphExecutionContext, GraphPluginResult


@graph_plugin(
    name="my_builder",
    description="Build a custom graph structure.",
    kind="builder",
    stage="edges",
    produces_tables=("graph.my_edges",),
)
def my_builder_plugin(ctx: GraphExecutionContext) -> GraphPluginResult:
    # Build graph and persist to database
    return GraphPluginResult.ok(row_counts={"graph.my_edges": 100})
```
"""

from codeintel.graphs.core.adapters import (
    adapt_legacy_computation,
    adapt_simple,
    adapt_with_row_counts,
)
from codeintel.graphs.core.computation import (
    ComputationFn,
    ComputationResult,
)
from codeintel.graphs.core.context import (
    GraphExecutionContext,
    GraphRuntimeScratch,
)
from codeintel.graphs.core.factories import (
    FactoryPlugin,
    make_builder_plugin,
    make_graph_plugin,
    make_metric_plugin,
    make_validation_plugin,
)
from codeintel.graphs.core.protocol import (
    DEFAULT_BUILDER_PLUGINS,
    DEFAULT_GRAPH_PLUGINS,
    DEFAULT_METRIC_PLUGINS,
    DEFAULT_VALIDATION_PLUGINS,
    FunctionalGraphPlugin,
    GraphPluginIsolation,
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginPlan,
    GraphPluginProtocol,
    GraphPluginResourceHints,
    GraphPluginSeverity,
    GraphPluginSkip,
    GraphPluginStage,
    graph_plugin,
)
from codeintel.graphs.core.registry import (
    GraphPluginRegistry,
    get_graph_registry,
    list_graph_plugins,
    plan_graph_plugins,
    register_graph_plugin,
    unregister_graph_plugin,
)
from codeintel.graphs.core.result import (
    GraphPluginResult,
    GraphPluginRunRecord,
    GraphPluginStatus,
)

__all__ = [
    "DEFAULT_BUILDER_PLUGINS",
    "DEFAULT_GRAPH_PLUGINS",
    "DEFAULT_METRIC_PLUGINS",
    "DEFAULT_VALIDATION_PLUGINS",
    "ComputationFn",
    "ComputationResult",
    "FactoryPlugin",
    "FunctionalGraphPlugin",
    "GraphExecutionContext",
    "GraphPluginIsolation",
    "GraphPluginKind",
    "GraphPluginMetadata",
    "GraphPluginPlan",
    "GraphPluginProtocol",
    "GraphPluginRegistry",
    "GraphPluginResourceHints",
    "GraphPluginResult",
    "GraphPluginRunRecord",
    "GraphPluginSeverity",
    "GraphPluginSkip",
    "GraphPluginStage",
    "GraphPluginStatus",
    "GraphRuntimeScratch",
    "adapt_legacy_computation",
    "adapt_simple",
    "adapt_with_row_counts",
    "get_graph_registry",
    "graph_plugin",
    "list_graph_plugins",
    "make_builder_plugin",
    "make_graph_plugin",
    "make_metric_plugin",
    "make_validation_plugin",
    "plan_graph_plugins",
    "register_graph_plugin",
    "unregister_graph_plugin",
]
