"""Graph plugin core infrastructure.

This package provides the unified plugin protocol, execution context,
and registry for graph plugins, supporting both graph builders and
graph metric plugins without any dependency on the analytics subsystem.

Key Components
--------------
- GraphPluginProtocol: Unified interface for all graph plugins
- GraphPluginExecutionContext: Execution context providing storage and engine access
- PluginResult: Standard result type for plugin execution
- GraphPluginRegistry: Central registry with dependency resolution
"""

from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.plugins.types.protocol import (
    PluginIsolation,
    PluginMetadata,
    PluginResourceHints,
    PluginSeverity,
)
from codeintel.core.plugins.types.result import (
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)
from codeintel.graphs.core.context import (
    GraphPluginExecutionContext,
    GraphPluginExecutionContextBuilder,
)
from codeintel.graphs.core.protocol import (
    DEFAULT_BUILDER_PLUGINS,
    DEFAULT_GRAPH_PLUGINS,
    DEFAULT_METRIC_PLUGINS,
    DEFAULT_VALIDATION_PLUGINS,
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginPlan,
    GraphPluginProtocol,
    GraphPluginSkip,
    GraphPluginStage,
    create_graph_metadata,
)
from codeintel.graphs.core.registry import (
    GraphPluginRegistry,
    get_graph_registry,
    list_graph_plugins,
    plan_graph_plugins,
    register_graph_plugin,
    reset_graph_registry,
    unregister_graph_plugin,
)

__all__ = [
    "DEFAULT_BUILDER_PLUGINS",
    "DEFAULT_GRAPH_PLUGINS",
    "DEFAULT_METRIC_PLUGINS",
    "DEFAULT_VALIDATION_PLUGINS",
    "GraphPluginExecutionContext",
    "GraphPluginExecutionContextBuilder",
    "GraphPluginKind",
    "GraphPluginMetadata",
    "GraphPluginPlan",
    "GraphPluginProtocol",
    "GraphPluginRegistry",
    "GraphPluginSkip",
    "GraphPluginStage",
    "PluginExecutionRecord",
    "PluginIsolation",
    "PluginMetadata",
    "PluginResourceHints",
    "PluginResult",
    "PluginScratch",
    "PluginSeverity",
    "PluginStatus",
    "create_graph_metadata",
    "get_graph_registry",
    "list_graph_plugins",
    "plan_graph_plugins",
    "register_graph_plugin",
    "reset_graph_registry",
    "unregister_graph_plugin",
]
