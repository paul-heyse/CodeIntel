"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:
- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin
"""

from codeintel.graphs.core import (
    GraphExecutionContext,
    GraphPluginMetadata,
    GraphPluginPlan,
    GraphPluginProtocol,
    GraphPluginResult,
    graph_plugin,
)

# Import subpackages to register plugins
from codeintel.graphs.plugins import builders, metrics  # noqa: F401
from codeintel.graphs.plugins.validation import (
    GraphValidationPlugin,
    get_graph_validation_plugin,
)

__all__ = [
    "GraphExecutionContext",
    "GraphPluginMetadata",
    "GraphPluginPlan",
    "GraphPluginProtocol",
    "GraphPluginResult",
    "GraphValidationPlugin",
    "get_graph_validation_plugin",
    "graph_plugin",
]
