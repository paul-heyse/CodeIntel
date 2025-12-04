"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:
- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin
"""

import importlib

from codeintel.graphs.core import (
    GraphPluginExecutionContext,
    GraphPluginMetadata,
    GraphPluginPlan,
    GraphPluginProtocol,
    PluginResult,
    graph_plugin,
)
from codeintel.graphs.plugins.validation import (
    GraphValidationPlugin,
    get_graph_validation_plugin,
)

# Backward-compatible alias
GraphExecutionContext = GraphPluginExecutionContext


def load_builtin_plugins() -> None:
    """Import built-in plugins to ensure registration side effects run once."""
    importlib.import_module("codeintel.graphs.plugins.builders")
    importlib.import_module("codeintel.graphs.plugins.metrics")
    importlib.import_module("codeintel.graphs.plugins.validation")


# Eagerly load built-in plugins so registry has them by default.
load_builtin_plugins()


__all__ = [
    "GraphExecutionContext",
    "GraphPluginExecutionContext",
    "GraphPluginMetadata",
    "GraphPluginPlan",
    "GraphPluginProtocol",
    "GraphValidationPlugin",
    "PluginResult",
    "get_graph_validation_plugin",
    "graph_plugin",
    "load_builtin_plugins",
]
