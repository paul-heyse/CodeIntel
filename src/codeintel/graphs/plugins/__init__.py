"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:
- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

All plugins implement the TargetPlugin protocol for the build system.
Adapters are used to register them with the GraphPluginRegistry.
"""

from __future__ import annotations

import logging

from codeintel.graphs.core.adapters import TargetPluginAdapter
from codeintel.graphs.core.registry import get_graph_registry
from codeintel.graphs.plugins.builders import (
    CallGraphPlugin,
    CfgDfgPlugin,
    GoidBuilderPlugin,
    ImportGraphPlugin,
    SymbolUsesPlugin,
)
from codeintel.graphs.plugins.metrics import CoreMetricsPlugin, SecondaryMetricsPlugin
from codeintel.graphs.plugins.validation import GraphValidationPlugin

_log = logging.getLogger(__name__)

_PLUGIN_STATE = {"registered": False}


def load_builtin_plugins() -> None:
    """Register adapted TargetPlugins with the GraphPluginRegistry.

    This function wraps each TargetPlugin implementation with a
    TargetPluginAdapter and registers it with the global graph registry.
    The adapters provide the GraphPluginProtocol interface required by
    the registry.

    This function is idempotent - calling it multiple times has no effect
    after the first successful registration.
    """
    if _PLUGIN_STATE["registered"]:
        return

    registry = get_graph_registry()

    target_plugins = [
        GoidBuilderPlugin(),
        CallGraphPlugin(),
        ImportGraphPlugin(),
        CfgDfgPlugin(),
        SymbolUsesPlugin(),
        CoreMetricsPlugin(),
        SecondaryMetricsPlugin(),
        GraphValidationPlugin(),
    ]

    for plugin in target_plugins:
        try:
            adapted = TargetPluginAdapter(plugin)
            registry.register(adapted)
            _log.debug("Registered graph plugin adapter: %s", plugin.plugin_name)
        except ValueError:
            _log.debug("Plugin already registered: %s", plugin.plugin_name)

    _PLUGIN_STATE["registered"] = True


load_builtin_plugins()


__all__ = [
    "GraphValidationPlugin",
    "load_builtin_plugins",
]
