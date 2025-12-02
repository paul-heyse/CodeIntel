"""Modernized graph plugin infrastructure.

This module provides the modernized graph plugin protocol and registry,
aligned with the new analytics plugin architecture while maintaining
graph-specific functionality.
"""

from __future__ import annotations

from codeintel.analytics.graphs.core.protocol import (
    DEFAULT_GRAPH_METRIC_PLUGINS,
    GraphMetricPluginMetadata,
    GraphMetricPluginPlan,
    GraphMetricPluginSkip,
    GraphMetricResourceHints,
    GraphPluginProtocol,
    GraphPluginResult,
    GraphRuntimeScratch,
    graph_plugin,
)
from codeintel.analytics.graphs.core.registry import (
    GraphPluginRegistry,
    get_graph_registry,
    list_graph_plugins,
    plan_graph_plugins,
    register_graph_plugin,
)

__all__ = [
    "DEFAULT_GRAPH_METRIC_PLUGINS",
    "GraphMetricPluginMetadata",
    "GraphMetricPluginPlan",
    "GraphMetricPluginSkip",
    "GraphMetricResourceHints",
    "GraphPluginProtocol",
    "GraphPluginRegistry",
    "GraphPluginResult",
    "GraphRuntimeScratch",
    "get_graph_registry",
    "graph_plugin",
    "list_graph_plugins",
    "plan_graph_plugins",
    "register_graph_plugin",
]
