"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:
- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

All plugins implement the TargetPlugin protocol for the build system.
Plugins are registered with the build registry in codeintel.build.plugin_registry.
"""

from __future__ import annotations

from codeintel.graphs.plugins.builders import (
    CallGraphPlugin,
    CfgDfgPlugin,
    GoidBuilderPlugin,
    ImportGraphPlugin,
    SymbolUsesPlugin,
)
from codeintel.graphs.plugins.metrics import CoreMetricsPlugin, SecondaryMetricsPlugin
from codeintel.graphs.plugins.validation import GraphValidationPlugin

__all__ = [
    "CallGraphPlugin",
    "CfgDfgPlugin",
    "CoreMetricsPlugin",
    "GoidBuilderPlugin",
    "GraphValidationPlugin",
    "ImportGraphPlugin",
    "SecondaryMetricsPlugin",
    "SymbolUsesPlugin",
]
