"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:

- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

All plugins implement the TargetPlugin protocol for the build system.
Plugins are registered with the build registry in codeintel.build.plugin_registry.
"""

from __future__ import annotations

from codeintel.build.plugins.graphs.builders import (
    CallGraphPlugin,
    CfgDfgPlugin,
    GoidBuilderPlugin,
    ImportGraphPlugin,
    SymbolUsesPlugin,
)
from codeintel.build.plugins.graphs.metrics import CoreMetricsPlugin, SecondaryMetricsPlugin
from codeintel.build.plugins.graphs.validation import GraphValidationPlugin

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
