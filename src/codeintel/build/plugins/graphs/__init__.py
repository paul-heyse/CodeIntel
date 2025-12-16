"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:

- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

All plugins implement the TargetPlugin protocol for the build system.
Plugins are registered with the build registry in codeintel.build.unified_registry.

Phase 3: All graph plugins migrated to native Hamilton modules.
The actual plugin classes are now stub classes that emit deprecation warnings.
Use the native modules in codeintel.build.hamilton.native.graphs instead.
"""

from __future__ import annotations

from codeintel.build.plugins.graphs.stubs import (
    CallGraphPlugin,
    CfgDfgPlugin,
    CoreMetricsPlugin,
    GoidBuilderPlugin,
    GraphValidationPlugin,
    ImportGraphPlugin,
    SymbolUsesPlugin,
)

__all__ = [
    "CallGraphPlugin",
    "CfgDfgPlugin",
    "CoreMetricsPlugin",
    "GoidBuilderPlugin",
    "GraphValidationPlugin",
    "ImportGraphPlugin",
    "SymbolUsesPlugin",
]
