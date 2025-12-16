"""Graph plugins for building and analyzing code graphs.

This package contains all graph plugins organized by category:

- builders: Graph construction plugins (callgraph, CFG, import graph, etc.)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

All plugins implement the TargetPlugin protocol for the build system.
Plugins are registered with the build registry in codeintel.build.unified_registry.

.. deprecated::
    All graph plugins have been migrated to native Hamilton modules.
    Use the native modules in ``codeintel.build.hamilton.native.graphs`` instead.
"""

from __future__ import annotations

from codeintel.build.plugins.graphs.builders import (
    CallGraphPlugin,
    CfgDfgPlugin,
    GoidBuilderPlugin,
    ImportGraphPlugin,
    SymbolUsesPlugin,
)

__all__ = [
    "CallGraphPlugin",
    "CfgDfgPlugin",
    "GoidBuilderPlugin",
    "ImportGraphPlugin",
    "SymbolUsesPlugin",
]
