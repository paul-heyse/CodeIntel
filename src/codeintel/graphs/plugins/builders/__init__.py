"""Graph builder plugins.

This package contains plugins that construct graph structures from parsed code:
- goid: GOID builder for entity identification
- callgraph: Call graph construction
- import_graph: Import/dependency graph construction
- cfg_dfg: Control-flow and data-flow graph construction
- symbol_uses: Symbol use edge construction

All plugins implement the TargetPlugin protocol for the build system.
"""

from codeintel.graphs.plugins.builders.callgraph import CallGraphPlugin
from codeintel.graphs.plugins.builders.cfg_dfg import CfgDfgPlugin
from codeintel.graphs.plugins.builders.goid import GoidBuilderPlugin
from codeintel.graphs.plugins.builders.import_graph import ImportGraphPlugin
from codeintel.graphs.plugins.builders.symbol_uses import SymbolUsesPlugin, build_scip_candidates

__all__ = [
    "CallGraphPlugin",
    "CfgDfgPlugin",
    "GoidBuilderPlugin",
    "ImportGraphPlugin",
    "SymbolUsesPlugin",
    "build_scip_candidates",
]
