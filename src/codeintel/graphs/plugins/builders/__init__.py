"""Graph builder plugins.

This package contains plugins that construct graph structures from parsed code:
- goid: GOID builder for entity identification
- callgraph: Call graph construction
- import_graph: Import/dependency graph construction
- cfg_dfg: Control-flow and data-flow graph construction
- symbol_uses: Symbol use edge construction
"""

# Plugins are registered when their modules are imported
# Import them here to ensure registration at package load time
from codeintel.graphs.plugins.builders.callgraph import (
    CallGraphBuilderPlugin,
    get_callgraph_builder_plugin,
)
from codeintel.graphs.plugins.builders.cfg_dfg import (
    CFGDFGBuilderPlugin,
    get_cfg_dfg_builder_plugin,
)
from codeintel.graphs.plugins.builders.goid import (
    GoidBuilderPlugin,
    get_goid_builder_plugin,
)
from codeintel.graphs.plugins.builders.import_graph import (
    ImportGraphBuilderPlugin,
    get_import_graph_builder_plugin,
)

__all__ = [
    "CFGDFGBuilderPlugin",
    "CallGraphBuilderPlugin",
    "GoidBuilderPlugin",
    "ImportGraphBuilderPlugin",
    "get_callgraph_builder_plugin",
    "get_cfg_dfg_builder_plugin",
    "get_goid_builder_plugin",
    "get_import_graph_builder_plugin",
]
