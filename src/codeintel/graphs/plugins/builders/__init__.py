"""Graph builder plugins.

This package contains plugins that construct graph structures from parsed code:
- goid: GOID builder for entity identification
- callgraph: Call graph construction
- import_graph: Import/dependency graph construction
- cfg_dfg: Control-flow and data-flow graph construction
- symbol_uses: Symbol use edge construction

All plugins use the hexagonal architecture's resource injection pattern
via ctx.require() with fallback to direct context properties.
"""

# Plugins are registered when their modules are imported
# Import them here to ensure registration at package load time
from codeintel.graphs.plugins.builders.callgraph import (
    callgraph_builder_plugin,
    get_callgraph_builder_plugin,
)
from codeintel.graphs.plugins.builders.cfg_dfg import (
    cfg_dfg_builder_plugin,
    get_cfg_dfg_builder_plugin,
)
from codeintel.graphs.plugins.builders.goid import (
    get_goid_builder_plugin,
    goid_builder_plugin,
)
from codeintel.graphs.plugins.builders.import_graph import (
    get_import_graph_builder_plugin,
    import_graph_builder_plugin,
)

__all__ = [
    "callgraph_builder_plugin",
    "cfg_dfg_builder_plugin",
    "get_callgraph_builder_plugin",
    "get_cfg_dfg_builder_plugin",
    "get_goid_builder_plugin",
    "get_import_graph_builder_plugin",
    "goid_builder_plugin",
    "import_graph_builder_plugin",
]
