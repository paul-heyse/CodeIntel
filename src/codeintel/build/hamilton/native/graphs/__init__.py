"""Native Hamilton graphs package (relation-first)."""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.call_graph import (
    CALL_GRAPH_EDGES_TABLE_KEY,
    CALL_GRAPH_NODES_TABLE_KEY,
    call_graph_edges_empty,
    call_graph_edges_existing,
    call_graph_nodes_empty,
    call_graph_nodes_existing,
)
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    CFG_BLOCKS_TABLE_KEY,
    CFG_EDGES_TABLE_KEY,
    DFG_EDGES_TABLE_KEY,
    cfg_blocks_empty,
    cfg_blocks_existing,
    cfg_edges_empty,
    cfg_edges_existing,
    dfg_edges_empty,
    dfg_edges_existing,
)
from codeintel.build.hamilton.native.graphs.graph_targets import (
    CALL_GRAPH_TARGET_NAME,
    CFG_TARGET_NAME,
    DFG_TARGET_NAME,
    IMPORT_GRAPH_TARGET_NAME,
    t__call_graph,
    t__cfg,
    t__dfg,
    t__import_graph,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    IMPORT_GRAPH_EDGES_TABLE_KEY,
    IMPORT_MODULES_TABLE_KEY,
    import_graph_edges_empty,
    import_graph_edges_existing,
    import_modules_empty,
    import_modules_existing,
)
from codeintel.build.hamilton.native.graphs.variants import (
    call_graph_edges,
    call_graph_nodes,
    cfg_blocks,
    cfg_edges,
    dfg_edges,
    import_graph_edges,
    import_modules,
)

__all__ = [
    "CALL_GRAPH_EDGES_TABLE_KEY",
    "CALL_GRAPH_NODES_TABLE_KEY",
    "CALL_GRAPH_TARGET_NAME",
    "CFG_BLOCKS_TABLE_KEY",
    "CFG_EDGES_TABLE_KEY",
    "CFG_TARGET_NAME",
    "DFG_EDGES_TABLE_KEY",
    "DFG_TARGET_NAME",
    "IMPORT_GRAPH_EDGES_TABLE_KEY",
    "IMPORT_GRAPH_TARGET_NAME",
    "IMPORT_MODULES_TABLE_KEY",
    "call_graph_edges",
    "call_graph_edges_empty",
    "call_graph_edges_existing",
    "call_graph_nodes",
    "call_graph_nodes_empty",
    "call_graph_nodes_existing",
    "cfg_blocks",
    "cfg_blocks_empty",
    "cfg_blocks_existing",
    "cfg_edges",
    "cfg_edges_empty",
    "cfg_edges_existing",
    "dfg_edges",
    "dfg_edges_empty",
    "dfg_edges_existing",
    "import_graph_edges",
    "import_graph_edges_empty",
    "import_graph_edges_existing",
    "import_modules",
    "import_modules_empty",
    "import_modules_existing",
    "t__call_graph",
    "t__cfg",
    "t__dfg",
    "t__import_graph",
]
