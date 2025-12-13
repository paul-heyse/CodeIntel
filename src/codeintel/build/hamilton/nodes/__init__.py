"""Hamilton node modules for build execution.

This package contains the Hamilton node definitions that wrap existing
target plugins.

Modules
-------
dataset_nodes
    Dataset extraction nodes for lineage tracking.
node_factory
    Dynamic node generation from TargetGraph.
"""

from __future__ import annotations

from codeintel.build.hamilton.nodes.dataset_nodes import (
    DATASET_NODES,
    d__analytics__function_metrics,
    d__analytics__risk_factors,
    d__graph__call_graph_edges,
    d__graph__call_graph_nodes,
    extract_datasets_from_record,
)
from codeintel.build.hamilton.nodes.node_factory import (
    build_target_module,
    clear_generated_module_cache,
    get_generated_module,
)

__all__ = [
    "DATASET_NODES",
    "build_target_module",
    "clear_generated_module_cache",
    "d__analytics__function_metrics",
    "d__analytics__risk_factors",
    "d__graph__call_graph_edges",
    "d__graph__call_graph_nodes",
    "extract_datasets_from_record",
    "get_generated_module",
]
