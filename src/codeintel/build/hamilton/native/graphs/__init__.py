"""Native Hamilton graphs package.

This package contains native Hamilton implementations for graph-related targets,
including derived views and graph metrics.

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.call_graph import (
    CallGraphExtractResult,
    t__call_graph,
    t__call_graph__extract,
)
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    CFGExtractResult,
    DFGExtractResult,
    FunctionInfo,
    t__cfg,
    t__cfg__extract,
    t__dfg,
    t__dfg__extract,
)
from codeintel.build.hamilton.native.graphs.metrics_targets import (
    GraphMetricsComputeResult,
    GraphValidationResult,
    t__graph_metrics,
    t__graph_metrics__compute,
    t__graph_validation,
    t__graph_validation__check,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    ImportGraphExtractResult,
    t__import_graph,
    t__import_graph__extract,
)
from codeintel.build.hamilton.native.graphs.support_targets import (
    GoidExtractionInputs,
    GoidExtractResult,
    SymbolUsesExtractResult,
    call_graph_depth_stats,
    call_graph_function_call_counts,
    t__call_graph_views,
    t__goids,
    t__goids__extract,
    t__symbol_uses,
    t__symbol_uses__extract,
)

__all__ = [
    "CFGExtractResult",
    "CallGraphExtractResult",
    "DFGExtractResult",
    "FunctionInfo",
    "GoidExtractResult",
    "GoidExtractionInputs",
    "GraphMetricsComputeResult",
    "GraphValidationResult",
    "ImportGraphExtractResult",
    "SymbolUsesExtractResult",
    "call_graph_depth_stats",
    "call_graph_function_call_counts",
    "t__call_graph",
    "t__call_graph__extract",
    "t__call_graph_views",
    "t__cfg",
    "t__cfg__extract",
    "t__dfg",
    "t__dfg__extract",
    "t__goids",
    "t__goids__extract",
    "t__graph_metrics",
    "t__graph_metrics__compute",
    "t__graph_validation",
    "t__graph_validation__check",
    "t__import_graph",
    "t__import_graph__extract",
    "t__symbol_uses",
    "t__symbol_uses__extract",
]
