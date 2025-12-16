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
from codeintel.build.hamilton.native.graphs.call_graph_views import (
    call_graph_depth_stats,
    call_graph_function_call_counts,
    t__call_graph_views,
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
from codeintel.build.hamilton.native.graphs.goids import (
    GoidExtractionContext,
    GoidExtractResult,
    t__goids,
    t__goids__extract,
)
from codeintel.build.hamilton.native.graphs.graph_metrics import (
    GraphMetricsComputeResult,
    t__graph_metrics,
    t__graph_metrics__compute,
)
from codeintel.build.hamilton.native.graphs.graph_validation import (
    GraphValidationResult,
    t__graph_validation,
    t__graph_validation__check,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    ImportGraphExtractResult,
    t__import_graph,
    t__import_graph__extract,
)
from codeintel.build.hamilton.native.graphs.symbol_uses import (
    SymbolUsesExtractResult,
    t__symbol_uses,
    t__symbol_uses__extract,
)

__all__ = [
    # call_graph
    "CallGraphExtractResult",
    "t__call_graph",
    "t__call_graph__extract",
    # call_graph_views
    "call_graph_depth_stats",
    "call_graph_function_call_counts",
    "t__call_graph_views",
    # cfg_dfg
    "CFGExtractResult",
    "DFGExtractResult",
    "FunctionInfo",
    "t__cfg",
    "t__cfg__extract",
    "t__dfg",
    "t__dfg__extract",
    # goids
    "GoidExtractionContext",
    "GoidExtractResult",
    "t__goids",
    "t__goids__extract",
    # graph_metrics
    "GraphMetricsComputeResult",
    "t__graph_metrics",
    "t__graph_metrics__compute",
    # graph_validation
    "GraphValidationResult",
    "t__graph_validation",
    "t__graph_validation__check",
    # import_graph
    "ImportGraphExtractResult",
    "t__import_graph",
    "t__import_graph__extract",
    # symbol_uses
    "SymbolUsesExtractResult",
    "t__symbol_uses",
    "t__symbol_uses__extract",
]
