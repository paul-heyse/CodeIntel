"""Native Hamilton graphs package.

This package contains native Hamilton implementations for graph-related targets,
including derived views and graph metrics.

Phase 3: Graphs domain consolidation with Hamilton-native validation.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.graphs.call_graph import (
    t__call_graph,
    t__call_graph__ingest,
    t__call_graph__run,
)
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    FunctionInfo,
    t__cfg,
    t__cfg__ingest,
    t__cfg__run,
    t__dfg,
    t__dfg__ingest,
    t__dfg__run,
)
from codeintel.build.hamilton.native.graphs.graph_targets import (
    GoidExtractionInputs,
    GraphMetricsToolOutput,
    GraphValidationIssue,
    GraphValidationToolOutput,
    call_graph_depth_stats,
    call_graph_function_call_counts,
    t__call_graph_views,
    t__goids,
    t__goids__ingest,
    t__goids__run,
    t__graph_metrics,
    t__graph_metrics__ingest,
    t__graph_metrics__run,
    t__graph_validation,
    t__graph_validation__ingest,
    t__graph_validation__run,
    t__symbol_uses,
    t__symbol_uses__ingest,
    t__symbol_uses__run,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    t__import_graph,
    t__import_graph__ingest,
    t__import_graph__run,
)

__all__ = [
    "FunctionInfo",
    "GoidExtractionInputs",
    "GraphMetricsToolOutput",
    "GraphValidationIssue",
    "GraphValidationToolOutput",
    "call_graph_depth_stats",
    "call_graph_function_call_counts",
    "t__call_graph",
    "t__call_graph__ingest",
    "t__call_graph__run",
    "t__call_graph_views",
    "t__cfg",
    "t__cfg__ingest",
    "t__cfg__run",
    "t__dfg",
    "t__dfg__ingest",
    "t__dfg__run",
    "t__goids",
    "t__goids__ingest",
    "t__goids__run",
    "t__graph_metrics",
    "t__graph_metrics__ingest",
    "t__graph_metrics__run",
    "t__graph_validation",
    "t__graph_validation__ingest",
    "t__graph_validation__run",
    "t__import_graph",
    "t__import_graph__ingest",
    "t__import_graph__run",
    "t__symbol_uses",
    "t__symbol_uses__ingest",
    "t__symbol_uses__run",
]
