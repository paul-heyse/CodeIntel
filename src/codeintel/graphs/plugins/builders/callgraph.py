"""Call graph builder plugin using factory pattern.

This module provides the call graph builder as a graph plugin, wrapping
the existing `build_call_graph` functionality with the plugin protocol.
"""

from __future__ import annotations

from codeintel.config.steps_graphs import CallGraphStepConfig
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    make_builder_plugin,
)
from codeintel.graphs.engine import GraphKind


def _build_call_graph(ctx: GraphExecutionContext) -> ComputationResult:
    """
    Build call graph nodes and edges from GOIDs.

    Returns
    -------
    ComputationResult
        Success result after building call graph artifacts.
    """
    from codeintel.graphs.callgraph_builder import build_call_graph  # noqa: PLC0415

    cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
    build_call_graph(ctx.gateway, cfg, catalog_provider=ctx.catalog_provider)
    return ComputationResult.ok()


callgraph_builder_plugin = make_builder_plugin(
    name="callgraph_builder",
    computation=_build_call_graph,
    stage="edges",
    produces_graphs=(GraphKind.CALL_GRAPH,),
    depends_on=("goid_builder",),
    provides=("call_graph",),
    produces_tables=("graph.call_graph_nodes", "graph.call_graph_edges"),
)


def get_callgraph_builder_plugin() -> object:
    """
    Return the call graph builder plugin instance.

    Returns
    -------
    object
        The configured call graph builder plugin.
    """
    return callgraph_builder_plugin


# Legacy class alias for backward compatibility
CallGraphBuilderPlugin = type(callgraph_builder_plugin)


__all__ = [
    "CallGraphBuilderPlugin",
    "callgraph_builder_plugin",
    "get_callgraph_builder_plugin",
]
