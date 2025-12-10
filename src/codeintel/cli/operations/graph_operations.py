"""Graph operation specifications.

Define operation specs for the graph command group including
stats, query, and plugins commands.

Note: These register to the LEGACY registry for backward compatibility.
New handler registrations are in handlers/graphs.py (NEW registry).
"""

from __future__ import annotations

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    GraphPluginsResult,
    GraphQueryResult,
    GraphStatsResult,
)
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection.registry import register_operation


def _graph_stats_handler(
    *,
    graph_type: str = "call",
) -> CliResult[GraphStatsResult]:
    """Get graph statistics handler.

    Parameters
    ----------
    graph_type
        Type of graph (call, import, etc).

    Returns
    -------
    CliResult[GraphStatsResult]
        Graph statistics result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    _ = graph_type
    return CliResult.ok(
        GraphStatsResult(
            node_count=0,
            edge_count=0,
            density=0.0,
            components=0,
            avg_degree=0.0,
        )
    )


def _graph_query_handler(
    *,
    query: str,
    limit: int = 100,
) -> CliResult[GraphQueryResult]:
    """Execute graph query handler.

    Parameters
    ----------
    query
        Graph query string.
    limit
        Maximum results to return.

    Returns
    -------
    CliResult[GraphQueryResult]
        Graph query result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    _ = limit
    return CliResult.ok(
        GraphQueryResult(
            nodes=[],
            edges=[],
            query=query,
        )
    )


def _graph_plugins_handler() -> CliResult[GraphPluginsResult]:
    """List graph plugins handler.

    Returns
    -------
    CliResult[GraphPluginsResult]
        Graph plugins result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(
        GraphPluginsResult(
            plugins=[],
            count=0,
        )
    )


# Graph Stats Operation (registers to LEGACY registry)
GRAPH_STATS_SPEC: OperationSpec[GraphStatsResult] = register_operation(
    OperationSpec(
        operation_id="graph.stats",
        handler=_graph_stats_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Show graph statistics",
    )
)

# Graph Query Operation (registers to LEGACY registry)
GRAPH_QUERY_SPEC: OperationSpec[GraphQueryResult] = register_operation(
    OperationSpec(
        operation_id="graph.query",
        handler=_graph_query_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Execute graph query",
    )
)

# Graph Plugins Operation (registers to LEGACY registry)
GRAPH_PLUGINS_SPEC: OperationSpec[GraphPluginsResult] = register_operation(
    OperationSpec(
        operation_id="graph.plugins",
        handler=_graph_plugins_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="List graph plugins",
    )
)

__all__ = [
    "GRAPH_PLUGINS_SPEC",
    "GRAPH_QUERY_SPEC",
    "GRAPH_STATS_SPEC",
]
