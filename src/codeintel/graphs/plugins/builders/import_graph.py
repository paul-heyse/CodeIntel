"""Import graph builder plugin using factory pattern.

This module provides the import graph builder as a graph plugin, wrapping
the existing functionality for constructing module-level import graphs.

Uses resource injection pattern via ctx.require() to access storage.
"""

from __future__ import annotations

from codeintel.config.steps_graphs import ImportGraphStepConfig
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_builder_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import StorageResource


def _build_import_graph(ctx: GraphExecutionContext) -> ComputationResult:
    """Build module-level import graphs from LibCST parsing.

    Uses resource injection to access storage.

    Returns
    -------
    ComputationResult
        Success result after constructing import graph tables.
    """
    from codeintel.graphs.import_graph import build_import_graph  # noqa: PLC0415

    # Get storage via resource injection
    storage = ctx.require(StorageResource)
    gateway = storage.gateway

    cfg = ImportGraphStepConfig(snapshot=ctx.snapshot)
    build_import_graph(gateway, cfg)
    return ComputationResult.ok()


import_graph_builder_plugin = make_builder_plugin(
    name="import_graph_builder",
    computation=_build_import_graph,
    stage="structure",
    produces_graphs=(GraphKind.IMPORT_GRAPH,),
    depends_on=(),
    provides=("import_graph",),
    produces_tables=("graph.import_modules", "graph.import_edges"),
)


def get_import_graph_builder_plugin() -> GraphPluginProtocol:
    """Return the import graph builder plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured import graph builder plugin.
    """
    return import_graph_builder_plugin


__all__ = [
    "get_import_graph_builder_plugin",
    "import_graph_builder_plugin",
]
