"""CFG and DFG builder plugin using factory pattern.

This module provides the control-flow graph (CFG) and data-flow graph (DFG)
builder as a graph plugin, wrapping the existing functionality.

Uses resource injection pattern via ctx.require() to access storage.
"""

from __future__ import annotations

from codeintel.config.steps_graphs import CFGBuilderStepConfig
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_builder_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import StorageResource


def _build_cfg_and_dfg(ctx: GraphExecutionContext) -> ComputationResult:
    """Build control-flow and data-flow graphs for functions.

    Uses resource injection to access storage.

    Returns
    -------
    ComputationResult
        Success result after building CFG and DFG artifacts.
    """
    from codeintel.graphs.cfg_builder import build_cfg_and_dfg  # noqa: PLC0415

    # Get storage via resource injection
    storage = ctx.require(StorageResource)
    gateway = storage.gateway

    cfg = CFGBuilderStepConfig(snapshot=ctx.snapshot)
    build_cfg_and_dfg(gateway, cfg)
    return ComputationResult.ok()


cfg_dfg_builder_plugin = make_builder_plugin(
    name="cfg_dfg_builder",
    computation=_build_cfg_and_dfg,
    stage="edges",
    produces_graphs=(GraphKind.CFG_GRAPH,),
    depends_on=("goid_builder",),
    provides=("cfg_graph", "dfg_graph"),
    produces_tables=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
)


def get_cfg_dfg_builder_plugin() -> GraphPluginProtocol:
    """Return the CFG/DFG builder plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured CFG/DFG builder plugin.
    """
    return cfg_dfg_builder_plugin


__all__ = [
    "cfg_dfg_builder_plugin",
    "get_cfg_dfg_builder_plugin",
]
