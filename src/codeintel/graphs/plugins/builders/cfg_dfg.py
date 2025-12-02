"""CFG and DFG builder plugin using factory pattern.

This module provides the control-flow graph (CFG) and data-flow graph (DFG)
builder as a graph plugin, wrapping the existing functionality.
"""

from __future__ import annotations

from codeintel.config.steps_graphs import CFGBuilderStepConfig
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    make_builder_plugin,
)
from codeintel.graphs.engine import GraphKind


def _build_cfg_and_dfg(ctx: GraphExecutionContext) -> ComputationResult:
    """Build control-flow and data-flow graphs for functions."""
    from codeintel.graphs.cfg_builder import build_cfg_and_dfg  # noqa: PLC0415

    cfg = CFGBuilderStepConfig(snapshot=ctx.snapshot)
    build_cfg_and_dfg(ctx.gateway, cfg, catalog_provider=ctx.catalog_provider)
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


def get_cfg_dfg_builder_plugin() -> object:
    """Return the CFG/DFG builder plugin instance."""
    return cfg_dfg_builder_plugin


# Legacy class alias for backward compatibility
CFGDFGBuilderPlugin = type(cfg_dfg_builder_plugin)


__all__ = [
    "CFGDFGBuilderPlugin",
    "cfg_dfg_builder_plugin",
    "get_cfg_dfg_builder_plugin",
]
