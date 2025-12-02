"""GOID builder plugin using factory pattern.

This module provides the GOID builder as a graph plugin, wrapping
the existing functionality for building Global Object Identifiers.
"""

from __future__ import annotations

from codeintel.config.steps_graphs import GoidBuilderStepConfig
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    make_builder_plugin,
)


def _build_goids(ctx: GraphExecutionContext) -> ComputationResult:
    """Build GOIDs and crosswalk entries from AST nodes."""
    from codeintel.graphs.goid_builder import build_goids  # noqa: PLC0415

    cfg = GoidBuilderStepConfig(snapshot=ctx.snapshot)
    build_goids(ctx.gateway, cfg)
    return ComputationResult.ok()


goid_builder_plugin = make_builder_plugin(
    name="goid_builder",
    computation=_build_goids,
    stage="goid",
    produces_graphs=(),
    depends_on=(),
    provides=("goids",),
    produces_tables=("core.goids", "core.goid_crosswalk"),
)


def get_goid_builder_plugin() -> object:
    """Return the GOID builder plugin instance."""
    return goid_builder_plugin


# Legacy class alias for backward compatibility
GoidBuilderPlugin = type(goid_builder_plugin)


__all__ = [
    "GoidBuilderPlugin",
    "get_goid_builder_plugin",
    "goid_builder_plugin",
]
