"""Import graph builder plugin.

This module builds the module-level import graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import ImportGraphStepConfig
from codeintel.graphs.compute.imports import build_import_graph_data

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class ImportGraphPlugin(TargetPlugin):
    """Build module-level import graph.

    Outputs
    -------
    - graphs.import_graph_nodes: Import graph nodes
    - graphs.import_graph_edges: Import graph edges
    """

    plugin_name: ClassVar[str] = "import_graph"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build module-level import graph."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute import graph construction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self  # Protocol method requires instance

        cfg = ImportGraphStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        try:
            row_counts = build_import_graph_data(ctx.gateway, cfg)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Import graph build failed: {e}")


__all__ = ["ImportGraphPlugin"]
