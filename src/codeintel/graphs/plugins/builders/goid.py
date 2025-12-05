"""GOID builder plugin.

This module builds Global Object IDs from SCIP data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GoidStepConfig
from codeintel.graphs.compute.goids import build_goid_data

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class GoidPlugin(TargetPlugin):
    """Build Global Object IDs from SCIP data.

    Outputs
    -------
    - core.goids: Global Object IDs
    - core.goid_crosswalk: SCIP symbol to GOID mapping
    """

    plugin_name: ClassVar[str] = "goids"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build Global Object IDs from SCIP data."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute GOID construction.

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

        cfg = GoidStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        try:
            row_counts = build_goid_data(ctx.gateway, cfg)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"GOID build failed: {e}")


__all__ = ["GoidPlugin"]
