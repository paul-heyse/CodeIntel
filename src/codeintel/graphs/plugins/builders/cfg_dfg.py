"""CFG/DFG builder plugin.

This module builds control flow and data flow graphs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import CfgDfgStepConfig
from codeintel.graphs.compute.cfg_dfg import build_cfg_dfg_data

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class CfgDfgPlugin(TargetPlugin):
    """Build control flow and data flow graphs.

    Outputs
    -------
    - graphs.cfg_edges: Control flow graph edges
    - graphs.dfg_edges: Data flow graph edges
    """

    plugin_name: ClassVar[str] = "cfg_dfg"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build control flow and data flow graphs."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute CFG/DFG construction.

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

        cfg = CfgDfgStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        catalog = ctx.resources.catalog

        try:
            row_counts = build_cfg_dfg_data(ctx.gateway, cfg, catalog=catalog)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"CFG/DFG build failed: {e}")


__all__ = ["CfgDfgPlugin"]
