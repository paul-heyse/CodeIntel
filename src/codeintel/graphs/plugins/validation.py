"""Graph validation plugin.

This module validates graph integrity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GraphValidationStepConfig
from codeintel.graphs.compute.validation import validate_graphs

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class GraphValidationPlugin(TargetPlugin):
    """Validate graph integrity.

    Outputs
    -------
    - graphs.validation_errors: Graph validation errors
    """

    plugin_name: ClassVar[str] = "graph_validation"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Validate graph integrity."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute graph validation.

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

        cfg = GraphValidationStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        try:
            row_counts = validate_graphs(ctx.gateway, cfg)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Graph validation failed: {e}")


__all__ = ["GraphValidationPlugin"]
