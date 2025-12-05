"""Secondary graph metrics plugin.

This module computes secondary graph metrics (CFG/DFG metrics).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GraphMetricsStepConfig
from codeintel.graphs.compute.metrics import compute_secondary_metrics

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class SecondaryMetricsPlugin(TargetPlugin):
    """Compute secondary graph metrics (CFG/DFG metrics).

    Outputs
    -------
    - analytics.cfg_metrics: CFG metrics
    - analytics.dfg_metrics: DFG metrics
    """

    plugin_name: ClassVar[str] = "graph_metrics.secondary"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute secondary graph metrics (CFG/DFG metrics)."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute secondary metrics computation.

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

        cfg = GraphMetricsStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        graph_runtime = ctx.resources.graph_runtime

        try:
            row_counts = compute_secondary_metrics(ctx.gateway, cfg, runtime=graph_runtime)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Secondary metrics computation failed: {e}")


__all__ = ["SecondaryMetricsPlugin"]
