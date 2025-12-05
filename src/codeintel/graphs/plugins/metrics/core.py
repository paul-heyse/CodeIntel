"""Core graph metrics plugin.

This module computes core graph metrics (PageRank, centrality, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GraphMetricsStepConfig
from codeintel.graphs.compute.metrics import compute_core_metrics

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class CoreMetricsPlugin(TargetPlugin):
    """Compute core graph metrics (PageRank, centrality, etc.).

    Outputs
    -------
    - analytics.call_graph_metrics: Call graph metrics
    - analytics.import_graph_metrics: Import graph metrics
    """

    plugin_name: ClassVar[str] = "graph_metrics.core"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute core graph metrics (PageRank, centrality, etc.)."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute core metrics computation.

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
            row_counts = compute_core_metrics(ctx.gateway, cfg, runtime=graph_runtime)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Core metrics computation failed: {e}")


__all__ = ["CoreMetricsPlugin"]
