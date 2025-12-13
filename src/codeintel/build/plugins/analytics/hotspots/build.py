"""Hotspots plugin.

This module computes file-level hotspots based on AST complexity
metrics and Git churn patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.hotspots.metrics import build_hotspots
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins._helpers import compute_row_counts
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


HOTSPOTS_METADATA = CorePluginMetadata(
    name="analytics.hotspots",
    version="3.0.0",
    description="Compute file-level hotspots from AST metrics and Git churn.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="hotspots",
    provides=("analytics.hotspots",),
    requires=("core.ast_metrics",),
    produces_tables=("analytics.hotspots",),
    consumes_tables=("core.ast_metrics",),
)


class HotspotsPlugin(MetadataPlugin):
    """Compute file-level hotspots from AST metrics and churn.

    Identifies high-risk code areas based on:
    - AST complexity metrics
    - Git churn patterns
    - Change frequency

    Outputs
    -------
    - analytics.hotspots: File-level hotspot scores
    """

    _core_metadata: ClassVar[CorePluginMetadata] = HOTSPOTS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the hotspots computation.

        Parameters
        ----------
        ctx
            Execution context with gateway and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.
        """
        _ = self  # Protocol method requires instance
        max_commits = ctx.parameters.get("max_commits", int, default=2000)

        build_hotspots(ctx.gateway, ctx.snapshot, max_commits=max_commits, runner=None)

        row_counts = compute_row_counts(ctx)
        return TargetResult.succeeded(row_counts=row_counts)


__all__ = ["HOTSPOTS_METADATA", "HotspotsPlugin"]
