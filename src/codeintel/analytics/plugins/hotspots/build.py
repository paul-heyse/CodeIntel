"""Hotspots plugin.

This module computes file-level hotspots based on AST complexity
metrics and Git churn patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.hotspots.metrics import build_hotspots
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import HotspotsStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata


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


class HotspotsPlugin(TargetPlugin):
    """Compute file-level hotspots from AST metrics and churn.

    Identifies high-risk code areas based on:
    - AST complexity metrics
    - Git churn patterns
    - Change frequency

    Outputs
    -------
    - analytics.hotspots: File-level hotspot scores
    """

    plugin_name: ClassVar[str] = "hotspots"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Compute file-level hotspots from AST metrics and Git churn."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = HOTSPOTS_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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
        # Get parameters
        max_commits = ctx.parameters.get("max_commits", int, default=2000)

        # Build hotspots config from parameters
        cfg = HotspotsStepConfig(
            snapshot=ctx.snapshot,
            max_commits=max_commits,
        )

        # Execute computation
        # Note: ToolRunner type mismatch between build.protocols and ingestion.engine.infrastructure
        # Passing None for now as runner is optional
        build_hotspots(ctx.gateway, cfg, runner=None)

        # Compute row counts
        row_counts = self._compute_row_counts(ctx)
        return TargetResult.succeeded(row_counts=row_counts)

    @staticmethod
    def _compute_row_counts(ctx: TargetExecutionContext) -> dict[str, int]:
        """Compute row counts for output tables.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, int]
            Row counts per table.
        """
        row_counts: dict[str, int] = {}
        for table_key in ctx.contract.table_keys:
            try:
                table = ctx.gateway.ibis.table(table_key)
                filtered = table.filter((table.repo == ctx.repo) & (table.commit == ctx.commit))
                result_df = filtered.aggregate(row_count=table.repo.count()).execute()
                row_count = int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0
                row_counts[table_key] = row_count
            except (RuntimeError, OSError):
                row_counts[table_key] = 0
        return row_counts


__all__ = ["HOTSPOTS_METADATA", "HotspotsPlugin"]
