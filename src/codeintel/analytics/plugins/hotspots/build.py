"""Hotspots plugin.

This module computes file-level hotspots based on AST complexity
metrics and Git churn patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.hotspots.metrics import build_hotspots
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import HotspotsStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


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

    def _compute_row_counts(self, ctx: TargetExecutionContext) -> dict[str, int]:
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
        _ = self  # Instance method for future extension
        row_counts: dict[str, int] = {}
        for table_key in ctx.contract.table_keys:
            try:
                count = ctx.gateway.con.execute(
                    f"SELECT COUNT(*) FROM {table_key} "  # noqa: S608
                    f"WHERE repo = ? AND commit = ?",
                    [ctx.repo, ctx.commit],
                ).fetchone()
                row_counts[table_key] = int(count[0]) if count else 0
            except (RuntimeError, OSError):
                row_counts[table_key] = 0
        return row_counts


__all__ = ["HotspotsPlugin"]
