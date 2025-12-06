"""History timeseries plugin.

This plugin aggregates analytics across commits into history timeseries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import HistoryTimeseriesStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import SnapshotGatewayResolver

log = logging.getLogger(__name__)


class HistoryTimeseriesPlugin(TargetPlugin):
    """Aggregate analytics across commits into history timeseries.

    Computes historical trends by:
    - Aggregating analytics across commits
    - Building time-based metrics
    - Tracking evolution patterns

    This plugin requires special configuration (multi-commit analysis) and will
    skip gracefully if the required parameters are not provided.

    Outputs
    -------
    - analytics.history_timeseries: Historical trends data
    """

    plugin_name: ClassVar[str] = "history_timeseries"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Aggregate analytics across commits into history timeseries."
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

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

        # Get snapshot resolver from parameters (if available)
        # This is a specialized resource for multi-commit analysis
        snapshot_resolver = ctx.parameters.get_optional("history_snapshot_resolver", object)
        if snapshot_resolver is None:
            log.info(
                "Skipping history timeseries - history_snapshot_resolver not provided "
                "(multi-commit analysis requires explicit configuration)"
            )
            return TargetResult.succeeded(row_counts={"analytics.history_timeseries": 0})

        # Get commits from parameters
        commits_raw = ctx.parameters.get_optional("commits", list)
        if not commits_raw:
            log.info(
                "Skipping history timeseries - commits list not provided "
                "(multi-commit analysis requires explicit configuration)"
            )
            return TargetResult.succeeded(row_counts={"analytics.history_timeseries": 0})
        commits = tuple(str(c) for c in commits_raw)

        # Build config from context
        cfg = HistoryTimeseriesStepConfig(
            snapshot=ctx.snapshot,
            commits=commits,
        )

        resolver = cast("SnapshotGatewayResolver", snapshot_resolver)

        try:
            # Note: ToolRunner type mismatch - passing None
            compute_history_timeseries_gateways(
                ctx.gateway,
                cfg,
                resolver,
                runner=None,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"History timeseries computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["HistoryTimeseriesPlugin"]
