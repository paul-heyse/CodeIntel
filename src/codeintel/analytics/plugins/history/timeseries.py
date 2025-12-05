"""History timeseries plugin.

This plugin aggregates analytics across commits into history timeseries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.steps_analytics import HistoryTimeseriesStepConfig

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import SnapshotGatewayResolver


class HistoryTimeseriesPlugin(TargetPlugin):
    """Aggregate analytics across commits into history timeseries.

    Computes historical trends by:
    - Aggregating analytics across commits
    - Building time-based metrics
    - Tracking evolution patterns

    Outputs
    -------
    - analytics.history_timeseries: Historical trends data
    """

    plugin_name: ClassVar[str] = "history.timeseries"
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

        # Build config from context
        cfg = HistoryTimeseriesStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        # Get snapshot resolver from parameters (if available)
        # This is a specialized resource that might need to be provided
        snapshot_resolver = ctx.parameters.get_optional("history_snapshot_resolver", object)
        if snapshot_resolver is None:
            return TargetResult.failed("history_snapshot_resolver is required in parameters")

        resolver = cast("SnapshotGatewayResolver", snapshot_resolver)
        tool_runner = ctx.resources.tool_runner

        try:
            compute_history_timeseries_gateways(
                ctx.gateway,
                cfg,
                resolver,
                runner=tool_runner,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"History timeseries computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["HistoryTimeseriesPlugin"]
