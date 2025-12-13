"""History timeseries plugin.

This plugin aggregates analytics across commits into history timeseries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins.analytics._metadata import to_plugin_metadata
from codeintel.config.steps_analytics import HistoryTimeseriesStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata
    from codeintel.storage.gateway import SnapshotGatewayResolver

log = logging.getLogger(__name__)


HISTORY_TIMESERIES_METADATA = CorePluginMetadata(
    name="analytics.history_timeseries",
    version="3.0.0",
    description="Aggregate analytics across commits into history timeseries.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="history",
    provides=("analytics.history_timeseries",),
    requires=("analytics.function_history",),
    produces_tables=("analytics.history_timeseries",),
    consumes_tables=("analytics.function_history",),
)


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
    _core_metadata: ClassVar[CorePluginMetadata] = HISTORY_TIMESERIES_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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
        _ = self

        snapshot_resolver = ctx.parameters.get_optional("history_snapshot_resolver", object)
        if snapshot_resolver is None:
            log.info(
                "Skipping history timeseries - history_snapshot_resolver not provided "
                "(multi-commit analysis requires explicit configuration)"
            )
            return TargetResult.succeeded(row_counts={"analytics.history_timeseries": 0})

        commits_raw = ctx.parameters.get_optional("commits", list)
        if not commits_raw:
            log.info(
                "Skipping history timeseries - commits list not provided "
                "(multi-commit analysis requires explicit configuration)"
            )
            return TargetResult.succeeded(row_counts={"analytics.history_timeseries": 0})
        commits = tuple(str(c) for c in commits_raw)

        cfg = HistoryTimeseriesStepConfig(
            snapshot=ctx.snapshot,
            commits=commits,
        )

        resolver = cast("SnapshotGatewayResolver", snapshot_resolver)

        try:
            compute_history_timeseries_gateways(
                ctx.gateway,
                cfg,
                resolver,
                runner=None,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"History timeseries computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["HISTORY_TIMESERIES_METADATA", "HistoryTimeseriesPlugin"]
