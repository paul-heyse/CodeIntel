"""Analytics plugin for history timeseries."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import SnapshotGatewayResolver


def _history_timeseries_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.history_cfg is None:
        message = "HistoryTimeseriesStepConfig required in AnalyticsExecutionContext.history_cfg"
        raise ValueError(message)
    snapshot_resolver = ctx.extra.get("history_snapshot_resolver")
    if snapshot_resolver is None:
        message = "history_snapshot_resolver is required for history.timeseries"
        raise ValueError(message)
    resolver = cast("SnapshotGatewayResolver", snapshot_resolver)
    compute_history_timeseries_gateways(
        ctx.gateway,
        ctx.history_cfg,
        resolver,
        runner=ctx.extra.get("tool_runner"),
    )
    return None


HISTORY_TIMESERIES_PLUGIN = AnalyticsPlugin(
    name="history.timeseries",
    description="Aggregate analytics across commits into history timeseries.",
    stage="history",
    enabled_by_default=True,
    run=_history_timeseries_run,
    severity="fatal",
    depends_on=("profiles.build",),
    provides=("analytics.history_timeseries",),
    requires=("analytics.function_profile",),
    resource_hints=ResourceHints(max_runtime_ms=120_000, priority=80),
    row_count_tables=("analytics.history_timeseries",),
)

register_analytics_plugin(HISTORY_TIMESERIES_PLUGIN)


__all__ = ["HISTORY_TIMESERIES_PLUGIN"]
