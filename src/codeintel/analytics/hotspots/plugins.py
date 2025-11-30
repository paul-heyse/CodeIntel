"""Analytics plugin for hotspots computation."""

from __future__ import annotations

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.config.steps_analytics import HotspotsStepConfig


def _hotspots_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.hotspots_cfg is None:
        message = "HotspotsStepConfig is required in AnalyticsExecutionContext.hotspots_cfg"
        raise ValueError(message)
    cfg: HotspotsStepConfig = ctx.hotspots_cfg
    build_hotspots(ctx.gateway, cfg, runner=ctx.extra.get("tool_runner"))
    return None


HOTSPOTS_PLUGIN = AnalyticsPlugin(
    name="hotspots.build",
    description="Compute file-level hotspots from AST metrics and churn.",
    stage="other",
    enabled_by_default=True,
    run=_hotspots_run,
    severity="fatal",
    provides=("analytics.hotspots",),
    requires=("core.ast_metrics",),
    resource_hints=ResourceHints(max_runtime_ms=60_000, priority=50),
    row_count_tables=("analytics.hotspots",),
)

register_analytics_plugin(HOTSPOTS_PLUGIN)


__all__ = ["HOTSPOTS_PLUGIN"]
