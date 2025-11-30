"""Analytics plugin for entrypoints detection."""

from __future__ import annotations

from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.config.steps_analytics import EntryPointsStepConfig


def _entrypoints_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.entrypoints_cfg is None:
        message = "EntryPointsStepConfig required in AnalyticsExecutionContext.entrypoints_cfg"
        raise ValueError(message)
    cfg: EntryPointsStepConfig = ctx.entrypoints_cfg
    build_entrypoints(
        ctx.gateway,
        cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


ENTRYPOINTS_PLUGIN = AnalyticsPlugin(
    name="entrypoints.build",
    description="Detect HTTP/CLI/job entrypoints and map them to handlers and tests.",
    stage="entrypoints",
    enabled_by_default=True,
    run=_entrypoints_run,
    severity="fatal",
    depends_on=("subsystems.build", "coverage.functions", "coverage.test_edges", "goids"),
    provides=("analytics.entrypoints", "analytics.entrypoint_tests"),
    requires=("core.goids",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=50),
    row_count_tables=("analytics.entrypoints", "analytics.entrypoint_tests"),
)

register_analytics_plugin(ENTRYPOINTS_PLUGIN)


__all__ = ["ENTRYPOINTS_PLUGIN"]
