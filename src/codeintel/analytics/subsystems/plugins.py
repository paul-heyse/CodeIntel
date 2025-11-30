"""Analytics plugin for subsystem materialization."""

from __future__ import annotations

from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.subsystems import build_subsystems


def _subsystems_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.subsystems_cfg is None:
        message = "SubsystemsStepConfig required in AnalyticsExecutionContext.subsystems_cfg"
        raise ValueError(message)
    build_subsystems(
        ctx.gateway,
        ctx.subsystems_cfg,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


SUBSYSTEMS_PLUGIN = AnalyticsPlugin(
    name="subsystems.build",
    description="Infer subsystems from module coupling and risk signals.",
    stage="subsystem",
    enabled_by_default=True,
    run=_subsystems_run,
    severity="fatal",
    depends_on=("import_graph", "symbol_uses", "risk_factors.build"),
    provides=(
        "analytics.subsystems",
        "analytics.subsystem_modules",
        "analytics.subsystem_functions",
    ),
    requires=("core.modules",),
    resource_hints=ResourceHints(max_runtime_ms=120_000, priority=60),
    row_count_tables=(
        "analytics.subsystems",
        "analytics.subsystem_modules",
        "analytics.subsystem_functions",
    ),
)

register_analytics_plugin(SUBSYSTEMS_PLUGIN)


__all__ = ["SUBSYSTEMS_PLUGIN"]
