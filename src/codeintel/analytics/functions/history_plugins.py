"""Analytics plugin for function history."""

from __future__ import annotations

from codeintel.analytics.functions import compute_function_history
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)


def _function_history_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.function_history_cfg is None:
        message = (
            "FunctionHistoryStepConfig required in AnalyticsExecutionContext.function_history_cfg"
        )
        raise ValueError(message)
    compute_function_history(
        ctx.gateway,
        ctx.function_history_cfg,
        runner=ctx.extra.get("tool_runner"),
        context=ctx.analytics_context,
    )
    return None


FUNCTION_HISTORY_PLUGIN = AnalyticsPlugin(
    name="functions.history",
    description="Aggregate git churn and commit history per function GOID.",
    stage="function_history",
    enabled_by_default=True,
    run=_function_history_run,
    severity="fatal",
    depends_on=("functions.metrics", "hotspots.build"),
    provides=("analytics.function_history",),
    requires=("core.goids",),
    resource_hints=ResourceHints(max_runtime_ms=60_000, priority=40),
    row_count_tables=("analytics.function_history",),
)

register_analytics_plugin(FUNCTION_HISTORY_PLUGIN)


__all__ = ["FUNCTION_HISTORY_PLUGIN"]
