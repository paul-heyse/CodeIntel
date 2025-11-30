"""Analytics plugins for function metrics."""

from __future__ import annotations

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_metrics_and_types,
)
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig


def _function_metrics_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """Bridge from generic AnalyticsExecutionContext to function metrics step."""
    if ctx.function_cfg is None:
        message = (
            "FunctionAnalyticsStepConfig is required in "
            "AnalyticsExecutionContext.function_cfg"
        )
        raise ValueError(message)

    cfg: FunctionAnalyticsStepConfig = ctx.function_cfg
    opts = FunctionAnalyticsOptions(context=ctx.analytics_context)
    summary = compute_function_metrics_and_types(
        ctx.gateway,
        cfg,
        options=opts,
    )
    return summary


FUNCTION_METRICS_PLUGIN = AnalyticsPlugin(
    name="functions.metrics",
    description="Compute function metrics, complexity, and type annotations.",
    stage="function",
    enabled_by_default=True,
    run=_function_metrics_run,
    severity="fatal",
    depends_on=(),
    provides=("analytics.function_metrics", "analytics.function_types"),
    requires=("core.goids",),
    options_model=None,
    options_default=None,
    resource_hints=ResourceHints(
        max_runtime_ms=60_000,
        requires_gpu=False,
        priority=10,
    ),
    version_hash=None,
    row_count_tables=("analytics.function_metrics", "analytics.function_types"),
)

register_analytics_plugin(FUNCTION_METRICS_PLUGIN)


__all__ = [
    "FUNCTION_METRICS_PLUGIN",
]
