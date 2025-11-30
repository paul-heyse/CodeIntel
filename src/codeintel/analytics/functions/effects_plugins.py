"""Analytics plugin for function effects."""

from __future__ import annotations

from codeintel.analytics.functions import compute_function_effects
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)


def _function_effects_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.function_effects_cfg is None:
        message = "FunctionEffectsStepConfig required in AnalyticsExecutionContext"
        raise ValueError(message)
    compute_function_effects(
        ctx.gateway,
        ctx.function_effects_cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


FUNCTION_EFFECTS_PLUGIN = AnalyticsPlugin(
    name="functions.effects",
    description="Classify side effects and purity for functions.",
    stage="function",
    enabled_by_default=True,
    run=_function_effects_run,
    severity="fatal",
    depends_on=("functions.metrics", "callgraph"),
    provides=("analytics.function_effects", "analytics.function_effects_evidence"),
    requires=("core.goids",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=30),
    row_count_tables=(
        "analytics.function_effects",
        "analytics.function_effects_evidence",
    ),
)

register_analytics_plugin(FUNCTION_EFFECTS_PLUGIN)


__all__ = ["FUNCTION_EFFECTS_PLUGIN"]
