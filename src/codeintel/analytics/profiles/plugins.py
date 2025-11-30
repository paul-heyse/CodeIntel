"""Analytics plugin for aggregated profiles."""

from __future__ import annotations

from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)


def _profiles_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.profiles_cfg is None:
        message = "ProfilesAnalyticsStepConfig required in AnalyticsExecutionContext.profiles_cfg"
        raise ValueError(message)
    cfg = ctx.profiles_cfg
    build_function_profile(
        ctx.gateway,
        cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
    )
    build_file_profile(
        ctx.gateway,
        cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
    )
    build_module_profile(
        ctx.gateway,
        cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
    )
    return None


PROFILES_PLUGIN = AnalyticsPlugin(
    name="profiles.build",
    description="Build aggregated profiles for functions, files, and modules.",
    stage="profiles",
    enabled_by_default=True,
    run=_profiles_run,
    severity="fatal",
    depends_on=(
        "risk_factors.build",
        "callgraph",
        "import_graph",
        "functions.effects",
        "functions.contracts",
        "semantic.roles",
        "functions.history",
    ),
    provides=(
        "analytics.function_profile",
        "analytics.file_profile",
        "analytics.module_profile",
    ),
    requires=("analytics.goid_risk_factors",),
    resource_hints=ResourceHints(max_runtime_ms=120_000, priority=70),
    row_count_tables=(
        "analytics.function_profile",
        "analytics.file_profile",
        "analytics.module_profile",
    ),
)

register_analytics_plugin(PROFILES_PLUGIN)


__all__ = ["PROFILES_PLUGIN"]
