"""Analytics plugin for external dependencies."""

from __future__ import annotations

from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.config import ExternalDependenciesStepConfig


def _external_deps_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.external_deps_cfg is None:
        message = (
            "ExternalDependenciesStepConfig required in AnalyticsExecutionContext.external_deps_cfg"
        )
        raise ValueError(message)
    cfg: ExternalDependenciesStepConfig = ctx.external_deps_cfg
    build_external_dependency_calls(
        ctx.gateway,
        cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    build_external_dependencies(ctx.gateway, cfg)
    return None


EXTERNAL_DEPS_PLUGIN = AnalyticsPlugin(
    name="deps.external",
    description="Identify external dependency usage across functions.",
    stage="other",
    enabled_by_default=True,
    run=_external_deps_run,
    severity="fatal",
    depends_on=("goids", "config_ingest"),
    provides=("analytics.external_dependency_calls", "analytics.external_dependencies"),
    requires=("core.goids",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=50),
    row_count_tables=(
        "analytics.external_dependency_calls",
        "analytics.external_dependencies",
    ),
)

register_analytics_plugin(EXTERNAL_DEPS_PLUGIN)


__all__ = ["EXTERNAL_DEPS_PLUGIN"]
