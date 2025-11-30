"""Analytics plugin for semantic roles."""

from __future__ import annotations

from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.semantic_roles import compute_semantic_roles


def _semantic_roles_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.semantic_roles_cfg is None:
        message = "SemanticRolesStepConfig required in AnalyticsExecutionContext.semantic_roles_cfg"
        raise ValueError(message)
    compute_semantic_roles(
        ctx.gateway,
        ctx.semantic_roles_cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


SEMANTIC_ROLES_PLUGIN = AnalyticsPlugin(
    name="semantic.roles",
    description="Compute semantic roles for functions and calls.",
    stage="other",
    enabled_by_default=True,
    run=_semantic_roles_run,
    severity="fatal",
    depends_on=("callgraph",),
    provides=("analytics.semantic_roles",),
    requires=("core.goids",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=50),
    row_count_tables=("analytics.semantic_roles",),
)

register_analytics_plugin(SEMANTIC_ROLES_PLUGIN)


__all__ = ["SEMANTIC_ROLES_PLUGIN"]
