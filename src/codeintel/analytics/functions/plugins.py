"""Analytics plugins for function metrics."""

from __future__ import annotations

from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.context import AnalyticsContextConfig, ensure_analytics_context
from codeintel.analytics.datasets import (
    DeleteScope,
    get_function_ast_features_contract,
    insert_analytics_rows,
)
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
    """
    Bridge from generic AnalyticsExecutionContext to function metrics step.

    Returns
    -------
    dict[str, int]
        Summary counters emitted by the metrics/type computation.

    Raises
    ------
    ValueError
        If the function analytics configuration is missing from the context.
    """
    if ctx.function_cfg is None:
        message = (
            "FunctionAnalyticsStepConfig is required in AnalyticsExecutionContext.function_cfg"
        )
        raise ValueError(message)

    cfg: FunctionAnalyticsStepConfig = ctx.function_cfg
    opts = FunctionAnalyticsOptions(context=ctx.analytics_context)
    return compute_function_metrics_and_types(
        ctx.gateway,
        cfg,
        options=opts,
    )


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


def _function_ast_features_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Compute and persist function AST features for a snapshot.

    Returns
    -------
    dict[str, int]
        Summary counters for rows written and function coverage.

    Raises
    ------
    ValueError
        If the function analytics configuration is missing.
    """
    if ctx.function_cfg is None:
        message = (
            "FunctionAnalyticsStepConfig is required in AnalyticsExecutionContext.function_cfg"
        )
        raise ValueError(message)

    cfg: FunctionAnalyticsStepConfig = ctx.function_cfg
    analytics_ctx = ensure_analytics_context(
        ctx.gateway,
        cfg=AnalyticsContextConfig(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
        ),
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    rows = [
        features_to_row(
            repo=cfg.repo,
            commit=cfg.commit,
            features=features,
        )
        for features in analytics_ctx.function_features_map.values()
    ]
    contract = get_function_ast_features_contract(ctx.gateway)
    delete_scope = DeleteScope(params=[cfg.repo, cfg.commit])
    insert_analytics_rows(
        ctx.gateway,
        contract,
        rows,
        delete_scope=delete_scope,
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    return {
        "rows_written": len(rows),
        "functions_seen": len(analytics_ctx.function_ast_map),
        "functions_missing": len(analytics_ctx.missing_function_goids),
    }


FUNCTION_AST_FEATURES_PLUGIN = AnalyticsPlugin(
    name="functions.ast_features",
    description="Compute AST-derived semantic features for each function.",
    stage="function",
    enabled_by_default=True,
    run=_function_ast_features_run,
    severity="fatal",
    depends_on=(),
    provides=("analytics.function_ast_features",),
    requires=("core.goids",),
    options_model=None,
    options_default=None,
    resource_hints=ResourceHints(
        max_runtime_ms=120_000,
        requires_gpu=False,
        priority=12,
    ),
    version_hash=None,
    row_count_tables=("analytics.function_ast_features",),
)

register_analytics_plugin(FUNCTION_AST_FEATURES_PLUGIN)


__all__ = [
    "FUNCTION_AST_FEATURES_PLUGIN",
    "FUNCTION_METRICS_PLUGIN",
]
