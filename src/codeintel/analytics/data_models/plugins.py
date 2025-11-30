"""Analytics plugins for data models and usage."""

from __future__ import annotations

from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)


def _data_models_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.data_models_cfg is None:
        message = "DataModelsStepConfig required in AnalyticsExecutionContext.data_models_cfg"
        raise ValueError(message)
    compute_data_models(ctx.gateway, ctx.data_models_cfg)
    return None


DATA_MODELS_PLUGIN = AnalyticsPlugin(
    name="data_models.build",
    description="Extract structured data models from class definitions.",
    stage="data_model",
    enabled_by_default=True,
    run=_data_models_run,
    severity="fatal",
    depends_on=("ast_extract", "goids", "docstrings_ingest"),
    provides=("analytics.data_models",),
    requires=("core.goids",),
    resource_hints=ResourceHints(max_runtime_ms=60_000, priority=40),
    row_count_tables=("analytics.data_models",),
)


def _data_model_usage_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.data_model_usage_cfg is None:
        message = (
            "DataModelUsageStepConfig required in AnalyticsExecutionContext.data_model_usage_cfg"
        )
        raise ValueError(message)
    compute_data_model_usage(
        ctx.gateway,
        ctx.data_model_usage_cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


DATA_MODEL_USAGE_PLUGIN = AnalyticsPlugin(
    name="data_models.usage",
    description="Classify per-function data model read/write usage patterns.",
    stage="data_model_usage",
    enabled_by_default=True,
    run=_data_model_usage_run,
    severity="fatal",
    depends_on=("data_models.build", "callgraph", "cfg", "functions.metrics"),
    provides=("analytics.data_model_usage",),
    requires=("analytics.data_models",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=45),
    row_count_tables=("analytics.data_model_usage",),
)

register_analytics_plugin(DATA_MODELS_PLUGIN)
register_analytics_plugin(DATA_MODEL_USAGE_PLUGIN)


__all__ = ["DATA_MODELS_PLUGIN", "DATA_MODEL_USAGE_PLUGIN"]
