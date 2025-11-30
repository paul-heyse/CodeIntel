"""Analytics plugin for config data flow."""

from __future__ import annotations

from codeintel.analytics.graphs import compute_config_data_flow
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)


def _config_data_flow_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.config_data_flow_cfg is None:
        message = (
            "ConfigDataFlowStepConfig required in AnalyticsExecutionContext.config_data_flow_cfg"
        )
        raise ValueError(message)
    compute_config_data_flow(
        ctx.gateway,
        ctx.config_data_flow_cfg,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


CONFIG_DATA_FLOW_PLUGIN = AnalyticsPlugin(
    name="config.data_flow",
    description="Track configuration key usage and data flow at the function level.",
    stage="subsystem",
    enabled_by_default=True,
    run=_config_data_flow_run,
    severity="fatal",
    depends_on=("config_ingest", "callgraph", "functions.metrics", "entrypoints.build"),
    provides=("analytics.config_data_flow",),
    requires=("core.config_keys",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=40),
    row_count_tables=("analytics.config_data_flow",),
)

register_analytics_plugin(CONFIG_DATA_FLOW_PLUGIN)


__all__ = ["CONFIG_DATA_FLOW_PLUGIN"]
