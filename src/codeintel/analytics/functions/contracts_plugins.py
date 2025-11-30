"""Analytics plugin for function contracts."""

from __future__ import annotations

from codeintel.analytics.functions import compute_function_contracts
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)


def _function_contracts_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.function_contracts_cfg is None:
        message = "FunctionContractsStepConfig required in AnalyticsExecutionContext"
        raise ValueError(message)
    compute_function_contracts(
        ctx.gateway,
        ctx.function_contracts_cfg,
        catalog_provider=ctx.analytics_context.catalog if ctx.analytics_context else None,
        context=ctx.analytics_context,
        runtime=ctx.graph_runtime,
    )
    return None


FUNCTION_CONTRACTS_PLUGIN = AnalyticsPlugin(
    name="functions.contracts",
    description="Infer pre/postconditions and nullability contracts for functions.",
    stage="function",
    enabled_by_default=True,
    run=_function_contracts_run,
    severity="fatal",
    depends_on=("functions.metrics",),
    provides=("analytics.function_contracts",),
    requires=("analytics.function_metrics", "analytics.docstrings"),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=30),
    row_count_tables=("analytics.function_contracts",),
)

register_analytics_plugin(FUNCTION_CONTRACTS_PLUGIN)


__all__ = ["FUNCTION_CONTRACTS_PLUGIN"]
