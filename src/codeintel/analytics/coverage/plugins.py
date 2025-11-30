"""Analytics plugins for coverage aggregation."""

from __future__ import annotations

from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.tests import compute_test_coverage_edges


def _coverage_functions_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.coverage_functions_cfg is None:
        message = (
            "CoverageAnalyticsStepConfig required in AnalyticsExecutionContext.coverage_functions_cfg"
        )
        raise ValueError(message)
    compute_coverage_functions(
        ctx.gateway,
        ctx.coverage_functions_cfg,
        context=ctx.analytics_context,
    )
    return None


COVERAGE_FUNCTIONS_PLUGIN = AnalyticsPlugin(
    name="coverage.functions",
    description="Aggregate line coverage to function-level metrics.",
    stage="coverage",
    enabled_by_default=True,
    run=_coverage_functions_run,
    severity="fatal",
    depends_on=("goids", "coverage_ingest"),
    provides=("analytics.coverage_functions",),
    requires=("coverage.lines",),
    resource_hints=ResourceHints(max_runtime_ms=60_000, priority=40),
    row_count_tables=("analytics.coverage_functions",),
)


def _coverage_test_edges_run(ctx: AnalyticsExecutionContext) -> object | None:
    if ctx.test_coverage_cfg is None:
        message = "TestCoverageStepConfig required in AnalyticsExecutionContext.test_coverage_cfg"
        raise ValueError(message)
    compute_test_coverage_edges(
        ctx.gateway,
        ctx.test_coverage_cfg,
        catalog_provider=ctx.catalog_provider,
    )
    return None


COVERAGE_TEST_EDGES_PLUGIN = AnalyticsPlugin(
    name="coverage.test_edges",
    description="Build test-to-function coverage edges from coverage contexts.",
    stage="coverage",
    enabled_by_default=True,
    run=_coverage_test_edges_run,
    severity="fatal",
    depends_on=("coverage_ingest", "tests_ingest", "goids"),
    provides=("coverage.test_edges",),
    requires=("coverage.lines",),
    resource_hints=ResourceHints(max_runtime_ms=60_000, priority=40),
    row_count_tables=("coverage.test_edges",),
)

register_analytics_plugin(COVERAGE_FUNCTIONS_PLUGIN)
register_analytics_plugin(COVERAGE_TEST_EDGES_PLUGIN)


__all__ = ["COVERAGE_FUNCTIONS_PLUGIN", "COVERAGE_TEST_EDGES_PLUGIN"]
