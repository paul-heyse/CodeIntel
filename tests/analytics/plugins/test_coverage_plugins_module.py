"""Integration-style tests for coverage analytics plugins."""

from __future__ import annotations

from typing import cast

from codeintel.analytics.plugins.coverage.functions import CoverageFunctionsPlugin
from codeintel.analytics.plugins.coverage.test_edges import CoverageTestEdgesPlugin
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.seeds.core import CORE_PACK, GOID_FUNC_B
from tests._helpers.seeds.coverage_lines import COVERAGE_LINES_PACK
from tests.analytics.conftest import PluginTestHarness


def test_coverage_functions_plugin_populates_function_metrics(
    plugin_harness: PluginTestHarness,
) -> None:
    """CoverageFunctionsPlugin should aggregate line coverage into function rows."""
    plugin_harness.ctx.require(CORE_PACK, COVERAGE_LINES_PACK)

    result = plugin_harness.execute_plugin(CoverageFunctionsPlugin())
    expect_true(result.success)

    fn_count = plugin_harness.ctx.query_count("core.goids")
    coverage_count = plugin_harness.ctx.query_count("analytics.coverage_functions")
    expect_equal(coverage_count, fn_count)

    row = plugin_harness.ctx.query(
        """
        SELECT coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE function_goid_h128 = ?
        """,
        [GOID_FUNC_B],
    )[0]
    ratio_value = cast("float", row.coverage_ratio)
    expect_true(0.0 < ratio_value < 1.0)
    expect_true(row.tested is True)


def test_coverage_test_edges_plugin_handles_missing_coverage_file(
    plugin_harness: PluginTestHarness,
) -> None:
    """CoverageTestEdgesPlugin should no-op when coverage data is absent."""
    plugin_harness.ctx.require(CORE_PACK)

    result = plugin_harness.execute_plugin(CoverageTestEdgesPlugin())
    expect_true(result.success)

    expect_equal(plugin_harness.ctx.query_count("analytics.test_coverage_edges"), 0)
