"""Integration-style tests for coverage analytics plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.plugins.coverage.functions import CoverageFunctionsPlugin
from codeintel.analytics.plugins.coverage.test_edges import CoverageTestEdgesPlugin
from tests._helpers.assertions import (
    assert_coverage_ratio_between,
    expect_equal,
    expect_true,
)
from tests._helpers.harnesses import coverage_plugin_harness, plugin_harness_with_packs
from tests._helpers.seeds.core import CORE_PACK, GOID_FUNC_B

if TYPE_CHECKING:
    from pathlib import Path


def test_coverage_functions_plugin_populates_function_metrics(tmp_path: Path) -> None:
    """CoverageFunctionsPlugin should aggregate line coverage into function rows."""
    with coverage_plugin_harness(tmp_path) as harness:
        result = harness.execute_plugin(CoverageFunctionsPlugin())
        expect_true(result.success)

        fn_count = harness.ctx.query_count("core.goids")
        coverage_count = harness.ctx.query_count("analytics.coverage_functions")
        expect_equal(coverage_count, fn_count)

        assert_coverage_ratio_between(harness.ctx, GOID_FUNC_B, low=0.0, high=1.0)


def test_coverage_test_edges_plugin_handles_missing_coverage_file(tmp_path: Path) -> None:
    """CoverageTestEdgesPlugin should no-op when coverage data is absent."""
    with plugin_harness_with_packs(tmp_path, CORE_PACK) as harness:
        result = harness.execute_plugin(CoverageTestEdgesPlugin())
        expect_true(result.success)

        expect_equal(harness.ctx.query_count("analytics.test_coverage_edges"), 0)
