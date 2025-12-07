"""Integration-style tests for coverage analytics plugins."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from codeintel.analytics.plugins.coverage.functions import CoverageFunctionsPlugin
from codeintel.analytics.plugins.coverage.test_edges import CoverageTestEdgesPlugin
from tests._helpers.context import create_test_context
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin
from tests._helpers.seeds.core import CORE_PACK, GOID_FUNC_B
from tests._helpers.seeds.coverage_lines import COVERAGE_LINES_PACK


def test_coverage_functions_plugin_populates_function_metrics(tmp_path: Path) -> None:
    """CoverageFunctionsPlugin should aggregate line coverage into function rows."""
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK, COVERAGE_LINES_PACK)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    result = execute_target_plugin(CoverageFunctionsPlugin(), plugin_ctx)
    assert result.success

    fn_count = ctx.query_count("core.goids")
    coverage_count = ctx.query_count("analytics.coverage_functions")
    assert coverage_count == fn_count

    row = ctx.query(
        """
        SELECT coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE function_goid_h128 = ?
        """,
        [GOID_FUNC_B],
    )[0]
    ratio_value = cast("float", row.coverage_ratio)
    assert 0.0 < ratio_value < 1.0
    assert row.tested is True

    ctx.close()


def test_coverage_test_edges_plugin_handles_missing_coverage_file(tmp_path: Path) -> None:
    """CoverageTestEdgesPlugin should no-op when coverage data is absent."""
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    result = execute_target_plugin(CoverageTestEdgesPlugin(), plugin_ctx)
    assert result.success

    assert ctx.query_count("analytics.test_coverage_edges") == 0

    ctx.close()
