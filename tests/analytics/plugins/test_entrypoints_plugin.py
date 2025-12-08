"""Tests for the entrypoints analytics plugin."""

from __future__ import annotations

from codeintel.analytics.plugins.entrypoints.build import EntrypointsPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.assertions import expect_false, expect_true
from tests._helpers.plugin_execution import execute_target_plugin
from tests.analytics.conftest import PluginTestHarness


def test_entrypoints_plugin_requires_catalog(plugin_harness: PluginTestHarness) -> None:
    """Plugin should fail fast when no catalog provider is available."""
    result = execute_target_plugin(EntrypointsPlugin(), plugin_harness.plugin_ctx)
    expect_false(result.success)


def test_entrypoints_plugin_handles_empty_features(plugin_harness: PluginTestHarness) -> None:
    """Plugin should succeed with an empty catalog and no modules."""
    catalog = FunctionCatalogService(FunctionCatalog(functions=[], module_by_path={}))
    plugin_harness.plugin_ctx.resources.catalog = catalog

    result = execute_target_plugin(EntrypointsPlugin(), plugin_harness.plugin_ctx)
    expect_true(result.success)
    # Tables should exist even if empty.
    expect_true(plugin_harness.ctx.query_count("analytics.entrypoints") >= 0)
    expect_true(plugin_harness.ctx.query_count("analytics.entrypoint_tests") >= 0)
