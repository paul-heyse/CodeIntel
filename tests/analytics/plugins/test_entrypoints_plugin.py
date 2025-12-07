"""Tests for the entrypoints analytics plugin."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.plugins.entrypoints.build import EntrypointsPlugin
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
from tests._helpers.context import create_test_context
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin


def test_entrypoints_plugin_requires_catalog(tmp_path: Path) -> None:
    """Plugin should fail fast when no catalog provider is available."""
    ctx = create_test_context(tmp_path)
    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    result = execute_target_plugin(EntrypointsPlugin(), plugin_ctx)
    assert result.success is False

    ctx.close()


def test_entrypoints_plugin_handles_empty_features(tmp_path: Path) -> None:
    """Plugin should succeed with an empty catalog and no modules."""
    ctx = create_test_context(tmp_path)

    catalog = FunctionCatalogService(FunctionCatalog(functions=[], module_by_path={}))
    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    plugin_ctx.resources.catalog = catalog

    result = execute_target_plugin(EntrypointsPlugin(), plugin_ctx)
    assert result.success
    # Tables should exist even if empty.
    assert ctx.query_count("analytics.entrypoints") >= 0
    assert ctx.query_count("analytics.entrypoint_tests") >= 0

    ctx.close()
