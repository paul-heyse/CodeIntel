"""Tests for the entrypoints analytics plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.plugins.analytics.entrypoints.build import EntrypointsPlugin
from codeintel.core.catalog import CatalogService, FunctionCatalog
from tests._helpers.assertions import expect_false, expect_true
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.plugin_harness import PluginHarnessFactory

if TYPE_CHECKING:
    from pathlib import Path

    from tests.analytics.conftest import PluginTestHarness


def test_entrypoints_plugin_requires_catalog(plugin_harness: PluginTestHarness) -> None:
    """Plugin should fail fast when no catalog provider is available."""
    result = plugin_harness.execute_plugin(EntrypointsPlugin())
    expect_false(result.success)


def test_entrypoints_plugin_handles_empty_features(tmp_path: Path) -> None:
    """Plugin should succeed with an empty catalog and no modules."""
    factory = PluginHarnessFactory(tmp_path)
    with factory.entrypoints() as harness:
        catalog = CatalogService(FunctionCatalog(functions=[], module_by_path={}))
        resources = TargetResourceOverrides(catalog=catalog)

        result = harness.execute_plugin(EntrypointsPlugin(), resources=resources)
        expect_true(result.success)
        expect_true(
            harness.ctx.query_count("analytics.entrypoints") == 0,
            message="Entrypoints table should be created but empty for an empty catalog",
        )
        expect_true(
            harness.ctx.query_count("analytics.entrypoint_tests") == 0,
            message="Entrypoint tests table should be created but empty for an empty catalog",
        )
