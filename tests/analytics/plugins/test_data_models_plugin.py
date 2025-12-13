"""Integration-style test for the data models analytics plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.plugins.analytics.data_models.build import DataModelsPlugin
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.plugin_harness import PluginHarnessFactory

if TYPE_CHECKING:
    from pathlib import Path


def test_data_models_plugin_extracts_models_and_usage(tmp_path: Path) -> None:
    """DataModelsPlugin should populate data_models and usage tables."""
    factory = PluginHarnessFactory(tmp_path)
    with factory.data_models() as harness:
        result = harness.execute_plugin(DataModelsPlugin())
        expect_true(result.success)

        model_count = harness.ctx.query_count("analytics.data_models")
        expect_true(
            model_count > 0,
            message="Data models table should have rows after plugin execution",
        )
        first = harness.ctx.query("SELECT model_name, module FROM analytics.data_models LIMIT 1")[0]
        expect_true(bool(first.model_name))
        expect_true(bool(first.module))

        usage_count = harness.ctx.query_count("analytics.data_model_usage")
        expect_equal(
            usage_count,
            0,
            label="data_model_usage row count (current seeds produce no usage rows)",
        )
