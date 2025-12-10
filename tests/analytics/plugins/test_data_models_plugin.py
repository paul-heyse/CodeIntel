"""Integration-style test for the data models analytics plugin."""

from __future__ import annotations

from codeintel.analytics.plugins.data_models.build import DataModelsPlugin
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.seeds.data_models import DATA_MODELS_PACK
from tests.analytics.conftest import PluginTestHarness


def test_data_models_plugin_extracts_models_and_usage(plugin_harness: PluginTestHarness) -> None:
    """DataModelsPlugin should populate data_models and usage tables."""
    plugin_harness.ctx.require(DATA_MODELS_PACK)

    result = plugin_harness.execute_plugin(DataModelsPlugin())
    expect_true(result.success)

    model_count = plugin_harness.ctx.query_count("analytics.data_models")
    expect_true(
        model_count > 0,
        message="Data models table should have rows after plugin execution",
    )
    first = plugin_harness.ctx.query(
        "SELECT model_name, module FROM analytics.data_models LIMIT 1"
    )[0]
    expect_true(bool(first.model_name))
    expect_true(bool(first.module))

    usage_count = plugin_harness.ctx.query_count("analytics.data_model_usage")
    expect_equal(
        usage_count,
        0,
        label="data_model_usage row count (current seeds produce no usage rows)",
    )
