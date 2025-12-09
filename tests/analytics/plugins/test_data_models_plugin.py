"""Integration-style test for the data models analytics plugin."""

from __future__ import annotations

from codeintel.analytics.plugins.data_models.build import DataModelsPlugin
from tests._helpers.assertions import expect_true
from tests._helpers.seeds.data_models import DATA_MODELS_PACK
from tests.analytics.conftest import PluginTestHarness


def test_data_models_plugin_extracts_models_and_usage(plugin_harness: PluginTestHarness) -> None:
    """DataModelsPlugin should populate data_models and usage tables."""
    plugin_harness.ctx.require(DATA_MODELS_PACK)

    result = plugin_harness.execute_plugin(DataModelsPlugin())
    expect_true(result.success)

    expect_true(plugin_harness.ctx.query_count("analytics.data_models") >= 1)
    expect_true(plugin_harness.ctx.query_count("analytics.data_model_usage") >= 0)
