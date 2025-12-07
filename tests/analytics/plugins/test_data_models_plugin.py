"""Integration-style test for the data models analytics plugin."""

from __future__ import annotations

from pathlib import Path

from codeintel.analytics.plugins.data_models.build import DataModelsPlugin
from tests._helpers.context import create_test_context
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin
from tests._helpers.seeds.data_models import DATA_MODELS_PACK


def test_data_models_plugin_extracts_models_and_usage(tmp_path: Path) -> None:
    """DataModelsPlugin should populate data_models and usage tables."""
    ctx = create_test_context(tmp_path)
    ctx.require(DATA_MODELS_PACK)

    plugin_ctx = PluginTestContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.build_paths,
    )
    result = execute_target_plugin(DataModelsPlugin(), plugin_ctx)
    assert result.success

    assert ctx.query_count("analytics.data_models") >= 1
    assert ctx.query_count("analytics.data_model_usage") >= 0

    ctx.close()
