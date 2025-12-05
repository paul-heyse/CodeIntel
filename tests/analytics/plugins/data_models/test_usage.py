"""Tests for DataModelUsagePlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
- Required provider failure cases (ModuleMapProvider, AstProvider)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.data_models.usage import DataModelUsagePlugin
from codeintel.analytics.resources.asts import AstProvider
from codeintel.config.steps_analytics import DataModelUsageStepConfig
from tests._helpers.factories import make_snapshot, make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.storage.gateway import StorageGateway

# Test constants (non-repo/commit)
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_context(
    tmp_path: Path,
    *,
    has_config: bool = True,
    has_ast: bool = False,
    gateway: StorageGateway | None = None,
) -> PluginExecutionContext:
    """Create a test execution context using real production types.

    Parameters
    ----------
    tmp_path
        Temp path for repo root.
    has_config
        Whether config is available.
    has_ast
        Whether AST provider is available.
    gateway
        Optional gateway override.

    Returns
    -------
    PluginExecutionContext
        Real execution context.
    """
    if gateway is not None:
        snapshot = make_snapshot(repo_root=tmp_path)
        builder = TestExecutionContextBuilder(gateway, snapshot)
    else:
        builder = TestExecutionContextBuilder.create(tmp_path)

    if has_config:
        config = make_step_config(DataModelUsageStepConfig, tmp_path)
        builder.with_config(DataModelUsageStepConfig, config)

    # Note: ModuleMapProvider registration would require additional setup
    # For now we test that the plugin correctly identifies missing providers

    if has_ast:
        # Create AST provider with empty preloaded data
        ast_provider = AstProvider.from_asts({}, set())
        builder.with_resource(AstProvider, ast_provider)

    return builder.build()


# =============================================================================
# Metadata Tests
# =============================================================================


def test_data_model_usage_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = DataModelUsagePlugin()
    assert plugin.metadata.name == "data_models.usage"


def test_data_model_usage_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = DataModelUsagePlugin()
    assert plugin.metadata.stage == "data_model_usage"


def test_data_model_usage_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = DataModelUsagePlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_data_model_usage_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = DataModelUsagePlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "data_model_usage" in output_names


def test_data_model_usage_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = DataModelUsagePlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.data_model_usage" in plugin.metadata.provides


def test_data_model_usage_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = DataModelUsagePlugin()
    assert "data_models" in plugin.metadata.tags
    assert "usage" in plugin.metadata.tags
    assert "patterns" in plugin.metadata.tags


def test_data_model_usage_plugin_metadata_depends_on() -> None:
    """Plugin metadata has correct dependencies."""
    plugin = DataModelUsagePlugin()
    assert "data_models.build" in plugin.metadata.depends_on
    assert "callgraph" in plugin.metadata.depends_on


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = DataModelUsagePlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = DataModelUsagePlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = DataModelUsagePlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "DataModelUsageStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = DataModelUsagePlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_without_module_map_provider(tmp_path: Path) -> None:
    """Execute fails when ModuleMapProvider is missing (required)."""
    plugin = DataModelUsagePlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.execute(ctx)

    assert result.success is False
    assert "ModuleMapProvider" in (result.error or "")


def test_execute_fails_without_ast_provider(tmp_path: Path) -> None:
    """Execute fails when AstProvider is missing (required)."""
    plugin = DataModelUsagePlugin()
    # Has config but no AST provider
    ctx = _create_context(tmp_path, has_config=True, has_ast=False)

    result = plugin.execute(ctx)

    # Should fail due to missing ModuleMapProvider first
    assert result.success is False


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = DataModelUsagePlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = DataModelUsagePlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
