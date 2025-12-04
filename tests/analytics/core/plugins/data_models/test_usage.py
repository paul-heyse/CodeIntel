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
from unittest.mock import MagicMock

from codeintel.analytics.core.plugin_protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.core.plugins.data_models.usage import DataModelUsagePlugin
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import DataModelUsageStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config() -> DataModelUsageStepConfig:
    """Create a test configuration.

    Returns
    -------
    DataModelUsageStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return DataModelUsageStepConfig(snapshot=snapshot)


def _create_mock_provider(name: str) -> MagicMock:
    """Create a mock provider for the given resource name.

    Parameters
    ----------
    name
        Resource provider name.

    Returns
    -------
    MagicMock
        Mock provider.
    """
    provider = MagicMock()
    if name == "ModuleMapProvider":
        provider.get.return_value = {}
    elif name == "AstProvider":
        ast_data = MagicMock()
        ast_data.function_ast_map = {}
        ast_data.missing_function_goids = set()
        provider.get.return_value = ast_data
    return provider


def _create_mock_context(
    *,
    has_config: bool = True,
    has_module_map: bool = False,
    has_ast: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_module_map
        Whether module map provider is available.
    has_ast
        Whether AST provider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    resource_map = {
        "ModuleMapProvider": has_module_map,
        "AstProvider": has_ast,
    }

    ctx.has_config.return_value = has_config
    if has_config:
        ctx.get_config.return_value = _create_config()
    else:
        ctx.get_config.side_effect = ValueError("Config not found")

    ctx.has_resource_by_name.side_effect = lambda n: resource_map.get(n, False)
    ctx.require_by_name.side_effect = _create_mock_provider
    ctx.gateway = MagicMock()

    return ctx


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


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = DataModelUsagePlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = DataModelUsagePlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = DataModelUsagePlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "DataModelUsageStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = DataModelUsagePlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_without_module_map_provider(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute fails when ModuleMapProvider is missing (required)."""
    plugin = DataModelUsagePlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.return_value = False

    result = plugin.execute(ctx)

    assert result.success is False
    assert "ModuleMapProvider" in (result.error or "")


def test_execute_fails_without_ast_provider(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute fails when AstProvider is missing (required)."""
    plugin = DataModelUsagePlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    # Has ModuleMapProvider but not AstProvider
    ctx.has_resource_by_name.side_effect = lambda n: n == "ModuleMapProvider"
    ctx.require_by_name.side_effect = _create_mock_provider

    result = plugin.execute(ctx)

    assert result.success is False
    assert "AstProvider" in (result.error or "")


def test_execute_succeeds_with_all_providers(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds when all required providers are present."""
    plugin = DataModelUsagePlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.return_value = True
    ctx.require_by_name.side_effect = _create_mock_provider

    result = plugin.execute(ctx)

    assert result.success is True
    assert result.error is None


def test_execute_handles_domain_error(fresh_gateway: StorageGateway) -> None:
    """Execute handles errors from domain function gracefully."""
    plugin = DataModelUsagePlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.return_value = True

    # Make the AST provider return bad data that causes an error
    bad_provider = MagicMock()
    bad_ast = MagicMock()
    bad_ast.function_ast_map = None  # This should cause an error
    bad_ast.missing_function_goids = None
    bad_provider.get.return_value = bad_ast

    def _provider_with_error(name: str) -> MagicMock:
        if name == "AstProvider":
            return bad_provider
        return _create_mock_provider(name)

    ctx.require_by_name.side_effect = _provider_with_error

    # Should handle the error gracefully
    result = plugin.execute(ctx)

    # The result should either succeed (if None is handled) or fail gracefully
    assert isinstance(result, PluginResult)


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


def test_plugin_requires_both_providers() -> None:
    """Plugin requires both ModuleMapProvider and AstProvider."""
    plugin = DataModelUsagePlugin()

    # Only has ModuleMapProvider
    ctx1 = MagicMock()
    ctx1.gateway = MagicMock()
    ctx1.has_config.return_value = True
    ctx1.get_config.return_value = _create_config()
    ctx1.has_resource_by_name.side_effect = lambda n: n == "ModuleMapProvider"
    ctx1.require_by_name.side_effect = _create_mock_provider

    result1 = plugin.execute(ctx1)
    assert result1.success is False

    # Only has AstProvider
    ctx2 = MagicMock()
    ctx2.gateway = MagicMock()
    ctx2.has_config.return_value = True
    ctx2.get_config.return_value = _create_config()
    ctx2.has_resource_by_name.side_effect = lambda n: n == "AstProvider"
    ctx2.require_by_name.side_effect = _create_mock_provider

    result2 = plugin.execute(ctx2)
    assert result2.success is False
