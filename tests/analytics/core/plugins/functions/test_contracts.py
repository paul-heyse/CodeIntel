"""Tests for FunctionContractsPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from codeintel.analytics.core.plugin_protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.core.plugins.functions.contracts import (
    FunctionContractsPlugin,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import FunctionContractsStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config() -> FunctionContractsStepConfig:
    """Create a test configuration.

    Returns
    -------
    FunctionContractsStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return FunctionContractsStepConfig(snapshot=snapshot)


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
    if name == "AstProvider":
        ast_data = MagicMock()
        ast_data.function_ast_map = {}
        ast_data.missing_function_goids = set()
        provider.get.return_value = ast_data
    elif name == "CatalogProvider":
        catalog = MagicMock()
        catalog.catalog.return_value = MagicMock(function_spans=[])
        provider.get.return_value = catalog
    return provider


def _create_mock_context(
    *,
    has_config: bool = True,
    has_ast: bool = True,
    has_catalog: bool = True,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_ast
        Whether AST provider is available.
    has_catalog
        Whether catalog provider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    resource_map = {
        "AstProvider": has_ast,
        "CatalogProvider": has_catalog,
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


def test_contracts_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = FunctionContractsPlugin()
    assert plugin.metadata.name == "functions.contracts"


def test_contracts_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = FunctionContractsPlugin()
    assert plugin.metadata.stage == "function"


def test_contracts_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = FunctionContractsPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_contracts_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = FunctionContractsPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "function_contracts" in output_names


def test_contracts_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = FunctionContractsPlugin()
    assert len(plugin.metadata.capabilities_provided) == EXPECTED_CAPABILITY_COUNT

    cap_names = {c.name for c in plugin.metadata.capabilities_provided}
    assert "analytics.function_contracts" in cap_names


def test_contracts_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = FunctionContractsPlugin()
    assert "functions" in plugin.metadata.tags
    assert "contracts" in plugin.metadata.tags
    assert "nullability" in plugin.metadata.tags


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = FunctionContractsPlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = FunctionContractsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = FunctionContractsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "FunctionContractsStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = FunctionContractsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_without_ast_provider() -> None:
    """Execute fails when AST provider is not available."""
    plugin = FunctionContractsPlugin()
    ctx = _create_mock_context(has_config=True, has_ast=False, has_catalog=True)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert "AstProvider is required" in str(result.error)


def test_execute_fails_without_catalog_provider() -> None:
    """Execute fails when catalog provider is not available."""
    plugin = FunctionContractsPlugin()
    ctx = _create_mock_context(has_config=True, has_ast=True, has_catalog=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert "CatalogProvider is required" in str(result.error)


def test_execute_succeeds_with_all_resources(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds with all required resources."""
    plugin = FunctionContractsPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()

    def has_resource(name: str) -> bool:
        return name in {"AstProvider", "CatalogProvider"}

    ctx.has_resource_by_name.side_effect = has_resource
    ctx.require_by_name.side_effect = _create_mock_provider

    result = plugin.execute(ctx)

    assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = FunctionContractsPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = FunctionContractsPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
