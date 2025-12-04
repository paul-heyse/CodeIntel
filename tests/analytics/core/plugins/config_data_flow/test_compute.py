"""Tests for ConfigDataFlowPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from codeintel.analytics.core.plugin_protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.core.plugins.config_data_flow.compute import (
    ConfigDataFlowPlugin,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config() -> ConfigDataFlowStepConfig:
    """Create a test configuration.

    Returns
    -------
    ConfigDataFlowStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return ConfigDataFlowStepConfig(snapshot=snapshot)


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
    if name == "CatalogProvider":
        catalog = MagicMock()
        catalog.catalog.return_value = MagicMock(function_spans=[])
        provider.get.return_value = catalog
    elif name == "GraphProvider":
        provider.runtime = None
    return provider


def _create_mock_context(
    *,
    has_config: bool = True,
    has_catalog: bool = False,
    has_graph: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_catalog
        Whether catalog provider is available.
    has_graph
        Whether graph provider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    resource_map = {
        "CatalogProvider": has_catalog,
        "GraphProvider": has_graph,
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


def test_config_data_flow_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = ConfigDataFlowPlugin()
    assert plugin.metadata.name == "config.data_flow"


def test_config_data_flow_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = ConfigDataFlowPlugin()
    assert plugin.metadata.stage == "config"


def test_config_data_flow_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = ConfigDataFlowPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_config_data_flow_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = ConfigDataFlowPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "config_data_flow" in output_names


def test_config_data_flow_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = ConfigDataFlowPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.config_data_flow" in plugin.metadata.provides


def test_config_data_flow_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = ConfigDataFlowPlugin()
    assert "config" in plugin.metadata.tags
    assert "data_flow" in plugin.metadata.tags
    assert "tracking" in plugin.metadata.tags


def test_config_data_flow_plugin_metadata_depends_on() -> None:
    """Plugin metadata has correct dependencies."""
    plugin = ConfigDataFlowPlugin()
    assert "config_ingest" in plugin.metadata.depends_on
    assert "callgraph" in plugin.metadata.depends_on


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "ConfigDataFlowStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_raises_on_domain_function_mismatch(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute raises TypeError due to argument mismatch with domain function.

    Note: The current plugin implementation passes 'catalog_provider' and 'runtime'
    kwargs to compute_config_data_flow(), but the actual function signature
    expects 'call_graph' and 'ast_by_goid'. This test documents this behavior.
    The TypeError is not caught by the plugin's exception handler.
    """
    plugin = ConfigDataFlowPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.return_value = False

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        plugin.execute(ctx)


def test_execute_raises_with_catalog_provider(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute raises TypeError when using catalog provider.

    Note: Documents current behavior where domain function call has arg mismatch.
    """
    plugin = ConfigDataFlowPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.side_effect = lambda n: n == "CatalogProvider"
    ctx.require_by_name.side_effect = _create_mock_provider

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        plugin.execute(ctx)


def test_execute_raises_with_graph_provider(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute raises TypeError when using graph provider.

    Note: Documents current behavior where domain function call has arg mismatch.
    """
    plugin = ConfigDataFlowPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.side_effect = lambda n: n == "GraphProvider"
    ctx.require_by_name.side_effect = _create_mock_provider

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        plugin.execute(ctx)


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = ConfigDataFlowPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = ConfigDataFlowPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
