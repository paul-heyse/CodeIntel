"""Tests for ConfigDataFlowPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.config_data_flow.compute import (
    ConfigDataFlowPlugin,
)
from codeintel.config.steps_graphs import ConfigDataFlowStepConfig
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
    gateway: StorageGateway | None = None,
) -> PluginExecutionContext:
    """Create a test execution context using real production types.

    Parameters
    ----------
    tmp_path
        Temp path for repo root.
    has_config
        Whether config is available.
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
        config = make_step_config(ConfigDataFlowStepConfig, tmp_path)
        builder.with_config(ConfigDataFlowStepConfig, config)

    return builder.build()


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


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "ConfigDataFlowStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = ConfigDataFlowPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_raises_on_domain_function_mismatch(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute raises TypeError due to argument mismatch with domain function.

    Note: The current plugin implementation passes 'catalog_provider' and 'runtime'
    kwargs to compute_config_data_flow(), but the actual function signature
    expects 'call_graph' and 'ast_by_goid'. This test documents this behavior.
    The TypeError is not caught by the plugin's exception handler.
    """
    plugin = ConfigDataFlowPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

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
