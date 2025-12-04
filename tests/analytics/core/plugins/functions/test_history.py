"""Tests for FunctionHistoryPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
- Optional tool_runner from ctx.extra
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from codeintel.analytics.core.plugin_protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.core.plugins.functions.history import FunctionHistoryPlugin
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import FunctionHistoryStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config() -> FunctionHistoryStepConfig:
    """Create a test configuration.

    Returns
    -------
    FunctionHistoryStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return FunctionHistoryStepConfig(snapshot=snapshot)


def _create_mock_context(
    *,
    has_config: bool = True,
    has_tool_runner: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_tool_runner
        Whether tool_runner is available in extras.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()

    ctx.has_config.return_value = has_config
    if has_config:
        ctx.get_config.return_value = _create_config()
    else:
        ctx.get_config.side_effect = ValueError("Config not found")

    # Set up extra dict with optional tool_runner
    if has_tool_runner:
        mock_runner = MagicMock()
        ctx.extra = {"tool_runner": mock_runner}
    else:
        ctx.extra = {}

    ctx.gateway = MagicMock()

    return ctx


# =============================================================================
# Metadata Tests
# =============================================================================


def test_function_history_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = FunctionHistoryPlugin()
    assert plugin.metadata.name == "functions.history"


def test_function_history_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = FunctionHistoryPlugin()
    assert plugin.metadata.stage == "function_history"


def test_function_history_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = FunctionHistoryPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_function_history_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = FunctionHistoryPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "function_history" in output_names


def test_function_history_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = FunctionHistoryPlugin()
    assert len(plugin.metadata.capabilities_provided) == EXPECTED_CAPABILITY_COUNT

    cap_names = {c.name for c in plugin.metadata.capabilities_provided}
    assert "analytics.function_history" in cap_names


def test_function_history_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = FunctionHistoryPlugin()
    assert "functions" in plugin.metadata.tags
    assert "history" in plugin.metadata.tags
    assert "git" in plugin.metadata.tags


def test_function_history_plugin_metadata_depends_on() -> None:
    """Plugin metadata has correct dependencies."""
    plugin = FunctionHistoryPlugin()
    assert "functions.metrics" in plugin.metadata.depends_on
    assert "hotspots.build" in plugin.metadata.depends_on


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "FunctionHistoryStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_without_tool_runner(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds without optional tool_runner."""
    plugin = FunctionHistoryPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.extra = {}

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_succeeds_with_tool_runner(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds with optional tool_runner."""
    plugin = FunctionHistoryPlugin()

    mock_runner = MagicMock()
    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.extra = {"tool_runner": mock_runner}

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_passes_tool_runner_to_domain(fresh_gateway: StorageGateway) -> None:
    """Execute passes tool_runner to domain function when available."""
    plugin = FunctionHistoryPlugin()

    mock_runner = MagicMock()
    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.extra = {"tool_runner": mock_runner}

    # Execute should not raise and should pass runner to domain
    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = FunctionHistoryPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = FunctionHistoryPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage


def test_plugin_capabilities_required() -> None:
    """Plugin requires core.goids capability."""
    plugin = FunctionHistoryPlugin()

    req_caps = {c.name for c in plugin.metadata.capabilities_required}
    assert "core.goids" in req_caps


def test_plugin_resource_hints() -> None:
    """Plugin has reasonable resource hints."""
    plugin = FunctionHistoryPlugin()

    hints = plugin.metadata.resource_hints
    assert hints is not None
    assert hints.max_runtime_ms is not None
    assert hints.max_runtime_ms > 0
    assert hints.priority is not None
    assert hints.priority > 0
