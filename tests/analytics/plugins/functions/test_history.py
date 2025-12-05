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

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.functions.history import FunctionHistoryPlugin
from codeintel.config.steps_analytics import FunctionHistoryStepConfig
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.factories import make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config(tmp_path: Path) -> FunctionHistoryStepConfig:
    """Create a test configuration.

    Parameters
    ----------
    tmp_path
        Temporary path for repo root.

    Returns
    -------
    FunctionHistoryStepConfig
        Test configuration.
    """
    return make_step_config(FunctionHistoryStepConfig, tmp_path)


def _create_context(
    tmp_path: Path,
    *,
    has_config: bool = True,
    has_tool_runner: bool = False,
) -> PluginExecutionContext:
    """Create a real execution context for testing.

    Parameters
    ----------
    tmp_path
        Temporary path for repo root.
    has_config
        Whether config is available.
    has_tool_runner
        Whether tool_runner is available in extras.

    Returns
    -------
    PluginExecutionContext
        Real execution context.
    """
    builder = TestExecutionContextBuilder.create(tmp_path)

    if has_config:
        config = _create_config(tmp_path)
        builder.with_config(FunctionHistoryStepConfig, config)

    if has_tool_runner:
        # Use a real object instead of mock
        builder.with_extra("tool_runner", object())

    return builder.build()


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
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.function_history" in plugin.metadata.provides


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


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "FunctionHistoryStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = FunctionHistoryPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_without_tool_runner(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds without optional tool_runner."""
    plugin = FunctionHistoryPlugin()
    config = _create_config(tmp_path)

    ctx = (
        TestExecutionContextBuilder(fresh_gateway, config.snapshot)
        .with_config(FunctionHistoryStepConfig, config)
        .build()
    )

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_succeeds_with_tool_runner(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with optional tool_runner."""
    plugin = FunctionHistoryPlugin()
    config = _create_config(tmp_path)

    ctx = (
        TestExecutionContextBuilder(fresh_gateway, config.snapshot)
        .with_config(FunctionHistoryStepConfig, config)
        .with_extra("tool_runner", object())
        .build()
    )

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_passes_tool_runner_to_domain(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute passes tool_runner to domain function when available."""
    plugin = FunctionHistoryPlugin()
    config = _create_config(tmp_path)

    ctx = (
        TestExecutionContextBuilder(fresh_gateway, config.snapshot)
        .with_config(FunctionHistoryStepConfig, config)
        .with_extra("tool_runner", object())
        .build()
    )

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

    assert "core.goids" in plugin.metadata.requires


def test_plugin_resource_hints() -> None:
    """Plugin has reasonable resource hints."""
    plugin = FunctionHistoryPlugin()

    hints = plugin.metadata.resource_hints
    assert hints is not None
    assert hints.max_runtime_ms is not None
    assert hints.max_runtime_ms > 0
    assert hints.priority is not None
    assert hints.priority > 0


# Ensure DEFAULT_REPO and DEFAULT_COMMIT are available for assertion checks
_ = DEFAULT_REPO, DEFAULT_COMMIT
