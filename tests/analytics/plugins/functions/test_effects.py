"""Tests for FunctionEffectsPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.functions.effects import FunctionEffectsPlugin
from codeintel.config.steps_analytics import FunctionEffectsStepConfig
from tests._helpers.factories import make_snapshot, make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.storage.gateway import StorageGateway

# Test constants (non-repo/commit)
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 2
EXPECTED_CAPABILITY_COUNT = 2


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
        config = make_step_config(FunctionEffectsStepConfig, tmp_path)
        builder.with_config(FunctionEffectsStepConfig, config)

    return builder.build()


# =============================================================================
# Metadata Tests
# =============================================================================


def test_effects_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = FunctionEffectsPlugin()
    assert plugin.metadata.name == "functions.effects"


def test_effects_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = FunctionEffectsPlugin()
    assert plugin.metadata.stage == "function"


def test_effects_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = FunctionEffectsPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_effects_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = FunctionEffectsPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "function_effects" in output_names
    assert "function_effects_evidence" in output_names


def test_effects_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = FunctionEffectsPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.function_effects" in plugin.metadata.provides
    assert "analytics.function_effects_evidence" in plugin.metadata.provides


def test_effects_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = FunctionEffectsPlugin()
    assert "functions" in plugin.metadata.tags
    assert "effects" in plugin.metadata.tags
    assert "purity" in plugin.metadata.tags


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_with_minimal_resources(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with minimal required resources."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    # Should succeed (may produce empty result but not fail)
    assert result.success is True


def test_execute_succeeds_with_config(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with config available."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = FunctionEffectsPlugin()
    # Dataclasses have __dataclass_fields__
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = FunctionEffectsPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "FunctionEffectsStepConfig" in result.errors[0]


def test_execute_returns_error_on_config_missing(tmp_path: Path) -> None:
    """Execute returns error details when config is missing."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert not result.success
    assert result.error is not None
