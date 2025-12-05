"""Tests for DataModelsPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.data_models.build import DataModelsPlugin
from codeintel.config.steps_analytics import DataModelsStepConfig
from tests._helpers.factories import make_snapshot, make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.storage.gateway import StorageGateway

# Test constants (non-repo/commit)
TEST_VERSION = "2.0.0"
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
        config = make_step_config(DataModelsStepConfig, tmp_path)
        builder.with_config(DataModelsStepConfig, config)

    return builder.build()


# =============================================================================
# Metadata Tests
# =============================================================================


def test_data_models_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = DataModelsPlugin()
    assert plugin.metadata.name == "data_models.build"


def test_data_models_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = DataModelsPlugin()
    assert plugin.metadata.stage == "data_model"


def test_data_models_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = DataModelsPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_data_models_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = DataModelsPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "data_models" in output_names


def test_data_models_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = DataModelsPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.data_models" in plugin.metadata.provides


def test_data_models_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = DataModelsPlugin()
    assert "data_models" in plugin.metadata.tags
    assert "schema" in plugin.metadata.tags
    assert "extraction" in plugin.metadata.tags


def test_data_models_plugin_metadata_depends_on() -> None:
    """Plugin metadata has correct dependencies."""
    plugin = DataModelsPlugin()
    assert "ast_extract" in plugin.metadata.depends_on
    assert "goids" in plugin.metadata.depends_on


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = DataModelsPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = DataModelsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = DataModelsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "DataModelsStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = DataModelsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_with_config(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds when config is present."""
    plugin = DataModelsPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_returns_ok_result(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute returns PluginResult.ok() on success."""
    plugin = DataModelsPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is True
    assert result.error is None


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = DataModelsPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = DataModelsPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage


def test_plugin_capabilities_required() -> None:
    """Plugin requires core.goids capability."""
    plugin = DataModelsPlugin()

    assert "core.goids" in plugin.metadata.requires


def test_plugin_resource_hints() -> None:
    """Plugin has reasonable resource hints."""
    plugin = DataModelsPlugin()

    hints = plugin.metadata.resource_hints
    assert hints is not None
    assert hints.max_runtime_ms is not None
    assert hints.max_runtime_ms > 0
    assert hints.priority is not None
    assert hints.priority > 0


def test_plugin_input_spec() -> None:
    """Plugin has correct input spec."""
    plugin = DataModelsPlugin()

    inputs = plugin.metadata.inputs
    assert len(inputs) > 0

    input_names = {i.name for i in inputs}
    assert "data_models_cfg" in input_names
