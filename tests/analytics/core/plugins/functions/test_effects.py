"""Tests for FunctionEffectsPlugin.

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
from codeintel.analytics.core.plugins.functions.effects import FunctionEffectsPlugin
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import FunctionEffectsStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 2
EXPECTED_CAPABILITY_COUNT = 2


def _create_config() -> FunctionEffectsStepConfig:
    """Create a test configuration.

    Returns
    -------
    FunctionEffectsStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return FunctionEffectsStepConfig(snapshot=snapshot)


def _create_resource_availability_map(
    *,
    has_catalog: bool,
    has_graph: bool,
    has_ast: bool,
) -> dict[str, bool]:
    """Create a resource availability mapping.

    Parameters
    ----------
    has_catalog
        Whether catalog provider is available.
    has_graph
        Whether graph provider is available.
    has_ast
        Whether AST provider is available.

    Returns
    -------
    dict[str, bool]
        Mapping of resource name to availability.
    """
    return {
        "CatalogProvider": has_catalog,
        "GraphProvider": has_graph,
        "AstProvider": has_ast,
    }


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
    elif name == "AstProvider":
        ast_data = MagicMock()
        ast_data.function_ast_map = {}
        ast_data.missing_function_goids = set()
        provider.get.return_value = ast_data
    return provider


def _create_mock_context(
    *,
    has_config: bool = True,
    has_catalog: bool = True,
    has_graph: bool = False,
    has_ast: bool = False,
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
    has_ast
        Whether AST provider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    resource_map = _create_resource_availability_map(
        has_catalog=has_catalog,
        has_graph=has_graph,
        has_ast=has_ast,
    )

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
    assert len(plugin.metadata.capabilities_provided) == EXPECTED_CAPABILITY_COUNT

    cap_names = {c.name for c in plugin.metadata.capabilities_provided}
    assert "analytics.function_effects" in cap_names
    assert "analytics.function_effects_evidence" in cap_names


def test_effects_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = FunctionEffectsPlugin()
    assert "functions" in plugin.metadata.tags
    assert "effects" in plugin.metadata.tags
    assert "purity" in plugin.metadata.tags


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_with_minimal_resources(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with minimal required resources."""
    plugin = FunctionEffectsPlugin()

    # Create a realistic mock context that uses fresh_gateway
    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.return_value = False

    result = plugin.execute(ctx)

    # Should succeed (may produce empty result but not fail)
    assert result.success is True


def test_execute_with_catalog_provider(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds with catalog provider available."""
    plugin = FunctionEffectsPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()

    ctx.has_resource_by_name.side_effect = lambda n: n == "CatalogProvider"

    catalog = MagicMock()
    catalog.catalog.return_value = MagicMock(function_spans=[])
    cat_provider = MagicMock()
    cat_provider.get.return_value = catalog
    ctx.require_by_name.return_value = cat_provider

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_with_all_resources(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds with all resources available."""
    plugin = FunctionEffectsPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()

    def has_resource(name: str) -> bool:
        return name in {"CatalogProvider", "GraphProvider", "AstProvider"}

    ctx.has_resource_by_name.side_effect = has_resource
    ctx.require_by_name.side_effect = _create_mock_provider

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


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "FunctionEffectsStepConfig" in result.errors[0]


def test_execute_returns_error_on_config_missing() -> None:
    """Execute returns error details when config is missing."""
    plugin = FunctionEffectsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert not result.success
    assert result.error is not None
