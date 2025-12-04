"""Tests for ExternalDepsPlugin.

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
from codeintel.analytics.core.plugins.dependencies.external import ExternalDepsPlugin
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 2
EXPECTED_CAPABILITY_COUNT = 2


def _create_config() -> ExternalDependenciesStepConfig:
    """Create a test configuration.

    Returns
    -------
    ExternalDependenciesStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return ExternalDependenciesStepConfig(snapshot=snapshot)


def _create_resource_availability_map(
    *,
    has_catalog: bool,
    has_module_map: bool,
    has_ast: bool,
    has_features: bool,
) -> dict[str, bool]:
    """Create a resource availability mapping.

    Parameters
    ----------
    has_catalog
        Whether catalog provider is available.
    has_module_map
        Whether module map provider is available.
    has_ast
        Whether AST provider is available.
    has_features
        Whether features provider is available.

    Returns
    -------
    dict[str, bool]
        Mapping of resource name to availability.
    """
    return {
        "CatalogProvider": has_catalog,
        "ModuleMapProvider": has_module_map,
        "AstProvider": has_ast,
        "FeaturesProvider": has_features,
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
    elif name == "ModuleMapProvider":
        provider.get.return_value = {}
    elif name == "AstProvider":
        ast_data = MagicMock()
        ast_data.function_ast_map = {}
        ast_data.missing_function_goids = set()
        provider.get.return_value = ast_data
    elif name == "FeaturesProvider":
        provider.get.return_value = {}
    return provider


def _create_mock_context(
    *,
    has_config: bool = True,
    has_catalog: bool = True,
    has_module_map: bool = False,
    has_ast: bool = False,
    has_features: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_catalog
        Whether catalog provider is available.
    has_module_map
        Whether module map provider is available.
    has_ast
        Whether AST provider is available.
    has_features
        Whether features provider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    resource_map = _create_resource_availability_map(
        has_catalog=has_catalog,
        has_module_map=has_module_map,
        has_ast=has_ast,
        has_features=has_features,
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


def test_external_deps_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = ExternalDepsPlugin()
    assert plugin.metadata.name == "deps.external"


def test_external_deps_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = ExternalDepsPlugin()
    assert plugin.metadata.stage == "other"


def test_external_deps_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = ExternalDepsPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_external_deps_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = ExternalDepsPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "external_dependency_calls" in output_names
    assert "external_dependencies" in output_names


def test_external_deps_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = ExternalDepsPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.external_dependency_calls" in plugin.metadata.provides
    assert "analytics.external_dependencies" in plugin.metadata.provides


def test_external_deps_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = ExternalDepsPlugin()
    assert "dependencies" in plugin.metadata.tags
    assert "external" in plugin.metadata.tags


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = ExternalDepsPlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = ExternalDepsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = ExternalDepsPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_without_catalog() -> None:
    """Execute fails when catalog provider is not available."""
    plugin = ExternalDepsPlugin()
    ctx = _create_mock_context(has_config=True, has_catalog=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert "CatalogProvider is required" in str(result.error)


def test_execute_succeeds_with_minimal_resources(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with minimal required resources."""
    plugin = ExternalDepsPlugin()

    # Create a realistic mock context that uses fresh_gateway
    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.side_effect = lambda n: n == "CatalogProvider"

    # Mock catalog provider
    catalog = MagicMock()
    catalog.catalog.return_value = MagicMock(function_spans=[])
    cat_provider = MagicMock()
    cat_provider.get.return_value = catalog
    ctx.require_by_name.return_value = cat_provider

    result = plugin.execute(ctx)

    # Should succeed (may produce empty result but not fail)
    assert result.success is True


def test_execute_with_all_resources(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds with all resources available."""
    plugin = ExternalDepsPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()

    def has_resource(name: str) -> bool:
        return name in {
            "CatalogProvider",
            "ModuleMapProvider",
            "AstProvider",
            "FeaturesProvider",
        }

    ctx.has_resource_by_name.side_effect = has_resource

    def require_resource(name: str) -> MagicMock:
        provider = MagicMock()
        if name == "CatalogProvider":
            catalog = MagicMock()
            catalog.catalog.return_value = MagicMock(function_spans=[])
            provider.get.return_value = catalog
        elif name == "ModuleMapProvider":
            provider.get.return_value = {}
        elif name == "AstProvider":
            ast_data = MagicMock()
            ast_data.function_ast_map = {}
            ast_data.missing_function_goids = set()
            provider.get.return_value = ast_data
        elif name == "FeaturesProvider":
            provider.get.return_value = {}
        return provider

    ctx.require_by_name.side_effect = require_resource

    result = plugin.execute(ctx)

    assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = ExternalDepsPlugin()
    # Dataclasses have __dataclass_fields__
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = ExternalDepsPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
