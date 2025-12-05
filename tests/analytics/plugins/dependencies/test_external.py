"""Tests for ExternalDepsPlugin.

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
from codeintel.analytics.plugins.dependencies.external import ExternalDepsPlugin
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogService
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
    has_catalog: bool = False,
    gateway: StorageGateway | None = None,
) -> PluginExecutionContext:
    """Create a test execution context using real production types.

    Parameters
    ----------
    tmp_path
        Temp path for repo root.
    has_config
        Whether config is available.
    has_catalog
        Whether catalog provider is available.
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
        config = make_step_config(ExternalDependenciesStepConfig, tmp_path)
        builder.with_config(ExternalDependenciesStepConfig, config)

    if has_catalog:
        # Create catalog provider with empty preloaded data
        empty_catalog = FunctionCatalog(functions=[], module_by_path={})
        catalog_service = FunctionCatalogService(empty_catalog)
        catalog_provider = CatalogProvider.from_catalog(catalog_service)
        builder.with_resource(CatalogProvider, catalog_provider)

    return builder.build()


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


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = ExternalDepsPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = ExternalDepsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = ExternalDepsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_without_catalog(tmp_path: Path) -> None:
    """Execute fails when catalog provider is not available."""
    plugin = ExternalDepsPlugin()
    ctx = _create_context(tmp_path, has_config=True, has_catalog=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert "CatalogProvider is required" in str(result.error)


def test_execute_succeeds_with_catalog(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with catalog provider available."""
    plugin = ExternalDepsPlugin()
    ctx = _create_context(
        tmp_path,
        has_config=True,
        has_catalog=True,
        gateway=fresh_gateway,
    )

    result = plugin.execute(ctx)

    # Should succeed (may produce empty result but not fail)
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
