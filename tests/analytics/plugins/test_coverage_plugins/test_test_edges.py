"""Tests for CoverageTestEdgesPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from codeintel.analytics.core.protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.plugins.coverage.test_edges import (
    CoverageTestEdgesPlugin,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import TestCoverageStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "2.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_config(tmp_path: Path | None = None) -> TestCoverageStepConfig:
    """Create a test configuration.

    Parameters
    ----------
    tmp_path
        Temporary path for coverage file.

    Returns
    -------
    TestCoverageStepConfig
        Test configuration.
    """
    repo_root = tmp_path or Path("/test/repo")
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=repo_root)
    coverage_file = repo_root / "coverage.json" if tmp_path else None
    return TestCoverageStepConfig(snapshot=snapshot, coverage_file=coverage_file)


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
    return provider


def _create_mock_context(
    *,
    has_config: bool = True,
    has_catalog: bool = False,
    tmp_path: Path | None = None,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_catalog
        Whether catalog provider is available.
    tmp_path
        Temporary path for coverage file.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    resource_map = {
        "CatalogProvider": has_catalog,
    }

    ctx.has_config.return_value = has_config
    if has_config:
        ctx.get_config.return_value = _create_config(tmp_path)
    else:
        ctx.get_config.side_effect = ValueError("Config not found")

    ctx.has_resource_by_name.side_effect = lambda n: resource_map.get(n, False)
    ctx.require_by_name.side_effect = _create_mock_provider
    ctx.gateway = MagicMock()

    return ctx


# =============================================================================
# Metadata Tests
# =============================================================================


def test_coverage_test_edges_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = CoverageTestEdgesPlugin()
    assert plugin.metadata.name == "coverage.test_edges"


def test_coverage_test_edges_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = CoverageTestEdgesPlugin()
    assert plugin.metadata.stage == "coverage"


def test_coverage_test_edges_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = CoverageTestEdgesPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_coverage_test_edges_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = CoverageTestEdgesPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "test_edges" in output_names


def test_coverage_test_edges_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = CoverageTestEdgesPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "coverage.test_edges" in plugin.metadata.provides


def test_coverage_test_edges_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = CoverageTestEdgesPlugin()
    assert "coverage" in plugin.metadata.tags
    assert "tests" in plugin.metadata.tags
    assert "edges" in plugin.metadata.tags


def test_coverage_test_edges_plugin_metadata_depends_on() -> None:
    """Plugin metadata has correct dependencies."""
    plugin = CoverageTestEdgesPlugin()
    assert "coverage_ingest" in plugin.metadata.depends_on
    assert "tests_ingest" in plugin.metadata.depends_on


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config() -> None:
    """Validation succeeds when config is present."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_mock_context(has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config() -> None:
    """Validation fails when config is missing."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details() -> None:
    """Validation returns specific error messages."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "TestCoverageStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config() -> None:
    """Execute fails when config is not available."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_mock_context(has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_without_catalog_provider(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds without optional catalog provider."""
    plugin = CoverageTestEdgesPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.return_value = False

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_succeeds_with_catalog_provider(
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with catalog provider."""
    plugin = CoverageTestEdgesPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = _create_config()
    ctx.has_resource_by_name.side_effect = lambda n: n == "CatalogProvider"
    ctx.require_by_name.side_effect = _create_mock_provider

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_handles_error_gracefully(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Execute handles errors from domain function gracefully."""
    plugin = CoverageTestEdgesPlugin()

    # Create a non-existent coverage file path to trigger error handling
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=tmp_path)
    bad_config = TestCoverageStepConfig(
        snapshot=snapshot,
        coverage_file=tmp_path / "nonexistent" / "coverage.db",
    )

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.has_config.return_value = True
    ctx.get_config.return_value = bad_config
    ctx.has_resource_by_name.return_value = False

    result = plugin.execute(ctx)

    # Should handle the error - either succeed (no coverage) or fail gracefully
    assert isinstance(result, PluginResult)


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = CoverageTestEdgesPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = CoverageTestEdgesPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage


def test_plugin_capabilities_required() -> None:
    """Plugin requires coverage.lines capability."""
    plugin = CoverageTestEdgesPlugin()

    assert "coverage.lines" in plugin.metadata.requires
