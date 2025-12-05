"""Tests for CoverageTestEdgesPlugin.

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
from codeintel.analytics.plugins.coverage.test_edges import (
    CoverageTestEdgesPlugin,
)
from codeintel.config.steps_analytics import TestCoverageStepConfig
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.storage.gateway import StorageGateway

# Test constants (non-repo/commit)
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
    snapshot = make_snapshot(repo_root=tmp_path)
    coverage_file = tmp_path / "coverage.json" if tmp_path else None
    return TestCoverageStepConfig(snapshot=snapshot, coverage_file=coverage_file)


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
        config = _create_config(tmp_path)
        builder.with_config(TestCoverageStepConfig, config)

    return builder.build()


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


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "TestCoverageStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_succeeds_without_catalog_provider(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds without optional catalog provider."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    assert result.success is True


def test_execute_handles_missing_coverage_file(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute handles missing coverage file gracefully."""
    plugin = CoverageTestEdgesPlugin()
    ctx = _create_context(tmp_path, has_config=True, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    # Should handle the missing file - either succeed (no coverage) or complete
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
