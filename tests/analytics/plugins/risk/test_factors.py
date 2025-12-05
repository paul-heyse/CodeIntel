"""Tests for RiskFactorsPlugin.

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
from codeintel.analytics.plugins.risk.factors import RiskFactorsPlugin
from tests._helpers.factories import make_snapshot
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
    gateway: StorageGateway | None = None,
) -> PluginExecutionContext:
    """Create a test execution context using real production types.

    Parameters
    ----------
    tmp_path
        Temp path for repo root.
    gateway
        Optional gateway override.

    Returns
    -------
    PluginExecutionContext
        Real execution context.
    """
    if gateway is not None:
        snapshot = make_snapshot(repo_root=tmp_path)
        return TestExecutionContextBuilder(gateway, snapshot).build()
    return TestExecutionContextBuilder.create(tmp_path).build()


# =============================================================================
# Metadata Tests
# =============================================================================


def test_risk_factors_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = RiskFactorsPlugin()
    assert plugin.metadata.name == "risk_factors.build"


def test_risk_factors_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = RiskFactorsPlugin()
    assert plugin.metadata.stage == "risk"


def test_risk_factors_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = RiskFactorsPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_risk_factors_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = RiskFactorsPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "goid_risk_factors" in output_names


def test_risk_factors_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = RiskFactorsPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.goid_risk_factors" in plugin.metadata.provides


def test_risk_factors_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = RiskFactorsPlugin()
    assert "risk" in plugin.metadata.tags
    assert "factors" in plugin.metadata.tags
    assert "scoring" in plugin.metadata.tags


def test_risk_factors_plugin_metadata_depends_on() -> None:
    """Plugin metadata has correct dependencies."""
    plugin = RiskFactorsPlugin()
    assert "functions.metrics" in plugin.metadata.depends_on
    assert "coverage.functions" in plugin.metadata.depends_on


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_always_succeeds(tmp_path: Path) -> None:
    """Validation always succeeds for this plugin."""
    plugin = RiskFactorsPlugin()
    ctx = _create_context(tmp_path)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_succeeds_without_catalog(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds without catalog provider."""
    plugin = RiskFactorsPlugin()
    ctx = _create_context(tmp_path, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is True


def test_execute_succeeds_with_context(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with standard context."""
    plugin = RiskFactorsPlugin()
    ctx = _create_context(tmp_path, gateway=fresh_gateway)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = RiskFactorsPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = RiskFactorsPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
