"""Tests for RiskFactorsPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from codeintel.analytics.core.plugin_protocol import (
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.core.plugins.risk.factors import RiskFactorsPlugin

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "2.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


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
        catalog_data = MagicMock()
        catalog_data.module_by_path = {}
        catalog.catalog.return_value = catalog_data
        provider.get.return_value = catalog
    return provider


def _create_mock_context(
    *,
    has_catalog: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_catalog
        Whether catalog provider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    ctx.repo = TEST_REPO
    ctx.commit = TEST_COMMIT
    ctx.has_resource_by_name.side_effect = lambda n: n == "CatalogProvider" and has_catalog
    ctx.require_by_name.side_effect = _create_mock_provider
    ctx.gateway = MagicMock()
    ctx.gateway.con = MagicMock()
    ctx.gateway.con.execute = MagicMock()
    ctx.gateway.con.executemany = MagicMock()

    return ctx


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
    assert len(plugin.metadata.capabilities_provided) == EXPECTED_CAPABILITY_COUNT

    cap_names = {c.name for c in plugin.metadata.capabilities_provided}
    assert "analytics.goid_risk_factors" in cap_names


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


def test_validate_inputs_always_succeeds() -> None:
    """Validation always succeeds for this plugin."""
    plugin = RiskFactorsPlugin()
    ctx = _create_mock_context()

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_succeeds_without_catalog(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds without catalog provider."""
    plugin = RiskFactorsPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.repo = TEST_REPO
    ctx.commit = TEST_COMMIT
    ctx.has_resource_by_name.return_value = False

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is True


def test_execute_succeeds_with_catalog(fresh_gateway: StorageGateway) -> None:
    """Execute succeeds with catalog provider."""
    plugin = RiskFactorsPlugin()

    ctx = MagicMock()
    ctx.gateway = fresh_gateway
    ctx.repo = TEST_REPO
    ctx.commit = TEST_COMMIT
    ctx.has_resource_by_name.side_effect = lambda n: n == "CatalogProvider"
    ctx.require_by_name.side_effect = _create_mock_provider

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
