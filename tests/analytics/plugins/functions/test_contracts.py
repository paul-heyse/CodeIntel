"""Tests for FunctionContractsPlugin.

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
from codeintel.analytics.plugins.functions.contracts import (
    FunctionContractsPlugin,
)
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.config.steps_analytics import FunctionContractsStepConfig
from tests._helpers.factories import make_snapshot, make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.storage.gateway import StorageGateway

# Test constants (non-repo/commit)
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_CAPABILITY_COUNT = 1


def _create_context(
    tmp_path: Path,
    *,
    has_config: bool = True,
    has_ast: bool = False,
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
    has_ast
        Whether AST provider is available.
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
        config = make_step_config(FunctionContractsStepConfig, tmp_path)
        builder.with_config(FunctionContractsStepConfig, config)

    if has_ast:
        # Create AST provider with empty preloaded data
        ast_provider = AstProvider.from_asts({}, set())
        builder.with_resource(AstProvider, ast_provider)

    if has_catalog:
        # Create catalog provider with empty mock catalog
        catalog_provider = CatalogProvider()
        catalog_provider.set_preloaded(MockFunctionCatalog())
        builder.with_resource(CatalogProvider, catalog_provider)

    return builder.build()


# =============================================================================
# Metadata Tests
# =============================================================================


def test_contracts_plugin_metadata_name() -> None:
    """Plugin metadata has correct name."""
    plugin = FunctionContractsPlugin()
    assert plugin.metadata.name == "functions.contracts"


def test_contracts_plugin_metadata_stage() -> None:
    """Plugin metadata has correct stage."""
    plugin = FunctionContractsPlugin()
    assert plugin.metadata.stage == "function"


def test_contracts_plugin_metadata_version() -> None:
    """Plugin metadata has correct version."""
    plugin = FunctionContractsPlugin()
    assert plugin.metadata.version == TEST_VERSION


def test_contracts_plugin_metadata_outputs() -> None:
    """Plugin metadata has correct outputs."""
    plugin = FunctionContractsPlugin()
    assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT

    output_names = {o.name for o in plugin.metadata.outputs}
    assert "function_contracts" in output_names


def test_contracts_plugin_metadata_capabilities_provided() -> None:
    """Plugin metadata provides correct capabilities."""
    plugin = FunctionContractsPlugin()
    assert len(plugin.metadata.provides) == EXPECTED_CAPABILITY_COUNT

    assert "analytics.function_contracts" in plugin.metadata.provides


def test_contracts_plugin_metadata_tags() -> None:
    """Plugin metadata has correct tags."""
    plugin = FunctionContractsPlugin()
    assert "functions" in plugin.metadata.tags
    assert "contracts" in plugin.metadata.tags
    assert "nullability" in plugin.metadata.tags


# =============================================================================
# Validation Tests
# =============================================================================


def test_validate_inputs_success_with_config(tmp_path: Path) -> None:
    """Validation succeeds when config is present."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(tmp_path, has_config=True)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is True


def test_validate_inputs_failure_without_config(tmp_path: Path) -> None:
    """Validation fails when config is missing."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert isinstance(result, ValidationResult)
    assert result.valid is False


def test_validate_inputs_returns_error_details(tmp_path: Path) -> None:
    """Validation returns specific error messages."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.validate_inputs(ctx)

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0
    assert "FunctionContractsStepConfig" in result.errors[0]


# =============================================================================
# Execute Tests
# =============================================================================


def test_execute_fails_without_config(tmp_path: Path) -> None:
    """Execute fails when config is not available."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(tmp_path, has_config=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False


def test_execute_fails_without_ast_provider(tmp_path: Path) -> None:
    """Execute fails when AST provider is not available."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(tmp_path, has_config=True, has_ast=False, has_catalog=True)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert "AstProvider is required" in str(result.error)


def test_execute_fails_without_catalog_provider(tmp_path: Path) -> None:
    """Execute fails when catalog provider is not available."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(tmp_path, has_config=True, has_ast=True, has_catalog=False)

    result = plugin.execute(ctx)

    assert isinstance(result, PluginResult)
    assert result.success is False
    assert "CatalogProvider is required" in str(result.error)


def test_execute_succeeds_with_all_resources(
    tmp_path: Path,
    fresh_gateway: StorageGateway,
) -> None:
    """Execute succeeds with all required resources."""
    plugin = FunctionContractsPlugin()
    ctx = _create_context(
        tmp_path,
        has_config=True,
        has_ast=True,
        has_catalog=True,
        gateway=fresh_gateway,
    )

    result = plugin.execute(ctx)

    assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_plugin_is_dataclass() -> None:
    """Plugin is a dataclass."""
    plugin = FunctionContractsPlugin()
    assert hasattr(plugin, "__dataclass_fields__")


def test_plugin_metadata_is_consistent() -> None:
    """Plugin metadata is consistent between calls."""
    plugin = FunctionContractsPlugin()

    meta1 = plugin.metadata
    meta2 = plugin.metadata

    assert meta1.name == meta2.name
    assert meta1.version == meta2.version
    assert meta1.stage == meta2.stage
