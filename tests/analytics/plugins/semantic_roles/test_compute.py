"""Tests for SemanticRolesPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.analytics.plugins.semantic_roles.compute import SemanticRolesPlugin
from codeintel.config.steps_analytics import SemanticRolesStepConfig
from tests._helpers.factories import make_step_config

# Test constants (non-repo/commit)
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_PROVIDES_COUNT = 1
EXPECTED_REQUIRES_COUNT = 1
EXPECTED_DEPENDS_ON_COUNT = 1
EXPECTED_TAGS_COUNT = 3
MAX_RUNTIME_MS = 90_000
PRIORITY_VALUE = 50


def _create_config(tmp_path: Path | None = None) -> SemanticRolesStepConfig:
    """Create a test configuration.

    Parameters
    ----------
    tmp_path
        Optional temp path for repo root.

    Returns
    -------
    SemanticRolesStepConfig
        Test configuration.
    """
    return make_step_config(SemanticRolesStepConfig, tmp_path)


def _create_mock_context(
    *,
    has_config: bool = True,
    has_catalog: bool = False,
    has_ast: bool = False,
    has_features: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_catalog
        Whether CatalogProvider is available.
    has_ast
        Whether AstProvider is available.
    has_features
        Whether FeaturesProvider is available.

    Returns
    -------
    MagicMock
        Mock execution context.
    """
    ctx = MagicMock()
    ctx.has_config.return_value = has_config

    if has_config:
        ctx.get_config.return_value = _create_config()
    else:
        ctx.get_config.side_effect = ValueError("Config not found")

    # Resource availability
    def has_resource_by_name(name: str) -> bool:
        resource_map = {
            "CatalogProvider": has_catalog,
            "AstProvider": has_ast,
            "FeaturesProvider": has_features,
        }
        return resource_map.get(name, False)

    ctx.has_resource_by_name.side_effect = has_resource_by_name

    # Mock resource providers
    def require_by_name(name: str) -> object:
        if name == "CatalogProvider":
            provider = MagicMock()
            catalog = MagicMock()
            catalog.module_by_path = {}
            provider.catalog.return_value = catalog
            return provider
        if name == "AstProvider":
            ast_data = MagicMock()
            ast_data.function_ast_map = {}
            return ast_data
        if name == "FeaturesProvider":
            return {}
        msg = f"Resource {name} not found"
        raise ValueError(msg)

    ctx.require_by_name.side_effect = require_by_name

    # Gateway mock
    gateway = MagicMock()
    ctx.gateway = gateway

    return ctx


class TestSemanticRolesPluginMetadata:
    """Tests for SemanticRolesPlugin metadata."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify plugin name is correctly set."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.name == "semantic.roles"

    @staticmethod
    def test_metadata_kind() -> None:
        """Verify plugin kind is analytics."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.kind == "analytics"

    @staticmethod
    def test_metadata_stage() -> None:
        """Verify plugin stage is semantic."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.stage == "semantic"

    @staticmethod
    def test_metadata_version() -> None:
        """Verify plugin version is set."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.version == TEST_VERSION

    @staticmethod
    def test_metadata_enabled_by_default() -> None:
        """Verify plugin is enabled by default."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.enabled_by_default is True

    @staticmethod
    def test_metadata_severity() -> None:
        """Verify plugin severity is fatal."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.severity == "fatal"

    @staticmethod
    def test_metadata_outputs() -> None:
        """Verify plugin outputs are correctly defined."""
        plugin = SemanticRolesPlugin()
        assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT
        assert "analytics.semantic_roles" in plugin.metadata.outputs[0].tables

    @staticmethod
    def test_metadata_provides() -> None:
        """Verify plugin provides semantic_roles table."""
        plugin = SemanticRolesPlugin()
        assert len(plugin.metadata.provides) == EXPECTED_PROVIDES_COUNT
        assert "analytics.semantic_roles" in plugin.metadata.provides

    @staticmethod
    def test_metadata_requires() -> None:
        """Verify plugin requires core.goids."""
        plugin = SemanticRolesPlugin()
        assert len(plugin.metadata.requires) == EXPECTED_REQUIRES_COUNT
        assert "core.goids" in plugin.metadata.requires

    @staticmethod
    def test_metadata_depends_on() -> None:
        """Verify plugin depends on callgraph."""
        plugin = SemanticRolesPlugin()
        assert len(plugin.metadata.depends_on) == EXPECTED_DEPENDS_ON_COUNT
        assert "callgraph" in plugin.metadata.depends_on

    @staticmethod
    def test_metadata_resource_hints() -> None:
        """Verify plugin resource hints are set."""
        plugin = SemanticRolesPlugin()
        assert plugin.metadata.resource_hints is not None
        assert plugin.metadata.resource_hints.max_runtime_ms == MAX_RUNTIME_MS
        assert plugin.metadata.resource_hints.priority == PRIORITY_VALUE

    @staticmethod
    def test_metadata_tags() -> None:
        """Verify plugin tags are set."""
        plugin = SemanticRolesPlugin()
        assert len(plugin.metadata.tags) == EXPECTED_TAGS_COUNT
        assert "semantic" in plugin.metadata.tags
        assert "roles" in plugin.metadata.tags
        assert "classification" in plugin.metadata.tags


class TestSemanticRolesPluginValidation:
    """Tests for SemanticRolesPlugin input validation."""

    @staticmethod
    def test_validate_inputs_succeeds_with_config() -> None:
        """Verify validation succeeds when config is available."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(has_config=True)
        result = plugin.validate_inputs(ctx)
        assert result.valid is True

    @staticmethod
    def test_validate_inputs_fails_without_config() -> None:
        """Verify validation fails when config is missing."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(has_config=False)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        assert any("SemanticRolesStepConfig" in msg for msg in result.errors)


class TestSemanticRolesPluginExecution:
    """Tests for SemanticRolesPlugin execute method."""

    @staticmethod
    def test_execute_fails_without_config() -> None:
        """Verify execute fails when config is missing."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(has_config=False)
        result = plugin.execute(ctx)
        assert result.success is False

    @staticmethod
    def test_execute_handles_no_resources() -> None:
        """Verify execute handles case with no resource providers."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(
            has_config=True,
            has_catalog=False,
            has_ast=False,
            has_features=False,
        )
        # Should not raise - uses empty defaults
        _result = plugin.execute(ctx)
        # Result depends on compute_semantic_roles behavior

    @staticmethod
    def test_execute_uses_catalog_provider() -> None:
        """Verify execute uses CatalogProvider when available."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(
            has_config=True,
            has_catalog=True,
            has_ast=False,
            has_features=False,
        )
        # Should call catalog provider
        _result = plugin.execute(ctx)
        ctx.require_by_name.assert_any_call("CatalogProvider")

    @staticmethod
    def test_execute_uses_ast_provider() -> None:
        """Verify execute uses AstProvider when available."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(
            has_config=True,
            has_catalog=False,
            has_ast=True,
            has_features=False,
        )
        _result = plugin.execute(ctx)
        ctx.require_by_name.assert_any_call("AstProvider")

    @staticmethod
    def test_execute_uses_features_provider() -> None:
        """Verify execute uses FeaturesProvider when available."""
        plugin = SemanticRolesPlugin()
        ctx = _create_mock_context(
            has_config=True,
            has_catalog=False,
            has_ast=False,
            has_features=True,
        )
        _result = plugin.execute(ctx)
        ctx.require_by_name.assert_any_call("FeaturesProvider")
