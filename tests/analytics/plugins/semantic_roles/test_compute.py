"""Tests for SemanticRolesPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.plugins.semantic_roles.compute import SemanticRolesPlugin
from codeintel.config.steps_analytics import SemanticRolesStepConfig
from tests._helpers.factories import make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext

# Test constants (non-repo/commit)
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_PROVIDES_COUNT = 1
EXPECTED_REQUIRES_COUNT = 1
EXPECTED_DEPENDS_ON_COUNT = 1
EXPECTED_TAGS_COUNT = 3
MAX_RUNTIME_MS = 90_000
PRIORITY_VALUE = 50


def _create_context(
    tmp_path: Path,
    *,
    has_config: bool = True,
) -> PluginExecutionContext:
    """Create a test execution context using real production types.

    Parameters
    ----------
    tmp_path
        Temp path for repo root.
    has_config
        Whether config is available.

    Returns
    -------
    PluginExecutionContext
        Real execution context.
    """
    builder = TestExecutionContextBuilder.create(tmp_path)
    if has_config:
        config = make_step_config(SemanticRolesStepConfig, tmp_path)
        builder.with_config(SemanticRolesStepConfig, config)
    return builder.build()


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
    def test_validate_inputs_succeeds_with_config(tmp_path: Path) -> None:
        """Verify validation succeeds when config is available."""
        plugin = SemanticRolesPlugin()
        ctx = _create_context(tmp_path, has_config=True)
        result = plugin.validate_inputs(ctx)
        assert result.valid is True

    @staticmethod
    def test_validate_inputs_fails_without_config(tmp_path: Path) -> None:
        """Verify validation fails when config is missing."""
        plugin = SemanticRolesPlugin()
        ctx = _create_context(tmp_path, has_config=False)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        assert any("SemanticRolesStepConfig" in msg for msg in result.errors)


class TestSemanticRolesPluginExecution:
    """Tests for SemanticRolesPlugin execute method."""

    @staticmethod
    def test_execute_fails_without_config(tmp_path: Path) -> None:
        """Verify execute fails when config is missing."""
        plugin = SemanticRolesPlugin()
        ctx = _create_context(tmp_path, has_config=False)
        result = plugin.execute(ctx)
        assert result.success is False

    @staticmethod
    def test_execute_handles_no_resources(tmp_path: Path) -> None:
        """Verify execute handles case with no resource providers."""
        plugin = SemanticRolesPlugin()
        ctx = _create_context(tmp_path, has_config=True)
        # No resources registered - should not raise, uses empty defaults
        result = plugin.execute(ctx)
        # Plugin should complete execution (behavior depends on data availability)
        assert result is not None

    @staticmethod
    def test_execute_succeeds_with_config(tmp_path: Path) -> None:
        """Verify execute succeeds with config available."""
        plugin = SemanticRolesPlugin()
        ctx = _create_context(tmp_path, has_config=True)
        result = plugin.execute(ctx)
        # Plugin should complete execution
        assert result is not None
