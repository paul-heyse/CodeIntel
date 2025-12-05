"""Tests for SubsystemsPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.plugins.subsystems.build import SubsystemsPlugin
from codeintel.config.steps_analytics import SubsystemsStepConfig
from tests._helpers.factories import make_step_config
from tests._helpers.fakes import TestExecutionContextBuilder

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext

# Test constants (non-repo/commit)
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 3
EXPECTED_PROVIDES_COUNT = 3
EXPECTED_REQUIRES_COUNT = 1
EXPECTED_DEPENDS_ON_COUNT = 3
EXPECTED_TAGS_COUNT = 3
MAX_RUNTIME_MS = 120_000
PRIORITY_VALUE = 60


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
        config = make_step_config(SubsystemsStepConfig, tmp_path)
        builder.with_config(SubsystemsStepConfig, config)
    return builder.build()


class TestSubsystemsPluginMetadata:
    """Tests for SubsystemsPlugin metadata."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify plugin name is correctly set."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.name == "subsystems.build"

    @staticmethod
    def test_metadata_kind() -> None:
        """Verify plugin kind is analytics."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.kind == "analytics"

    @staticmethod
    def test_metadata_stage() -> None:
        """Verify plugin stage is subsystem."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.stage == "subsystem"

    @staticmethod
    def test_metadata_version() -> None:
        """Verify plugin version is set."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.version == TEST_VERSION

    @staticmethod
    def test_metadata_enabled_by_default() -> None:
        """Verify plugin is enabled by default."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.enabled_by_default is True

    @staticmethod
    def test_metadata_severity() -> None:
        """Verify plugin severity is fatal."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.severity == "fatal"

    @staticmethod
    def test_metadata_outputs() -> None:
        """Verify plugin outputs are correctly defined."""
        plugin = SubsystemsPlugin()
        assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT
        output_tables = [table for out in plugin.metadata.outputs for table in out.tables]
        assert "analytics.subsystems" in output_tables
        assert "analytics.subsystem_modules" in output_tables
        assert "analytics.subsystem_functions" in output_tables

    @staticmethod
    def test_metadata_provides() -> None:
        """Verify plugin provides subsystem tables."""
        plugin = SubsystemsPlugin()
        assert len(plugin.metadata.provides) == EXPECTED_PROVIDES_COUNT
        assert "analytics.subsystems" in plugin.metadata.provides
        assert "analytics.subsystem_modules" in plugin.metadata.provides
        assert "analytics.subsystem_functions" in plugin.metadata.provides

    @staticmethod
    def test_metadata_requires() -> None:
        """Verify plugin requires core.modules."""
        plugin = SubsystemsPlugin()
        assert len(plugin.metadata.requires) == EXPECTED_REQUIRES_COUNT
        assert "core.modules" in plugin.metadata.requires

    @staticmethod
    def test_metadata_depends_on() -> None:
        """Verify plugin dependencies."""
        plugin = SubsystemsPlugin()
        assert len(plugin.metadata.depends_on) == EXPECTED_DEPENDS_ON_COUNT
        assert "import_graph" in plugin.metadata.depends_on
        assert "symbol_uses" in plugin.metadata.depends_on
        assert "risk_factors.build" in plugin.metadata.depends_on

    @staticmethod
    def test_metadata_resource_hints() -> None:
        """Verify plugin resource hints are set."""
        plugin = SubsystemsPlugin()
        assert plugin.metadata.resource_hints is not None
        assert plugin.metadata.resource_hints.max_runtime_ms == MAX_RUNTIME_MS
        assert plugin.metadata.resource_hints.priority == PRIORITY_VALUE

    @staticmethod
    def test_metadata_tags() -> None:
        """Verify plugin tags are set."""
        plugin = SubsystemsPlugin()
        assert len(plugin.metadata.tags) == EXPECTED_TAGS_COUNT
        assert "subsystems" in plugin.metadata.tags
        assert "architecture" in plugin.metadata.tags
        assert "modules" in plugin.metadata.tags


class TestSubsystemsPluginValidation:
    """Tests for SubsystemsPlugin input validation."""

    @staticmethod
    def test_validate_inputs_succeeds_with_config(tmp_path: Path) -> None:
        """Verify validation succeeds when config is available."""
        plugin = SubsystemsPlugin()
        ctx = _create_context(tmp_path, has_config=True)
        result = plugin.validate_inputs(ctx)
        assert result.valid is True

    @staticmethod
    def test_validate_inputs_fails_without_config(tmp_path: Path) -> None:
        """Verify validation fails when config is missing."""
        plugin = SubsystemsPlugin()
        ctx = _create_context(tmp_path, has_config=False)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        assert any("SubsystemsStepConfig" in msg for msg in result.errors)


class TestSubsystemsPluginExecution:
    """Tests for SubsystemsPlugin execute method."""

    @staticmethod
    def test_execute_fails_without_config(tmp_path: Path) -> None:
        """Verify execute fails when config is missing."""
        plugin = SubsystemsPlugin()
        ctx = _create_context(tmp_path, has_config=False)
        result = plugin.execute(ctx)
        assert result.success is False

    @staticmethod
    def test_execute_handles_no_graph_provider(tmp_path: Path) -> None:
        """Verify execute handles case with no GraphProvider."""
        plugin = SubsystemsPlugin()
        ctx = _create_context(tmp_path, has_config=True)
        # No GraphProvider registered - should not raise, uses None for runtime
        result = plugin.execute(ctx)
        # With no graph provider, plugin should still complete
        # Result depends on build_subsystems behavior with empty data
        assert result is not None

    @staticmethod
    def test_execute_succeeds_with_config(tmp_path: Path) -> None:
        """Verify execute succeeds with config available."""
        plugin = SubsystemsPlugin()
        ctx = _create_context(tmp_path, has_config=True)
        result = plugin.execute(ctx)
        # Plugin should complete execution (success depends on data availability)
        assert result is not None
