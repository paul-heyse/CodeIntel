"""Tests for SubsystemsPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.analytics.plugins.subsystems.build import SubsystemsPlugin
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import SubsystemsStepConfig

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "3.0.0"
EXPECTED_OUTPUT_COUNT = 3
EXPECTED_PROVIDES_COUNT = 3
EXPECTED_REQUIRES_COUNT = 1
EXPECTED_DEPENDS_ON_COUNT = 3
EXPECTED_TAGS_COUNT = 3
MAX_RUNTIME_MS = 120_000
PRIORITY_VALUE = 60


def _create_config() -> SubsystemsStepConfig:
    """Create a test configuration.

    Returns
    -------
    SubsystemsStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return SubsystemsStepConfig(snapshot=snapshot)


def _create_mock_context(
    *,
    has_config: bool = True,
    has_graph_provider: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_graph_provider
        Whether GraphProvider is available.

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
            "GraphProvider": has_graph_provider,
        }
        return resource_map.get(name, False)

    ctx.has_resource_by_name.side_effect = has_resource_by_name

    # Mock resource providers
    def require_by_name(name: str) -> object:
        if name == "GraphProvider":
            provider = MagicMock()
            provider.runtime = MagicMock()
            return provider
        msg = f"Resource {name} not found"
        raise ValueError(msg)

    ctx.require_by_name.side_effect = require_by_name

    # Gateway mock
    gateway = MagicMock()
    ctx.gateway = gateway

    return ctx


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
    def test_validate_inputs_succeeds_with_config() -> None:
        """Verify validation succeeds when config is available."""
        plugin = SubsystemsPlugin()
        ctx = _create_mock_context(has_config=True)
        result = plugin.validate_inputs(ctx)
        assert result.valid is True

    @staticmethod
    def test_validate_inputs_fails_without_config() -> None:
        """Verify validation fails when config is missing."""
        plugin = SubsystemsPlugin()
        ctx = _create_mock_context(has_config=False)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        assert any("SubsystemsStepConfig" in msg for msg in result.errors)


class TestSubsystemsPluginExecution:
    """Tests for SubsystemsPlugin execute method."""

    @staticmethod
    def test_execute_fails_without_config() -> None:
        """Verify execute fails when config is missing."""
        plugin = SubsystemsPlugin()
        ctx = _create_mock_context(has_config=False)
        result = plugin.execute(ctx)
        assert result.success is False

    @staticmethod
    def test_execute_handles_no_graph_provider() -> None:
        """Verify execute handles case with no GraphProvider."""
        plugin = SubsystemsPlugin()
        ctx = _create_mock_context(
            has_config=True,
            has_graph_provider=False,
        )
        # Should not raise - uses None for runtime
        _result = plugin.execute(ctx)
        # Result depends on build_subsystems behavior

    @staticmethod
    def test_execute_uses_graph_provider() -> None:
        """Verify execute uses GraphProvider when available."""
        plugin = SubsystemsPlugin()
        ctx = _create_mock_context(
            has_config=True,
            has_graph_provider=True,
        )
        # Should call graph provider
        _result = plugin.execute(ctx)
        ctx.require_by_name.assert_any_call("GraphProvider")
