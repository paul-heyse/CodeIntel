"""Tests for HistoryTimeseriesPlugin.

This module tests:
- Plugin metadata correctness
- Input validation behavior
- Execute method with various resource availability scenarios
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.analytics.plugins.history.timeseries import HistoryTimeseriesPlugin
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import HistoryTimeseriesStepConfig

# Test constants
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"
TEST_VERSION = "2.0.0"
EXPECTED_OUTPUT_COUNT = 1
EXPECTED_PROVIDES_COUNT = 1
EXPECTED_REQUIRES_COUNT = 1
EXPECTED_DEPENDS_ON_COUNT = 1
EXPECTED_TAGS_COUNT = 3
MAX_RUNTIME_MS = 120_000
PRIORITY_VALUE = 80


def _create_config() -> HistoryTimeseriesStepConfig:
    """Create a test configuration.

    Returns
    -------
    HistoryTimeseriesStepConfig
        Test configuration.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=Path("/test/repo"))
    return HistoryTimeseriesStepConfig(snapshot=snapshot, commits=(TEST_COMMIT,))


def _create_mock_context(
    *,
    has_config: bool = True,
    has_snapshot_resolver: bool = False,
) -> MagicMock:
    """Create a mock execution context.

    Parameters
    ----------
    has_config
        Whether config is available.
    has_snapshot_resolver
        Whether history_snapshot_resolver is in extra.

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

    # Extra dict
    extra = {}
    if has_snapshot_resolver:
        resolver = MagicMock()
        extra["history_snapshot_resolver"] = resolver
    ctx.extra = extra

    # Gateway mock
    gateway = MagicMock()
    ctx.gateway = gateway

    return ctx


class TestHistoryTimeseriesPluginMetadata:
    """Tests for HistoryTimeseriesPlugin metadata."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify plugin name is correctly set."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.name == "history.timeseries"

    @staticmethod
    def test_metadata_kind() -> None:
        """Verify plugin kind is analytics."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.kind == "analytics"

    @staticmethod
    def test_metadata_stage() -> None:
        """Verify plugin stage is history."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.stage == "history"

    @staticmethod
    def test_metadata_version() -> None:
        """Verify plugin version is set."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.version == TEST_VERSION

    @staticmethod
    def test_metadata_enabled_by_default() -> None:
        """Verify plugin is enabled by default."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.enabled_by_default is True

    @staticmethod
    def test_metadata_severity() -> None:
        """Verify plugin severity is fatal."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.severity == "fatal"

    @staticmethod
    def test_metadata_outputs() -> None:
        """Verify plugin outputs are correctly defined."""
        plugin = HistoryTimeseriesPlugin()
        assert len(plugin.metadata.outputs) == EXPECTED_OUTPUT_COUNT
        assert "analytics.history_timeseries" in plugin.metadata.outputs[0].tables

    @staticmethod
    def test_metadata_provides() -> None:
        """Verify plugin provides history_timeseries table."""
        plugin = HistoryTimeseriesPlugin()
        assert len(plugin.metadata.provides) == EXPECTED_PROVIDES_COUNT
        assert "analytics.history_timeseries" in plugin.metadata.provides

    @staticmethod
    def test_metadata_requires() -> None:
        """Verify plugin requires function_profile."""
        plugin = HistoryTimeseriesPlugin()
        assert len(plugin.metadata.requires) == EXPECTED_REQUIRES_COUNT
        assert "analytics.function_profile" in plugin.metadata.requires

    @staticmethod
    def test_metadata_depends_on() -> None:
        """Verify plugin depends on profiles.build."""
        plugin = HistoryTimeseriesPlugin()
        assert len(plugin.metadata.depends_on) == EXPECTED_DEPENDS_ON_COUNT
        assert "profiles.build" in plugin.metadata.depends_on

    @staticmethod
    def test_metadata_resource_hints() -> None:
        """Verify plugin resource hints are set."""
        plugin = HistoryTimeseriesPlugin()
        assert plugin.metadata.resource_hints.max_runtime_ms == MAX_RUNTIME_MS
        assert plugin.metadata.resource_hints.priority == PRIORITY_VALUE

    @staticmethod
    def test_metadata_tags() -> None:
        """Verify plugin tags are set."""
        plugin = HistoryTimeseriesPlugin()
        assert len(plugin.metadata.tags) == EXPECTED_TAGS_COUNT
        assert "history" in plugin.metadata.tags
        assert "timeseries" in plugin.metadata.tags
        assert "trends" in plugin.metadata.tags


class TestHistoryTimeseriesPluginValidation:
    """Tests for HistoryTimeseriesPlugin input validation."""

    @staticmethod
    def test_validate_inputs_succeeds_with_config_and_resolver() -> None:
        """Verify validation succeeds when config and resolver available."""
        plugin = HistoryTimeseriesPlugin()
        ctx = _create_mock_context(has_config=True, has_snapshot_resolver=True)
        result = plugin.validate_inputs(ctx)
        assert result.valid is True

    @staticmethod
    def test_validate_inputs_fails_without_config() -> None:
        """Verify validation fails when config is missing."""
        plugin = HistoryTimeseriesPlugin()
        ctx = _create_mock_context(has_config=False, has_snapshot_resolver=True)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        assert any("HistoryTimeseriesStepConfig" in msg for msg in result.errors)

    @staticmethod
    def test_validate_inputs_fails_without_resolver() -> None:
        """Verify validation fails when resolver is missing."""
        plugin = HistoryTimeseriesPlugin()
        ctx = _create_mock_context(has_config=True, has_snapshot_resolver=False)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        assert any("history_snapshot_resolver" in msg for msg in result.errors)

    @staticmethod
    def test_validate_inputs_fails_without_both() -> None:
        """Verify validation fails with multiple errors when both missing."""
        plugin = HistoryTimeseriesPlugin()
        ctx = _create_mock_context(has_config=False, has_snapshot_resolver=False)
        result = plugin.validate_inputs(ctx)
        assert result.valid is False
        # Both errors should be present
        error_text = " ".join(result.errors)
        assert "HistoryTimeseriesStepConfig" in error_text
        assert "history_snapshot_resolver" in error_text


class TestHistoryTimeseriesPluginExecution:
    """Tests for HistoryTimeseriesPlugin execute method."""

    @staticmethod
    def test_execute_fails_without_config() -> None:
        """Verify execute fails when config is missing."""
        plugin = HistoryTimeseriesPlugin()
        ctx = _create_mock_context(has_config=False, has_snapshot_resolver=True)
        result = plugin.execute(ctx)
        assert result.success is False

    @staticmethod
    def test_execute_fails_without_resolver() -> None:
        """Verify execute fails when resolver is missing."""
        plugin = HistoryTimeseriesPlugin()
        ctx = _create_mock_context(has_config=True, has_snapshot_resolver=False)
        result = plugin.execute(ctx)
        assert result.success is False
        assert result.error is not None
        assert "history_snapshot_resolver" in result.error
