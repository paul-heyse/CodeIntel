"""Coverage tests for tests_plugin.

This module provides comprehensive tests for TestsIngestPlugin metadata
and basic execution paths.
"""

from __future__ import annotations

from codeintel.ingestion.plugins.tests_plugin import TestsIngestPlugin

# =============================================================================
# Plugin Metadata Tests
# =============================================================================


class TestTestsIngestPluginMetadata:
    """Tests for TestsIngestPlugin metadata."""

    def test_plugin_name(self) -> None:
        """Plugin has correct name."""
        plugin = TestsIngestPlugin()
        assert plugin.metadata.name == "tests_ingest"

    def test_plugin_stage(self) -> None:
        """Plugin is in enrich stage."""
        plugin = TestsIngestPlugin()
        assert plugin.metadata.stage == "enrich"

    def test_plugin_version(self) -> None:
        """Plugin has version class attribute."""
        plugin = TestsIngestPlugin()
        assert plugin.plugin_version == "2.0.0"

    def test_output_tables(self) -> None:
        """Plugin produces test_results table."""
        plugin = TestsIngestPlugin()
        assert "analytics.test_results" in plugin.metadata.produces_tables

    def test_dependencies(self) -> None:
        """Plugin depends on repo_scan."""
        plugin = TestsIngestPlugin()
        assert "repo_scan" in plugin.metadata.depends_on

    def test_requirements(self) -> None:
        """Plugin requires change_tracker capability."""
        plugin = TestsIngestPlugin()
        assert "change_tracker" in plugin.metadata.requires

    def test_tool_dependencies(self) -> None:
        """Plugin requires pytest tool."""
        plugin = TestsIngestPlugin()
        assert "pytest" in plugin.tool_dependencies

    def test_supports_incremental(self) -> None:
        """Plugin supports incremental mode."""
        plugin = TestsIngestPlugin()
        assert plugin.supports_incremental is True

    def test_tracker_required_is_false(self) -> None:
        """Plugin does not strictly require tracker."""
        plugin = TestsIngestPlugin()
        assert plugin.tracker_required is False

    def test_tool_required_is_false(self) -> None:
        """Plugin doesn't strictly require tools (graceful degradation)."""
        plugin = TestsIngestPlugin()
        assert plugin.tool_required is False

    def test_resource_hints(self) -> None:
        """Plugin has correct resource hints."""
        plugin = TestsIngestPlugin()
        hints = plugin.resource_hints
        assert hints.cpu_intensive is False
        assert hints.io_intensive is True

    def test_description(self) -> None:
        """Plugin has meaningful description."""
        plugin = TestsIngestPlugin()
        assert "pytest" in plugin.plugin_description.lower()


# =============================================================================
# Plugin Instance Tests
# =============================================================================


class TestTestsIngestPluginInstance:
    """Tests for TestsIngestPlugin instance behavior."""

    def test_multiple_instances_independent(self) -> None:
        """Multiple plugin instances are independent."""
        plugin1 = TestsIngestPlugin()
        plugin2 = TestsIngestPlugin()

        assert plugin1 is not plugin2
        assert plugin1.metadata.name == plugin2.metadata.name

    def test_plugin_inheritance(self) -> None:
        """Plugin has correct base classes."""
        plugin = TestsIngestPlugin()

        # Check key capabilities
        assert hasattr(plugin, "compute")
        assert hasattr(plugin, "metadata")
        assert hasattr(plugin, "supports_incremental")
        assert hasattr(plugin, "tool_dependencies")
