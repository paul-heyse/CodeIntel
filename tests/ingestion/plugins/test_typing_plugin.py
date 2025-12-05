"""Coverage tests for typing_plugin.

This module provides comprehensive tests for TypingIngestPlugin metadata
and basic execution paths.
"""

from __future__ import annotations

from codeintel.ingestion.plugins.typing_plugin import TypingIngestPlugin

# =============================================================================
# Plugin Metadata Tests
# =============================================================================


class TestTypingIngestPluginMetadata:
    """Tests for TypingIngestPlugin metadata."""

    def test_plugin_name(self) -> None:
        """Plugin has correct name."""
        plugin = TypingIngestPlugin()
        assert plugin.metadata.name == "typing_ingest"

    def test_plugin_stage(self) -> None:
        """Plugin is in enrich stage."""
        plugin = TypingIngestPlugin()
        assert plugin.metadata.stage == "enrich"

    def test_plugin_version(self) -> None:
        """Plugin has version class attribute."""
        plugin = TypingIngestPlugin()
        assert plugin.plugin_version == "2.0.0"

    def test_output_tables(self) -> None:
        """Plugin produces typedness and diagnostics tables."""
        plugin = TypingIngestPlugin()
        assert "analytics.typedness" in plugin.metadata.produces_tables
        assert "analytics.static_diagnostics" in plugin.metadata.produces_tables

    def test_dependencies(self) -> None:
        """Plugin depends on repo_scan."""
        plugin = TypingIngestPlugin()
        assert "repo_scan" in plugin.metadata.depends_on

    def test_requirements(self) -> None:
        """Plugin requires change_tracker capability."""
        plugin = TypingIngestPlugin()
        assert "change_tracker" in plugin.metadata.requires

    def test_tool_dependencies(self) -> None:
        """Plugin requires type checking tools."""
        plugin = TypingIngestPlugin()
        assert "pyright" in plugin.tool_dependencies
        assert "pyrefly" in plugin.tool_dependencies
        assert "ruff" in plugin.tool_dependencies

    def test_supports_incremental(self) -> None:
        """Plugin supports incremental mode."""
        plugin = TypingIngestPlugin()
        assert plugin.supports_incremental is True

    def test_tracker_required(self) -> None:
        """Plugin requires tracker."""
        plugin = TypingIngestPlugin()
        assert plugin.tracker_required is True

    def test_tool_required_is_false(self) -> None:
        """Plugin doesn't strictly require tools (graceful degradation)."""
        plugin = TypingIngestPlugin()
        assert plugin.tool_required is False

    def test_resource_hints(self) -> None:
        """Plugin has correct resource hints."""
        plugin = TypingIngestPlugin()
        hints = plugin.resource_hints
        assert hints.cpu_intensive is False
        assert hints.io_intensive is True
        assert hints.max_runtime_ms == 180000

    def test_description(self) -> None:
        """Plugin has meaningful description."""
        plugin = TypingIngestPlugin()
        assert "typedness" in plugin.plugin_description.lower()
        assert "diagnostics" in plugin.plugin_description.lower()


# =============================================================================
# Plugin Instance Tests
# =============================================================================


class TestTypingIngestPluginInstance:
    """Tests for TypingIngestPlugin instance behavior."""

    def test_multiple_instances_independent(self) -> None:
        """Multiple plugin instances are independent."""
        plugin1 = TypingIngestPlugin()
        plugin2 = TypingIngestPlugin()

        assert plugin1 is not plugin2
        assert plugin1.metadata.name == plugin2.metadata.name

    def test_plugin_inheritance(self) -> None:
        """Plugin has correct base classes."""
        plugin = TypingIngestPlugin()

        # Check key capabilities
        assert hasattr(plugin, "compute")
        assert hasattr(plugin, "metadata")
        assert hasattr(plugin, "supports_incremental")
        assert hasattr(plugin, "tool_dependencies")
