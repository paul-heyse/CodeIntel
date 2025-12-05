"""Coverage tests for config_plugin.

This module provides comprehensive tests for ConfigIngestPlugin,
covering both successful execution and error handling paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.infrastructure.scanning import default_config_profile
from codeintel.ingestion.plugins.config_plugin import ConfigIngestPlugin
from tests._helpers.fakes import create_test_build_paths
from tests._helpers.harnesses import IngestPluginTestHarness

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Plugin Metadata Tests
# =============================================================================


class TestConfigIngestPluginMetadata:
    """Tests for ConfigIngestPlugin metadata."""

    def test_plugin_name(self) -> None:
        """Plugin has correct name."""
        plugin = ConfigIngestPlugin()
        assert plugin.metadata.name == "config_ingest"

    def test_plugin_stage(self) -> None:
        """Plugin is in enrich stage."""
        plugin = ConfigIngestPlugin()
        assert plugin.metadata.stage == "enrich"

    def test_plugin_version(self) -> None:
        """Plugin has version class attribute."""
        plugin = ConfigIngestPlugin()
        assert plugin.plugin_version == "2.0.0"

    def test_output_tables(self) -> None:
        """Plugin produces config_values table."""
        plugin = ConfigIngestPlugin()
        assert "core.config_values" in plugin.metadata.produces_tables

    def test_dependencies(self) -> None:
        """Plugin depends on repo_scan."""
        plugin = ConfigIngestPlugin()
        assert "repo_scan" in plugin.metadata.depends_on

    def test_requirements(self) -> None:
        """Plugin requires change_tracker capability."""
        plugin = ConfigIngestPlugin()
        assert "change_tracker" in plugin.metadata.requires

    def test_supports_incremental(self) -> None:
        """Plugin supports incremental mode."""
        plugin = ConfigIngestPlugin()
        assert plugin.supports_incremental is True

    def test_resource_hints(self) -> None:
        """Plugin has correct resource hints."""
        plugin = ConfigIngestPlugin()
        hints = plugin.resource_hints
        assert hints.cpu_intensive is False
        assert hints.io_intensive is True


# =============================================================================
# Plugin Compute Tests
# =============================================================================


class TestConfigIngestPluginCompute:
    """Tests for ConfigIngestPlugin.compute method."""

    def test_no_config_files_returns_empty(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Compute returns empty dict when no config files found."""
        plugin = ConfigIngestPlugin()

        # Create a repo with no config files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text("# Python file")

        snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        config_profile = default_config_profile(tmp_path)

        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            paths=paths,
            config_profile=config_profile,
        )

        result = plugin.compute(ctx)

        assert result == {}


class TestConfigIngestPluginWithHarness:
    """Tests using IngestPluginTestHarness."""

    def test_harness_setup(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Verify harness can set up context for plugin."""
        plugin = ConfigIngestPlugin()
        paths = create_test_build_paths(tmp_path)

        harness = (
            IngestPluginTestHarness.for_plugin(plugin)
            .with_gateway(fresh_gateway)
            .with_snapshot("test/repo", "abc123", tmp_path)
            .with_build_dir(paths.build_dir)
        )

        ctx = harness.build_context()

        assert ctx.repo == "test/repo"
        assert ctx.commit == "abc123"
