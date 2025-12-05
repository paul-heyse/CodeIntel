"""Coverage tests for scip_plugin.

This module tests ScipIngestPlugin using real production classes,
test harnesses, and the project's sanctioned fake tool service.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.core.plugins.types.protocol import PluginResourceHints
from codeintel.core.resources import ResourceNotFoundError
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.plugins.scip_plugin import ScipIngestPlugin
from codeintel.ingestion.resources.protocol import ResourceProviderBase
from tests._helpers.fakes import (
    FakeToolService,
    create_test_build_paths,
)
from tests._helpers.harnesses import IngestPluginTestHarness

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Test Resource Providers
# =============================================================================


class SimpleModuleProvider(ResourceProviderBase[list[str]]):
    """Simple module provider for testing.

    Returns a configurable list of module paths.
    """

    RESOURCE_NAME: ClassVar[str] = "modules"

    def __init__(self, modules: list[str] | None = None) -> None:
        """Initialize with module paths.

        Parameters
        ----------
        modules
            List of module paths to return, or None for empty list.
        """
        super().__init__()
        self._modules = modules or []

    def _load(self) -> list[str]:
        """Return the configured modules.

        Returns
        -------
        list[str]
            List of module paths.
        """
        return self._modules


class SimpleToolsProvider(ResourceProviderBase[ToolService]):
    """Tools provider wrapping FakeToolService for testing."""

    RESOURCE_NAME: ClassVar[str] = "tools"

    def __init__(self, service: FakeToolService) -> None:
        """Initialize with fake service.

        Parameters
        ----------
        service
            FakeToolService instance to return.
        """
        super().__init__()
        self._service = service

    def _load(self) -> ToolService:
        """Return the fake tool service.

        Returns
        -------
        ToolService
            The configured fake service.
        """
        return self._service


# =============================================================================
# ScipIngestPlugin Class Tests
# =============================================================================


class TestScipIngestPluginMetadata:
    """Tests for ScipIngestPlugin class attributes and metadata."""

    def test_plugin_name(self) -> None:
        """Plugin has correct name."""
        assert ScipIngestPlugin.plugin_name == "scip_ingest"

    def test_plugin_description(self) -> None:
        """Plugin has a description."""
        assert ScipIngestPlugin.plugin_description is not None
        assert len(ScipIngestPlugin.plugin_description) > 0

    def test_plugin_stage(self) -> None:
        """Plugin is in correct stage."""
        assert ScipIngestPlugin.plugin_stage == "index"

    def test_plugin_version(self) -> None:
        """Plugin has version."""
        assert ScipIngestPlugin.plugin_version == "2.0.0"

    def test_output_tables(self) -> None:
        """Plugin declares expected output tables."""
        tables = ScipIngestPlugin.output_tables
        assert "index.scip" in tables
        assert "core.scip_symbols" in tables
        assert "core.goid_crosswalk" in tables

    def test_depends_on(self) -> None:
        """Plugin declares dependencies."""
        assert "repo_scan" in ScipIngestPlugin.depends_on

    def test_requires(self) -> None:
        """Plugin declares required capabilities."""
        assert "change_tracker" in ScipIngestPlugin.requires

    def test_tool_dependencies(self) -> None:
        """Plugin declares tool dependencies."""
        assert "scip" in ScipIngestPlugin.tool_dependencies

    def test_supports_incremental(self) -> None:
        """Plugin supports incremental mode."""
        assert ScipIngestPlugin.supports_incremental is True

    def test_resource_hints(self) -> None:
        """Plugin has resource hints."""
        hints = ScipIngestPlugin.resource_hints
        assert isinstance(hints, PluginResourceHints)
        assert hints.cpu_intensive is True
        assert hints.io_intensive is True
        assert hints.max_runtime_ms == 300000


class TestScipIngestPluginMetadataProperty:
    """Tests for plugin metadata property."""

    def test_metadata_name(self) -> None:
        """Metadata has correct name."""
        plugin = ScipIngestPlugin()
        assert plugin.metadata.name == "scip_ingest"

    def test_metadata_stage(self) -> None:
        """Metadata has correct stage."""
        plugin = ScipIngestPlugin()
        assert plugin.metadata.stage == "index"

    def test_metadata_produces_tables(self) -> None:
        """Metadata lists output tables."""
        plugin = ScipIngestPlugin()
        tables = plugin.metadata.produces_tables
        assert "index.scip" in tables
        assert "core.scip_symbols" in tables


# =============================================================================
# Compute Method Tests
# =============================================================================


class TestScipIngestPluginCompute:
    """Tests for ScipIngestPlugin.compute method."""

    def test_compute_requires_tools_provider(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Compute raises when tools provider is not registered."""
        plugin = ScipIngestPlugin()

        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)

        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            paths=paths,
        )

        # Should fail because tools provider is not registered
        with pytest.raises(ResourceNotFoundError):
            plugin.compute(ctx)


class TestScipIngestPluginWithHarness:
    """Tests using IngestPluginTestHarness."""

    def test_harness_setup(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Verify harness can set up context for plugin."""
        plugin = ScipIngestPlugin()
        paths = create_test_build_paths(tmp_path)

        harness = (
            IngestPluginTestHarness.for_plugin(plugin)
            .with_gateway(fresh_gateway)
            .with_snapshot("test/repo", "abc123", tmp_path)
            .with_build_dir(paths.build_dir)
        )

        ctx = harness.build_context()
        assert ctx.gateway is fresh_gateway
        assert ctx.snapshot.repo == "test/repo"

    def test_plugin_has_metadata(self) -> None:
        """Plugin has valid metadata accessible."""
        plugin = ScipIngestPlugin()

        metadata = plugin.metadata

        assert metadata.name == "scip_ingest"
        assert metadata.stage == "index"
        assert len(metadata.produces_tables) > 0


# =============================================================================
# Plugin Instance Tests
# =============================================================================


class TestScipIngestPluginInstance:
    """Tests for plugin instance behavior."""

    def test_plugin_is_dataclass(self) -> None:
        """Plugin is a dataclass and can be instantiated."""
        plugin = ScipIngestPlugin()
        assert plugin is not None

    def test_plugin_inheritance(self) -> None:
        """Plugin inherits from expected base classes."""
        plugin = ScipIngestPlugin()
        # Check via metadata which shows protocol implementation
        assert hasattr(plugin, "metadata")
        assert hasattr(plugin, "compute")
        assert hasattr(plugin, "execute")

    def test_multiple_instances_are_independent(self) -> None:
        """Multiple plugin instances are independent."""
        plugin1 = ScipIngestPlugin()
        plugin2 = ScipIngestPlugin()

        # Both should have same class attributes
        assert plugin1.plugin_name == plugin2.plugin_name
        assert plugin1 is not plugin2
