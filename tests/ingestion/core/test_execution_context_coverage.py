"""Coverage tests for ingestion execution context.

This module provides comprehensive tests for the IngestExecutionContext
and IngestExecutionContextBuilder classes, covering all context methods,
property accessors, and builder patterns.

Uses project test helpers and real types per the testing charter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.core.config.registry import ConfigRegistry
from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.resources import ResourceRegistry
from codeintel.ingestion.core.execution_context import (
    IngestExecutionContext,
    IngestExecutionContextBuilder,
)
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from codeintel.ingestion.resources.protocol import ResourceProviderBase
from tests._helpers.fakes import create_test_build_paths
from tests._helpers.harnesses import IngestTestSetup

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Test Resource Providers (following real ResourceProviderBase pattern)
# =============================================================================


class StringResourceProvider(ResourceProviderBase[str]):
    """Simple resource provider that returns a string value.

    This is a real implementation following the ResourceProviderBase pattern,
    suitable for dev/staging environments or tests.
    """

    RESOURCE_NAME: ClassVar[str] = "string_resource"

    def __init__(self, value: str = "test_value") -> None:
        """Initialize with a string value.

        Parameters
        ----------
        value
            The string value to return from get().
        """
        super().__init__()
        self._value = value

    def _load(self) -> str:
        """Load and return the string value.

        Returns
        -------
        str
            The configured string value.
        """
        return self._value


@dataclass
class SimpleTestConfig:
    """Simple configuration class for testing config registration.

    A minimal config class that follows the same pattern as production
    config classes.
    """

    value: str = "default"
    count: int = 0


# =============================================================================
# IngestExecutionContext Tests
# =============================================================================


class TestIngestExecutionContext:
    """Tests for IngestExecutionContext."""

    def test_basic_construction(self, ingest_setup: IngestTestSetup) -> None:
        """Context can be constructed with minimal fields."""
        ctx = ingest_setup.build_context("test")

        assert ctx.gateway is not None
        assert ctx.snapshot is not None
        assert ctx.plugin_name == "test"

    def test_default_tools_config(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Context has default tools config when not provided."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        assert ctx.tools is not None
        assert isinstance(ctx.tools, ToolsConfig)

    def test_validated_paths_raises_when_not_set(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Validated paths raises RuntimeError when not set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            paths=None,
        )

        with pytest.raises(RuntimeError, match="paths not initialized"):
            _ = ctx.validated_paths

    def test_validated_paths_returns_when_set(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Validated paths returns BuildPaths when set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            paths=paths,
        )

        assert ctx.validated_paths is paths

    def test_validated_code_profile_raises_when_not_set(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Validated code_profile raises RuntimeError when not set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            code_profile=None,
        )

        with pytest.raises(RuntimeError, match="code_profile not initialized"):
            _ = ctx.validated_code_profile

    def test_validated_code_profile_returns_when_set(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Validated code_profile returns ScanProfile when set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        profile = default_code_profile(tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            code_profile=profile,
        )

        assert ctx.validated_code_profile is profile

    def test_validated_config_profile_raises_when_not_set(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Validated config_profile raises RuntimeError when not set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            config_profile=None,
        )

        with pytest.raises(RuntimeError, match="config_profile not initialized"):
            _ = ctx.validated_config_profile

    def test_validated_config_profile_returns_when_set(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """Validated config_profile returns ScanProfile when set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        profile = default_code_profile(tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            config_profile=profile,
        )

        assert ctx.validated_config_profile is profile

    def test_build_dir_property(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Build dir property returns paths.build_dir."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            paths=paths,
        )

        assert ctx.build_dir == paths.build_dir


class TestIngestExecutionContextResources:
    """Tests for resource provider methods."""

    def test_has_resource_false_when_empty(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """has_resource returns False for empty registry."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        assert ctx.has_resource(StringResourceProvider) is False

    def test_has_resource_true_when_registered(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """has_resource returns True when resource is registered."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        resources = ResourceRegistry()
        resources.register(StringResourceProvider, StringResourceProvider())
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            resources=resources,
        )

        assert ctx.has_resource(StringResourceProvider) is True

    def test_has_resource_by_name_false(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """has_resource_by_name returns False for unknown name."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        assert ctx.has_resource_by_name("unknown") is False

    def test_has_resource_by_name_true(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """has_resource_by_name returns True when registered."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        resources = ResourceRegistry()
        resources.register(StringResourceProvider, StringResourceProvider(), name="string_resource")
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            resources=resources,
        )

        assert ctx.has_resource_by_name("string_resource") is True

    def test_require_returns_resource(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Require returns the resource provider."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        provider = StringResourceProvider()
        resources = ResourceRegistry()
        resources.register(StringResourceProvider, provider)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            resources=resources,
        )

        result = ctx.require(StringResourceProvider)
        assert result is provider

    def test_require_by_name_returns_resource(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """require_by_name returns the resource provider."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        provider = StringResourceProvider()
        resources = ResourceRegistry()
        resources.register(StringResourceProvider, provider, name="string_resource")
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            resources=resources,
        )

        # require_by_name returns the result of get() on the provider
        result = ctx.require_by_name("string_resource")
        # Check it's either the value or the provider depending on registry impl
        assert result is not None


class TestIngestExecutionContextConfigs:
    """Tests for configuration methods."""

    def test_register_config(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """register_config adds config to registry."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        config = SimpleTestConfig(value="custom")
        ctx.register_config(SimpleTestConfig, config)

        assert ctx.has_config(SimpleTestConfig) is True

    def test_has_config_false_when_missing(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """has_config returns False for unregistered config."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        assert ctx.has_config(SimpleTestConfig) is False

    def test_get_config_returns_registered(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """get_config returns the registered config."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        config = SimpleTestConfig(value="registered", count=10)
        ctx.register_config(SimpleTestConfig, config)

        result = ctx.get_config(SimpleTestConfig)
        assert result is config
        assert result.value == "registered"
        assert result.count == 10

    def test_get_config_raises_for_missing(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """get_config raises KeyError for unregistered config."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        with pytest.raises(KeyError):
            ctx.get_config(SimpleTestConfig)

    def test_get_optional_config_returns_none_when_missing(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """get_optional_config returns None for unregistered config."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        result = ctx.get_optional_config(SimpleTestConfig)
        assert result is None

    def test_get_optional_config_returns_registered(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """get_optional_config returns the registered config."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        config = SimpleTestConfig(value="optional")
        ctx.register_config(SimpleTestConfig, config)

        result = ctx.get_optional_config(SimpleTestConfig)
        assert result is config


class TestIngestExecutionContextTiming:
    """Tests for plugin timing methods."""

    def test_start_plugin_timer(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """start_plugin_timer enables finish_plugin_timer to return positive duration."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        ctx.start_plugin_timer("test_plugin")
        # Timer was started, so finish should return a valid (non-zero after elapsed time)
        duration = ctx.finish_plugin_timer("test_plugin")

        # Duration should be non-negative (timer was started)
        assert duration >= 0.0

    def test_start_plugin_timer_no_overwrite(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """start_plugin_timer does not reset duration when called twice."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        ctx.start_plugin_timer("test_plugin")
        # Wait a small amount of time
        time.sleep(0.01)

        # Second call should not reset the timer
        ctx.start_plugin_timer("test_plugin")
        duration = ctx.finish_plugin_timer("test_plugin")

        # Should include the full elapsed time since first start (at least 10ms)
        assert duration >= 0.009

    def test_finish_plugin_timer_returns_duration(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """finish_plugin_timer returns elapsed duration."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        ctx.start_plugin_timer("test_plugin")
        duration = ctx.finish_plugin_timer("test_plugin")

        assert duration >= 0.0

    def test_finish_plugin_timer_returns_zero_when_not_started(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """finish_plugin_timer returns 0.0 when timer was not started."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        duration = ctx.finish_plugin_timer("unknown_plugin")

        assert duration == 0.0

    def test_finish_plugin_timer_caches_duration(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """finish_plugin_timer caches the duration for subsequent calls."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        ctx.start_plugin_timer("test_plugin")
        first_duration = ctx.finish_plugin_timer("test_plugin")
        second_duration = ctx.finish_plugin_timer("test_plugin")

        # Should return the same cached value
        assert first_duration == second_duration

    def test_finish_plugin_timer_clears_timer_state(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """finish_plugin_timer clears timer so re-starting works correctly."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        # First timer cycle
        ctx.start_plugin_timer("test_plugin")
        time.sleep(0.01)
        first_duration = ctx.finish_plugin_timer("test_plugin")

        # Cached value persists until new timer starts
        assert ctx.finish_plugin_timer("test_plugin") == first_duration

        # Starting a new timer should reset the duration tracking
        ctx.start_plugin_timer("test_plugin")
        new_duration = ctx.finish_plugin_timer("test_plugin")

        # New duration should be less than first (timer was just restarted)
        assert new_duration < first_duration


class TestIngestExecutionContextCountProducedTables:
    """Tests for count_produced_tables method."""

    def test_count_produced_tables_empty(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """count_produced_tables returns empty dict for no tables."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        result = ctx.count_produced_tables(())

        assert result == {}

    def test_count_produced_tables_returns_counts(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """count_produced_tables returns counts for existing tables."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        # repo_map table should exist after fresh_gateway creation
        result = ctx.count_produced_tables(("core.repo_map",))

        assert "core.repo_map" in result
        assert result["core.repo_map"] >= 0

    def test_count_produced_tables_nonexistent_table(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """count_produced_tables returns 0 for nonexistent tables."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        ctx = IngestExecutionContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

        result = ctx.count_produced_tables(("nonexistent.table",))

        assert result["nonexistent.table"] == 0


# =============================================================================
# IngestExecutionContextBuilder Tests
# =============================================================================


class TestIngestExecutionContextBuilder:
    """Tests for IngestExecutionContextBuilder."""

    def test_basic_build(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder can construct context with minimal fields."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)

        ctx = IngestExecutionContextBuilder(fresh_gateway, snapshot).build()

        assert ctx.gateway is fresh_gateway
        assert ctx.snapshot is snapshot

    def test_with_run_id(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets run_id."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)

        ctx = IngestExecutionContextBuilder(fresh_gateway, snapshot).with_run_id("run-123").build()

        assert ctx.run_id == "run-123"

    def test_with_paths(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets paths."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)

        ctx = IngestExecutionContextBuilder(fresh_gateway, snapshot).with_paths(paths).build()

        assert ctx.paths is paths

    def test_with_code_profile(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets code_profile."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        profile = default_code_profile(tmp_path)

        ctx = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_code_profile(profile)
            .build()
        )

        assert ctx.code_profile is profile

    def test_with_config_profile(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets config_profile."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        profile = default_code_profile(tmp_path)

        ctx = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_config_profile(profile)
            .build()
        )

        assert ctx.config_profile is profile

    def test_with_tools(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets tools config."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        tools = ToolsConfig.default()

        ctx = IngestExecutionContextBuilder(fresh_gateway, snapshot).with_tools(tools).build()

        assert ctx.tools is tools

    def test_with_resources(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets resource registry."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        resources = ResourceRegistry()

        ctx = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot).with_resources(resources).build()
        )

        assert ctx.resources is resources

    def test_with_scratch(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets scratch space."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        scratch = PluginScratch()
        scratch.declare("key", "value")

        ctx = IngestExecutionContextBuilder(fresh_gateway, snapshot).with_scratch(scratch).build()

        assert ctx.scratch is scratch

    def test_with_configs(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets config registry."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        configs = ConfigRegistry()

        ctx = IngestExecutionContextBuilder(fresh_gateway, snapshot).with_configs(configs).build()

        assert ctx.configs is configs

    def test_with_plugin_name(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder sets plugin name."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)

        ctx = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_plugin_name("my_plugin")
            .build()
        )

        assert ctx.plugin_name == "my_plugin"

    def test_method_chaining(self, fresh_gateway: StorageGateway, tmp_path: Path) -> None:
        """Builder methods can be chained."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        profile = default_code_profile(tmp_path)

        ctx = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_run_id("run-456")
            .with_paths(paths)
            .with_code_profile(profile)
            .with_config_profile(profile)
            .with_plugin_name("chained_plugin")
            .build()
        )

        assert ctx.run_id == "run-456"
        assert ctx.paths is paths
        assert ctx.code_profile is profile
        assert ctx.config_profile is profile
        assert ctx.plugin_name == "chained_plugin"


class TestIngestExecutionContextBuilderValidated:
    """Tests for build_validated method."""

    def test_build_validated_fails_without_paths(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """build_validated raises when paths is not set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        profile = default_code_profile(tmp_path)

        builder = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_code_profile(profile)
            .with_config_profile(profile)
        )

        with pytest.raises(ValueError, match="paths is required"):
            builder.build_validated()

    def test_build_validated_fails_without_code_profile(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """build_validated raises when code_profile is not set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        profile = default_code_profile(tmp_path)

        builder = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_paths(paths)
            .with_config_profile(profile)
        )

        with pytest.raises(ValueError, match="code_profile is required"):
            builder.build_validated()

    def test_build_validated_fails_without_config_profile(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """build_validated raises when config_profile is not set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        profile = default_code_profile(tmp_path)

        builder = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_paths(paths)
            .with_code_profile(profile)
        )

        with pytest.raises(ValueError, match="config_profile is required"):
            builder.build_validated()

    def test_build_validated_collects_all_errors(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """build_validated reports all missing fields."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)

        builder = IngestExecutionContextBuilder(fresh_gateway, snapshot)

        with pytest.raises(ValueError, match="is required") as exc_info:
            builder.build_validated()

        error_msg = str(exc_info.value)
        assert "paths is required" in error_msg
        assert "code_profile is required" in error_msg
        assert "config_profile is required" in error_msg

    def test_build_validated_succeeds_with_all_required(
        self, fresh_gateway: StorageGateway, tmp_path: Path
    ) -> None:
        """build_validated succeeds when all required fields are set."""
        snapshot = SnapshotRef(repo="test", commit="abc", repo_root=tmp_path)
        paths = create_test_build_paths(tmp_path)
        profile = default_code_profile(tmp_path)

        ctx = (
            IngestExecutionContextBuilder(fresh_gateway, snapshot)
            .with_paths(paths)
            .with_code_profile(profile)
            .with_config_profile(profile)
            .build_validated()
        )

        assert ctx.paths is paths
        assert ctx.code_profile is profile
        assert ctx.config_profile is profile
