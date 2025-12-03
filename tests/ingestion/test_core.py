"""Tests for core ingestion base classes and utilities.

This module tests ValidationResult, ResolvedConfig, and plugin base classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from codeintel.ingestion.core.base import (
    BaseIngestPlugin,
    ResolvedConfig,
    TrackerRequiringPlugin,
    ValidationResult,
)
from codeintel.ingestion.plugins.protocol import (
    IngestPluginMetadata,
    IngestPluginResult,
    IngestResourceHints,
    IngestStage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.ingestion.core.execution_context import IngestExecutionContext


# =============================================================================
# Test ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_success(self) -> None:
        """ValidationResult.success() should create valid result."""
        result = ValidationResult.success()

        assert result.valid is True
        assert result.errors == ()

    def test_failure(self) -> None:
        """ValidationResult.failure() should create failed result."""
        errors = ("Error 1", "Error 2")
        result = ValidationResult.failure(errors)

        assert result.valid is False
        assert result.errors == errors

    def test_failure_single_error(self) -> None:
        """ValidationResult.failure() should work with single error."""
        result = ValidationResult.failure(("Single error",))

        assert result.valid is False
        assert len(result.errors) == 1

    def test_direct_construction(self) -> None:
        """ValidationResult should be directly constructable."""
        result = ValidationResult(valid=True, errors=())

        assert result.valid is True


# =============================================================================
# Test ResolvedConfig
# =============================================================================


class TestResolvedConfig:
    """Tests for ResolvedConfig generic container."""

    def test_initial_state(self) -> None:
        """ResolvedConfig should start unresolved."""
        config: ResolvedConfig[str] = ResolvedConfig()

        assert config.resolved is False
        assert config.value is None

    def test_set_marks_resolved(self) -> None:
        """ResolvedConfig.set() should mark as resolved."""
        config: ResolvedConfig[str] = ResolvedConfig()

        config.set("test_value")

        assert config.resolved is True
        assert config.value == "test_value"

    def test_get_returns_value(self) -> None:
        """ResolvedConfig.get() should return set value."""
        config: ResolvedConfig[str] = ResolvedConfig()
        config.set("test_value")

        result = config.get()

        assert result == "test_value"

    def test_get_raises_when_not_resolved(self) -> None:
        """ResolvedConfig.get() should raise when not resolved."""
        config: ResolvedConfig[str] = ResolvedConfig()

        with pytest.raises(ValueError, match="not resolved"):
            config.get("test_plugin")

    def test_get_includes_plugin_name_in_error(self) -> None:
        """ResolvedConfig.get() error should include plugin name."""
        config: ResolvedConfig[str] = ResolvedConfig()

        with pytest.raises(ValueError, match="my_plugin"):
            config.get("my_plugin")

    def test_get_or_none_when_resolved(self) -> None:
        """ResolvedConfig.get_or_none() should return value when resolved."""
        config: ResolvedConfig[str] = ResolvedConfig()
        config.set("test_value")

        result = config.get_or_none()

        assert result == "test_value"

    def test_get_or_none_when_not_resolved(self) -> None:
        """ResolvedConfig.get_or_none() should return None when not resolved."""
        config: ResolvedConfig[str] = ResolvedConfig()

        result = config.get_or_none()

        assert result is None

    def test_with_dataclass_type(self) -> None:
        """ResolvedConfig should work with dataclass types."""

        @dataclass
        class TestConfig:
            """Test configuration."""

            value: int
            name: str

        config: ResolvedConfig[TestConfig] = ResolvedConfig()
        test_cfg = TestConfig(value=42, name="test")

        config.set(test_cfg)
        result = config.get()

        assert result.value == 42
        assert result.name == "test"


# =============================================================================
# Mock Plugin for Testing
# =============================================================================


@dataclass
class TestPlugin(BaseIngestPlugin):
    """Test plugin for testing base class behavior."""

    plugin_name: ClassVar[str] = "test_plugin"
    plugin_description: ClassVar[str] = "Test plugin for testing"
    plugin_stage: ClassVar[IngestStage] = "extract"
    plugin_version: ClassVar[str] = "1.0.0"
    output_tables: ClassVar[tuple[str, ...]] = ("core.test",)

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Execute the test plugin."""
        return {"core.test": 10}


@dataclass
class TrackerPlugin(TrackerRequiringPlugin):
    """Plugin that requires tracker for testing."""

    plugin_name: ClassVar[str] = "tracker_plugin"
    plugin_description: ClassVar[str] = "Plugin requiring tracker"
    plugin_stage: ClassVar[IngestStage] = "extract"
    plugin_version: ClassVar[str] = "1.0.0"
    output_tables: ClassVar[tuple[str, ...]] = ("core.test",)
    tracker_required: ClassVar[bool] = True

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Execute the plugin."""
        return {"core.test": 5}


# =============================================================================
# Test BaseIngestPlugin
# =============================================================================


class TestBaseIngestPlugin:
    """Tests for BaseIngestPlugin abstract base class."""

    def test_metadata_property(self) -> None:
        """BaseIngestPlugin.metadata should return correct metadata."""
        plugin = TestPlugin()

        meta = plugin.metadata

        assert meta.name == "test_plugin"
        assert meta.description == "Test plugin for testing"
        assert meta.stage == "extract"
        assert meta.version_hash == "1.0.0"
        assert "core.test" in meta.produces_tables

    def test_plugin_name(self) -> None:
        """Plugin should have correct name from class variable."""
        plugin = TestPlugin()

        assert plugin.plugin_name == "test_plugin"

    def test_default_depends_on(self) -> None:
        """BaseIngestPlugin should have empty depends_on by default."""
        plugin = TestPlugin()

        assert plugin.metadata.depends_on == ()

    def test_default_provides(self) -> None:
        """BaseIngestPlugin should have empty provides by default."""
        plugin = TestPlugin()

        assert plugin.metadata.provides == ()

    def test_default_requires(self) -> None:
        """BaseIngestPlugin should have empty requires by default."""
        plugin = TestPlugin()

        assert plugin.metadata.requires == ()

    def test_default_supports_incremental(self) -> None:
        """BaseIngestPlugin should not support incremental by default."""
        plugin = TestPlugin()

        assert plugin.metadata.supports_incremental is False

    def test_resource_hints(self) -> None:
        """BaseIngestPlugin should have resource hints (may be None by default)."""
        plugin = TestPlugin()

        hints = plugin.metadata.resource_hints
        # resource_hints may be None if not set in plugin class vars
        assert hints is None or isinstance(hints, IngestResourceHints)


class TestTrackerRequiringPlugin:
    """Tests for TrackerRequiringPlugin mixin."""

    def test_tracker_required_class_var(self) -> None:
        """TrackerRequiringPlugin should have tracker_required set."""
        plugin = TrackerPlugin()

        assert plugin.tracker_required is True


# =============================================================================
# Test IngestPluginResult
# =============================================================================


class TestIngestPluginResult:
    """Tests for IngestPluginResult dataclass."""

    def test_ok(self) -> None:
        """IngestPluginResult.ok() should create success result."""
        result = IngestPluginResult.ok(row_counts={"table1": 100})

        assert result.success is True
        assert result.row_counts == {"table1": 100}
        assert result.skipped is False

    def test_fail(self) -> None:
        """IngestPluginResult.fail() should create failure result."""
        result = IngestPluginResult.fail("Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_fail_with_kind(self) -> None:
        """IngestPluginResult.fail() should accept error_kind."""
        result = IngestPluginResult.fail("Error", error_kind="parse_error")

        assert result.success is False
        assert result.error_kind == "parse_error"

    def test_skip(self) -> None:
        """IngestPluginResult.skip() should create skipped result."""
        result = IngestPluginResult.skip("No work needed")

        assert result.success is True
        assert result.skipped is True
        assert result.skip_reason == "No work needed"

    def test_ok_with_empty_counts(self) -> None:
        """IngestPluginResult.ok() should work with empty counts."""
        result = IngestPluginResult.ok(row_counts={})

        assert result.success is True
        assert result.row_counts == {}


# =============================================================================
# Test IngestPluginMetadata
# =============================================================================


class TestIngestPluginMetadata:
    """Tests for IngestPluginMetadata dataclass."""

    def test_create_minimal(self) -> None:
        """IngestPluginMetadata should be creatable with required fields."""
        meta = IngestPluginMetadata(
            name="test",
            description="Test plugin",
            stage="extract",
            provides=(),
            requires=(),
            depends_on=(),
            produces_tables=("core.test",),
            tool_dependencies=(),
            supports_incremental=False,
            resource_hints=IngestResourceHints(),
            version_hash="1.0.0",
        )

        assert meta.name == "test"
        assert meta.stage == "extract"

    def test_frozen(self) -> None:
        """IngestPluginMetadata should be immutable."""
        meta = IngestPluginMetadata(
            name="test",
            description="Test plugin",
            stage="extract",
            provides=(),
            requires=(),
            depends_on=(),
            produces_tables=(),
            tool_dependencies=(),
            supports_incremental=False,
            resource_hints=IngestResourceHints(),
            version_hash="1.0.0",
        )

        with pytest.raises(AttributeError):
            meta.name = "new_name"  # type: ignore[misc]


# =============================================================================
# Test IngestResourceHints
# =============================================================================


class TestIngestResourceHints:
    """Tests for IngestResourceHints dataclass."""

    def test_defaults(self) -> None:
        """IngestResourceHints should have sensible defaults."""
        hints = IngestResourceHints()

        assert hints.cpu_intensive is False
        assert hints.io_intensive is False
        assert hints.memory_mb_hint is None
        assert hints.max_runtime_ms is None

    def test_custom_values(self) -> None:
        """IngestResourceHints should accept custom values."""
        hints = IngestResourceHints(
            cpu_intensive=True,
            io_intensive=True,
            memory_mb_hint=1024,
            max_runtime_ms=60000,
        )

        assert hints.cpu_intensive is True
        assert hints.io_intensive is True
        assert hints.memory_mb_hint == 1024
        assert hints.max_runtime_ms == 60000

