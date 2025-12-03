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
from tests._helpers.frozen_test import try_setattr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.ingestion.core.execution_context import IngestExecutionContext


# Test constants for magic values
TEST_VALUE = 42
TEST_MEMORY_MB = 1024
TEST_TIMEOUT_MS = 60000


# =============================================================================
# Mock Plugin for Testing
# =============================================================================


@dataclass
class TestPlugin(BaseIngestPlugin):
    """Test plugin for testing base class behavior."""

    plugin_name: ClassVar[str] = "test_plugin"
    plugin_description: ClassVar[str] = "Test plugin for testing"
    plugin_stage: ClassVar[IngestStage] = "parse"
    plugin_version: ClassVar[str] = "1.0.0"
    output_tables: ClassVar[tuple[str, ...]] = ("core.test",)

    @staticmethod
    def compute(ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Execute the test plugin.

        Parameters
        ----------
        ctx
            Execution context (unused in this mock).

        Returns
        -------
        Mapping[str, int] | None
            Row counts for the produced tables.
        """
        _ = ctx  # Mock doesn't use context
        return {"core.test": 10}


@dataclass
class TrackerPlugin(TrackerRequiringPlugin):
    """Plugin that requires tracker for testing."""

    plugin_name: ClassVar[str] = "tracker_plugin"
    plugin_description: ClassVar[str] = "Plugin requiring tracker"
    plugin_stage: ClassVar[IngestStage] = "parse"
    plugin_version: ClassVar[str] = "1.0.0"
    output_tables: ClassVar[tuple[str, ...]] = ("core.test",)
    tracker_required: ClassVar[bool] = True

    @staticmethod
    def compute(ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context (unused in this mock).

        Returns
        -------
        Mapping[str, int] | None
            Row counts for the produced tables.
        """
        _ = ctx  # Mock doesn't use context
        return {"core.test": 5}


# =============================================================================
# ValidationResult Tests
# =============================================================================


def test_validation_result_success() -> None:
    """ValidationResult.success() should create valid result."""
    result = ValidationResult.success()

    assert result.valid is True
    assert result.errors == ()


def test_validation_result_failure() -> None:
    """ValidationResult.failure() should create failed result."""
    errors = ("Error 1", "Error 2")
    result = ValidationResult.failure(errors)

    assert result.valid is False
    assert result.errors == errors


def test_validation_result_failure_single_error() -> None:
    """ValidationResult.failure() should work with single error."""
    result = ValidationResult.failure(("Single error",))

    assert result.valid is False
    assert len(result.errors) == 1


def test_validation_result_direct_construction() -> None:
    """ValidationResult should be directly constructable."""
    result = ValidationResult(valid=True, errors=())

    assert result.valid is True


# =============================================================================
# ResolvedConfig Tests
# =============================================================================


def test_resolved_config_initial_state() -> None:
    """ResolvedConfig should start unresolved."""
    config: ResolvedConfig[str] = ResolvedConfig()

    assert config.resolved is False
    assert config.value is None


def test_resolved_config_set_marks_resolved() -> None:
    """ResolvedConfig.set() should mark as resolved."""
    config: ResolvedConfig[str] = ResolvedConfig()

    config.set("test_value")

    assert config.resolved is True
    assert config.value == "test_value"


def test_resolved_config_get_returns_value() -> None:
    """ResolvedConfig.get() should return set value."""
    config: ResolvedConfig[str] = ResolvedConfig()
    config.set("test_value")

    result = config.get()

    assert result == "test_value"


def test_resolved_config_get_raises_when_not_resolved() -> None:
    """ResolvedConfig.get() should raise when not resolved."""
    config: ResolvedConfig[str] = ResolvedConfig()

    with pytest.raises(ValueError, match="not resolved"):
        config.get("test_plugin")


def test_resolved_config_get_includes_plugin_name_in_error() -> None:
    """ResolvedConfig.get() error should include plugin name."""
    config: ResolvedConfig[str] = ResolvedConfig()

    with pytest.raises(ValueError, match="my_plugin"):
        config.get("my_plugin")


def test_resolved_config_get_or_none_when_resolved() -> None:
    """ResolvedConfig.get_or_none() should return value when resolved."""
    config: ResolvedConfig[str] = ResolvedConfig()
    config.set("test_value")

    result = config.get_or_none()

    assert result == "test_value"


def test_resolved_config_get_or_none_when_not_resolved() -> None:
    """ResolvedConfig.get_or_none() should return None when not resolved."""
    config: ResolvedConfig[str] = ResolvedConfig()

    result = config.get_or_none()

    assert result is None


def test_resolved_config_with_dataclass_type() -> None:
    """ResolvedConfig should work with dataclass types."""

    @dataclass
    class TestConfig:
        """Test configuration."""

        value: int
        name: str

    config: ResolvedConfig[TestConfig] = ResolvedConfig()
    test_cfg = TestConfig(value=TEST_VALUE, name="test")

    config.set(test_cfg)
    result = config.get()

    assert result.value == TEST_VALUE
    assert result.name == "test"


# =============================================================================
# BaseIngestPlugin Tests
# =============================================================================


def test_base_ingest_plugin_metadata_property() -> None:
    """BaseIngestPlugin.metadata should return correct metadata."""
    plugin = TestPlugin()

    meta = plugin.metadata

    assert meta.name == "test_plugin"
    assert meta.description == "Test plugin for testing"
    assert meta.stage == "parse"
    assert meta.version_hash == "1.0.0"
    assert "core.test" in meta.produces_tables


def test_base_ingest_plugin_name() -> None:
    """Plugin should have correct name from class variable."""
    plugin = TestPlugin()

    assert plugin.plugin_name == "test_plugin"


def test_base_ingest_plugin_default_depends_on() -> None:
    """BaseIngestPlugin should have empty depends_on by default."""
    plugin = TestPlugin()

    assert plugin.metadata.depends_on == ()


def test_base_ingest_plugin_default_provides() -> None:
    """BaseIngestPlugin should have empty provides by default."""
    plugin = TestPlugin()

    assert plugin.metadata.provides == ()


def test_base_ingest_plugin_default_requires() -> None:
    """BaseIngestPlugin should have empty requires by default."""
    plugin = TestPlugin()

    assert plugin.metadata.requires == ()


def test_base_ingest_plugin_default_supports_incremental() -> None:
    """BaseIngestPlugin should not support incremental by default."""
    plugin = TestPlugin()

    assert plugin.metadata.supports_incremental is False


def test_base_ingest_plugin_resource_hints() -> None:
    """BaseIngestPlugin should have resource hints (may be None by default)."""
    plugin = TestPlugin()

    hints = plugin.metadata.resource_hints
    # resource_hints may be None if not set in plugin class vars
    assert hints is None or isinstance(hints, IngestResourceHints)


# =============================================================================
# TrackerRequiringPlugin Tests
# =============================================================================


def test_tracker_requiring_plugin_tracker_required() -> None:
    """TrackerRequiringPlugin should have tracker_required set."""
    plugin = TrackerPlugin()

    assert plugin.tracker_required is True


# =============================================================================
# IngestPluginResult Tests
# =============================================================================


def test_ingest_plugin_result_ok() -> None:
    """IngestPluginResult.ok() should create success result."""
    result = IngestPluginResult.ok(row_counts={"table1": 100})

    assert result.success is True
    assert result.row_counts == {"table1": 100}
    assert result.skipped is False


def test_ingest_plugin_result_fail() -> None:
    """IngestPluginResult.fail() should create failure result."""
    result = IngestPluginResult.fail("Something went wrong")

    assert result.success is False
    assert result.error == "Something went wrong"


def test_ingest_plugin_result_fail_with_kind() -> None:
    """IngestPluginResult.fail() should accept error_kind."""
    result = IngestPluginResult.fail("Error", error_kind="parse_error")

    assert result.success is False
    assert result.error_kind == "parse_error"


def test_ingest_plugin_result_skip() -> None:
    """IngestPluginResult.skip() should create skipped result."""
    result = IngestPluginResult.skip("No work needed")

    assert result.success is True
    assert result.skipped is True
    assert result.skip_reason == "No work needed"


def test_ingest_plugin_result_ok_with_empty_counts() -> None:
    """IngestPluginResult.ok() should work with empty counts."""
    result = IngestPluginResult.ok(row_counts={})

    assert result.success is True
    assert result.row_counts == {}


# =============================================================================
# IngestPluginMetadata Tests
# =============================================================================


def test_ingest_plugin_metadata_create_minimal() -> None:
    """IngestPluginMetadata should be creatable with required fields."""
    meta = IngestPluginMetadata(
        name="test",
        description="Test plugin",
        stage="parse",
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
    assert meta.stage == "parse"


def test_ingest_plugin_metadata_frozen() -> None:
    """IngestPluginMetadata should be immutable."""
    meta = IngestPluginMetadata(
        name="test",
        description="Test plugin",
        stage="parse",
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
        try_setattr(meta, "name", "new_name")


# =============================================================================
# IngestResourceHints Tests
# =============================================================================


def test_ingest_resource_hints_defaults() -> None:
    """IngestResourceHints should have sensible defaults."""
    hints = IngestResourceHints()

    assert hints.cpu_intensive is False
    assert hints.io_intensive is False
    assert hints.memory_mb_hint is None
    assert hints.max_runtime_ms is None


def test_ingest_resource_hints_custom_values() -> None:
    """IngestResourceHints should accept custom values."""
    hints = IngestResourceHints(
        cpu_intensive=True,
        io_intensive=True,
        memory_mb_hint=TEST_MEMORY_MB,
        max_runtime_ms=TEST_TIMEOUT_MS,
    )

    assert hints.cpu_intensive is True
    assert hints.io_intensive is True
    assert hints.memory_mb_hint == TEST_MEMORY_MB
    assert hints.max_runtime_ms == TEST_TIMEOUT_MS
