"""Tests for the config factory module.

This module tests ConfigFactory, BuildOptions, ConfigMapping, and related
utilities for building step configurations from plugin context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from codeintel.ingestion.core import IngestExecutionContext
from codeintel.ingestion.plugins.config_factory import (
    DEFAULT_CONTEXT_MAPPINGS,
    BuildOptions,
    ConfigFactory,
    ConfigMapping,
    get_config_fields,
    infer_config_mapping,
)
from tests._helpers.assertions import assert_cannot_setattr
from tests._helpers.fakes import FakeBuildPaths, FakeSnapshotRef

# Test constants
EXPECTED_FIELD_COUNT = 3


# Test config classes for factory tests
@dataclass(frozen=True)
class SimpleDataclassConfig:
    """Simple dataclass config for testing."""

    snapshot: FakeSnapshotRef
    paths: FakeBuildPaths
    extra_value: str = "default"


@dataclass(frozen=True)
class ConfigWithTools:
    """Config with tool-related fields."""

    snapshot: FakeSnapshotRef
    paths: FakeBuildPaths
    tools: dict[str, str] | None = None
    tracker: dict[str, str] | None = None
    tool_service: dict[str, str] | None = None


class NonDataclassConfig:
    """Non-dataclass config for testing signature introspection."""

    def __init__(
        self,
        snapshot: FakeSnapshotRef,
        paths: FakeBuildPaths,
        *,
        optional_value: str = "default",
    ) -> None:
        """Initialize non-dataclass config."""
        self.snapshot = snapshot
        self.paths = paths
        self.optional_value = optional_value


@dataclass
class MockContext:
    """Mock execution context for testing with typed fields."""

    snapshot: FakeSnapshotRef = field(default_factory=FakeSnapshotRef)
    paths: FakeBuildPaths = field(default_factory=FakeBuildPaths)
    tools: Any | None = None
    code_profile: Any | None = None
    config_profile: Any | None = None
    tool_runner: Any | None = None


# =============================================================================
# ConfigMapping Tests
# =============================================================================


def test_config_mapping_create_minimal() -> None:
    """Test creating ConfigMapping with minimal args."""
    mapping = ConfigMapping(config_class=SimpleDataclassConfig)

    assert mapping.config_class is SimpleDataclassConfig
    assert mapping.field_map == {}
    assert mapping.extra_fields == {}
    assert mapping.auto_infer is True


def test_config_mapping_create_with_field_map() -> None:
    """Test creating ConfigMapping with custom field map."""
    field_map = {"snapshot": "custom_snapshot", "paths": "custom_paths"}
    mapping = ConfigMapping(
        config_class=SimpleDataclassConfig,
        field_map=field_map,
    )

    assert mapping.field_map == field_map


def test_config_mapping_create_with_extra_fields() -> None:
    """Test creating ConfigMapping with extra fields."""
    extra_fields = {"extra_value": "overridden"}
    mapping = ConfigMapping(
        config_class=SimpleDataclassConfig,
        extra_fields=extra_fields,
    )

    assert mapping.extra_fields == extra_fields


def test_config_mapping_frozen_dataclass() -> None:
    """Test that ConfigMapping is immutable."""
    mapping = ConfigMapping(config_class=SimpleDataclassConfig)

    assert_cannot_setattr(mapping, "auto_infer", value=False)


# =============================================================================
# BuildOptions Tests
# =============================================================================


def test_build_options_create_empty() -> None:
    """Test creating BuildOptions with defaults."""
    options = BuildOptions()

    assert options.mapping is None
    assert options.extra is None
    assert options.tracker is None
    assert options.tool_service is None


def test_build_options_create_with_mapping() -> None:
    """Test creating BuildOptions with custom mapping."""
    mapping = {"field1": "attr1"}
    options = BuildOptions(mapping=mapping)

    assert options.mapping == mapping


def test_build_options_create_with_extra() -> None:
    """Test creating BuildOptions with extra values."""
    extra = {"extra_value": "custom"}
    options = BuildOptions(extra=extra)

    assert options.extra == extra


def test_build_options_create_with_tracker() -> None:
    """Test creating BuildOptions with tracker."""
    # Use a dict as a mock tracker value for testing
    tracker: Any = {"mock": "tracker"}
    options = BuildOptions(tracker=tracker)

    assert options.tracker is tracker


def test_build_options_create_with_tool_service() -> None:
    """Test creating BuildOptions with tool_service."""
    # Use a dict as a mock service value for testing
    service: Any = {"mock": "service"}
    options = BuildOptions(tool_service=service)

    assert options.tool_service is service


def test_build_options_frozen_dataclass() -> None:
    """Test that BuildOptions is immutable."""
    options = BuildOptions()

    assert_cannot_setattr(options, "mapping", {})


# =============================================================================
# get_config_fields Tests
# =============================================================================


def test_get_config_fields_dataclass() -> None:
    """Test getting fields from a dataclass."""
    fields = get_config_fields(SimpleDataclassConfig)

    assert "snapshot" in fields
    assert "paths" in fields
    assert "extra_value" in fields
    assert len(fields) == EXPECTED_FIELD_COUNT


def test_get_config_fields_non_dataclass() -> None:
    """Test getting fields from a non-dataclass via signature."""
    fields = get_config_fields(NonDataclassConfig)

    assert "snapshot" in fields
    assert "paths" in fields
    assert "optional_value" in fields
    assert "self" not in fields


def test_get_config_fields_config_with_many_fields() -> None:
    """Test getting fields from config with many fields."""
    fields = get_config_fields(ConfigWithTools)

    assert "snapshot" in fields
    assert "paths" in fields
    assert "tools" in fields
    assert "tracker" in fields
    assert "tool_service" in fields


# =============================================================================
# infer_config_mapping Tests
# =============================================================================


def test_infer_config_mapping_basic() -> None:
    """Test inferring mapping for basic config."""
    mapping = infer_config_mapping(SimpleDataclassConfig)

    assert mapping.config_class is SimpleDataclassConfig
    assert mapping.auto_infer is True
    # snapshot and paths should be in field_map as they match defaults
    assert "snapshot" in mapping.field_map
    assert "paths" in mapping.field_map
    # extra_value should not be in field_map (not in defaults)
    assert "extra_value" not in mapping.field_map


def test_infer_config_mapping_with_tools() -> None:
    """Test inferring mapping for config with tool fields."""
    mapping = infer_config_mapping(ConfigWithTools)

    assert "snapshot" in mapping.field_map
    assert "paths" in mapping.field_map
    assert "tools" in mapping.field_map
    # tracker and tool_service should not be in default mappings
    assert "tracker" not in mapping.field_map
    assert "tool_service" not in mapping.field_map


def test_infer_config_mapping_non_dataclass() -> None:
    """Test inferring mapping for non-dataclass config."""
    mapping = infer_config_mapping(NonDataclassConfig)

    assert mapping.config_class is NonDataclassConfig
    assert "snapshot" in mapping.field_map
    assert "paths" in mapping.field_map


# =============================================================================
# ConfigFactory Tests
# =============================================================================


def test_config_factory_create_with_default_mappings() -> None:
    """Test creating factory with default mappings."""
    factory = ConfigFactory()

    # The factory should have the default mappings
    assert factory.default_mappings == dict(DEFAULT_CONTEXT_MAPPINGS)


def test_config_factory_create_with_custom_mappings() -> None:
    """Test creating factory with custom default mappings."""
    custom_mappings = {"custom_field": "custom_attr"}
    factory = ConfigFactory(default_mappings=custom_mappings)

    assert factory.default_mappings == custom_mappings


def test_config_factory_build_simple_config() -> None:
    """Test building a simple dataclass config."""
    factory = ConfigFactory()
    ctx = MockContext()

    config = factory.build(SimpleDataclassConfig, cast("IngestExecutionContext", ctx))

    assert isinstance(config, SimpleDataclassConfig)
    assert config.snapshot == ctx.snapshot
    assert config.paths == ctx.paths
    assert config.extra_value == "default"


def test_config_factory_build_with_extra_values() -> None:
    """Test building config with extra values."""
    factory = ConfigFactory()
    ctx = MockContext()
    options = BuildOptions(extra={"extra_value": "custom_extra"})

    config = factory.build(SimpleDataclassConfig, cast("IngestExecutionContext", ctx), options)

    assert isinstance(config, SimpleDataclassConfig)
    assert config.extra_value == "custom_extra"


def test_config_factory_build_with_custom_mapping() -> None:
    """Test building config with custom field mapping."""
    factory = ConfigFactory()

    @dataclass(frozen=True)
    class CustomFieldConfig:
        """Config with custom field names."""

        custom_snapshot: Any
        custom_paths: Any

    # Create a context with the expected attributes
    @dataclass
    class CustomContext:
        """Context with custom attribute names."""

        my_snapshot: FakeSnapshotRef = field(default_factory=FakeSnapshotRef)
        my_paths: FakeBuildPaths = field(default_factory=FakeBuildPaths)

    ctx = CustomContext()
    options = BuildOptions(
        mapping={
            "custom_snapshot": "my_snapshot",
            "custom_paths": "my_paths",
        }
    )

    config = factory.build(CustomFieldConfig, cast("IngestExecutionContext", ctx), options)
    assert isinstance(config, CustomFieldConfig)

    assert config.custom_snapshot == ctx.my_snapshot
    assert config.custom_paths == ctx.my_paths


def test_config_factory_build_with_tracker() -> None:
    """Test building config with tracker in options via custom mapping."""
    factory = ConfigFactory()
    ctx = MockContext()
    mock_tracker: Any = {"mock": "tracker"}
    # tracker requires custom mapping since it's not in DEFAULT_CONTEXT_MAPPINGS
    options = BuildOptions(
        tracker=mock_tracker,
        mapping={"tracker": "tracker"},
    )

    config = factory.build(ConfigWithTools, cast("IngestExecutionContext", ctx), options)
    assert isinstance(config, ConfigWithTools)

    # tracker should be set from options
    assert config.tracker is mock_tracker


def test_config_factory_build_with_tool_service() -> None:
    """Test building config with tool_service in options via custom mapping."""
    factory = ConfigFactory()
    ctx = MockContext()
    mock_service: Any = {"mock": "service"}
    # tool_service requires custom mapping since it's not in DEFAULT_CONTEXT_MAPPINGS
    options = BuildOptions(
        tool_service=mock_service,
        mapping={"tool_service": "tool_service"},
    )

    config = factory.build(ConfigWithTools, cast("IngestExecutionContext", ctx), options)
    assert isinstance(config, ConfigWithTools)

    # tool_service should be set from options
    assert config.tool_service is mock_service


def test_config_factory_build_with_tools_from_context() -> None:
    """Test building config with tools from context."""
    factory = ConfigFactory()
    mock_tools = {"pyright": "/usr/bin/pyright"}
    ctx = MockContext(tools=mock_tools)

    config = factory.build(ConfigWithTools, cast("IngestExecutionContext", ctx))
    assert isinstance(config, ConfigWithTools)

    assert config.tools is mock_tools


def test_config_factory_build_non_dataclass_config() -> None:
    """Test building a non-dataclass config."""
    factory = ConfigFactory()
    ctx = MockContext()

    config = factory.build(NonDataclassConfig, cast("IngestExecutionContext", ctx))

    assert isinstance(config, NonDataclassConfig)
    assert config.snapshot == ctx.snapshot
    assert config.paths == ctx.paths
    assert config.optional_value == "default"


def test_config_factory_build_ignores_unknown_extra_fields() -> None:
    """Test that extra fields not in config class are ignored."""
    factory = ConfigFactory()
    ctx = MockContext()
    options = BuildOptions(
        extra={
            "extra_value": "custom",
            "nonexistent_field": "should_be_ignored",
        }
    )

    config = factory.build(SimpleDataclassConfig, cast("IngestExecutionContext", ctx), options)
    assert isinstance(config, SimpleDataclassConfig)

    assert config.extra_value == "custom"
    # nonexistent_field should be silently ignored


def test_config_factory_custom_mapping_overrides_default() -> None:
    """Test that custom mapping takes precedence over defaults."""
    factory = ConfigFactory()

    @dataclass
    class ContextWithBoth:
        """Context with both snapshot and alt_snapshot."""

        snapshot: FakeSnapshotRef = field(default_factory=FakeSnapshotRef)
        alt_snapshot: FakeSnapshotRef = field(
            default_factory=lambda: FakeSnapshotRef(repo="alt/repo", commit="altcommit")
        )
        paths: FakeBuildPaths = field(default_factory=FakeBuildPaths)

    ctx = ContextWithBoth()
    options = BuildOptions(mapping={"snapshot": "alt_snapshot"})

    config = factory.build(SimpleDataclassConfig, cast("IngestExecutionContext", ctx), options)
    assert isinstance(config, SimpleDataclassConfig)

    # Should use alt_snapshot due to custom mapping
    assert config.snapshot == ctx.alt_snapshot


# =============================================================================
# Context Value Tests (via ConfigFactory)
# =============================================================================


def test_get_context_value_existing_attribute() -> None:
    """Test getting an existing attribute from context."""
    factory = ConfigFactory()
    custom_snapshot = FakeSnapshotRef(repo="test-repo", commit="abc")
    ctx = MockContext(snapshot=custom_snapshot)

    config = factory.build(SimpleDataclassConfig, cast("IngestExecutionContext", ctx))
    assert isinstance(config, SimpleDataclassConfig)

    assert config.snapshot == custom_snapshot


def test_get_context_value_none_attribute() -> None:
    """Test getting a None attribute from context."""
    factory = ConfigFactory()
    ctx = MockContext(tools=None)

    config = factory.build(ConfigWithTools, cast("IngestExecutionContext", ctx))
    assert isinstance(config, ConfigWithTools)

    # tools should not be set when context value is None
    assert config.tools is None


def test_tracker_from_options_takes_precedence() -> None:
    """Test that tracker from options takes precedence over context."""
    factory = ConfigFactory()

    @dataclass
    class ContextWithTracker:
        """Context with tracker attribute."""

        snapshot: FakeSnapshotRef = field(default_factory=FakeSnapshotRef)
        paths: FakeBuildPaths = field(default_factory=FakeBuildPaths)
        tracker: Any = field(default_factory=lambda: {"from": "context"})

    ctx = ContextWithTracker()
    options_tracker: Any = {"from": "options"}
    # tracker requires explicit mapping to be resolved
    options = BuildOptions(
        tracker=options_tracker,
        mapping={"tracker": "tracker"},
    )

    config = factory.build(ConfigWithTools, cast("IngestExecutionContext", ctx), options)
    assert isinstance(config, ConfigWithTools)

    assert config.tracker == options_tracker


# =============================================================================
# DEFAULT_CONTEXT_MAPPINGS Tests
# =============================================================================


def test_default_context_mappings_contains_expected_keys() -> None:
    """Test that default mappings contain expected keys."""
    expected_keys = {
        "snapshot",
        "paths",
        "tool_runner",
        "code_profile",
        "config_profile",
        "tools",
    }

    assert set(DEFAULT_CONTEXT_MAPPINGS.keys()) == expected_keys


def test_default_context_mappings_are_identity() -> None:
    """Test that default mappings map to same-named attributes."""
    # In the default case, field names map to identical context attr names
    for key, value in DEFAULT_CONTEXT_MAPPINGS.items():
        assert key == value
