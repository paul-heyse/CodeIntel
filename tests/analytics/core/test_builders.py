"""Tests for fluent builders for plugin metadata.

This module tests:
- PluginSpecBuilder fluent API
- PluginSpec factory
- PluginMetaSection, PluginContractsSection, PluginRuntimeSection
"""

from __future__ import annotations

from codeintel.analytics.core.builders import (
    PluginContractsSection,
    PluginMetaSection,
    PluginRuntimeSection,
    PluginSpec,
    PluginSpecBuilder,
)
from codeintel.analytics.core.plugin_protocol import (
    PluginCapability,
    PluginInputSpec,
    PluginOutputSpec,
)

# Test constants
DEFAULT_VERSION = "1.0.0"
CUSTOM_VERSION = "2.5.0"
MAX_RUNTIME_MS = 5000
MAX_MEMORY_MB = 512
PRIORITY_HIGH = 10
EXPECTED_TWO_ITEMS = 2
EXPECTED_THREE_OUTPUTS = 3


# =============================================================================
# Section Dataclass Tests
# =============================================================================


def test_plugin_meta_section_defaults() -> None:
    """PluginMetaSection has sensible defaults."""
    section = PluginMetaSection(name="test.plugin")

    assert section.name == "test.plugin"
    assert not section.description  # Empty string
    assert section.stage == "other"
    assert section.version == DEFAULT_VERSION
    assert section.enabled_by_default is True
    assert section.severity == "fatal"
    assert section.tags == []


def test_plugin_meta_section_custom_values() -> None:
    """PluginMetaSection accepts custom values."""
    section = PluginMetaSection(
        name="custom.plugin",
        description="A custom plugin",
        stage="function",
        version=CUSTOM_VERSION,
        enabled_by_default=False,
        severity="soft_fail",
        tags=["test", "custom"],
    )

    assert section.name == "custom.plugin"
    assert section.description == "A custom plugin"
    assert section.stage == "function"
    assert section.version == CUSTOM_VERSION
    assert section.enabled_by_default is False
    assert section.severity == "soft_fail"
    assert section.tags == ["test", "custom"]


def test_plugin_contracts_section_defaults() -> None:
    """PluginContractsSection has empty defaults."""
    section = PluginContractsSection()

    assert section.inputs == []
    assert section.outputs == []
    assert section.capabilities_provided == []
    assert section.capabilities_required == []
    assert section.depends_on == []


def test_plugin_runtime_section_defaults() -> None:
    """PluginRuntimeSection has None/False defaults."""
    section = PluginRuntimeSection()

    assert section.resource_hints is None
    assert section.requires_isolation is False
    assert section.isolation_kind is None


# =============================================================================
# PluginSpecBuilder Tests
# =============================================================================


def test_builder_init() -> None:
    """Builder initializes with name."""
    builder = PluginSpecBuilder("test.plugin")

    # Build to verify name
    metadata = builder.build()
    assert metadata.name == "test.plugin"


def test_builder_description() -> None:
    """Builder sets description."""
    metadata = (
        PluginSpecBuilder("test.plugin")
        .description("A test plugin description")
        .build()
    )

    assert metadata.description == "A test plugin description"


def test_builder_stage() -> None:
    """Builder sets stage."""
    metadata = PluginSpecBuilder("test.plugin").stage("function").build()

    assert metadata.stage == "function"


def test_builder_version() -> None:
    """Builder sets version."""
    metadata = PluginSpecBuilder("test.plugin").version(CUSTOM_VERSION).build()

    assert metadata.version == CUSTOM_VERSION


def test_builder_enabled_by_default() -> None:
    """Builder sets enabled_by_default."""
    metadata = (
        PluginSpecBuilder("test.plugin").enabled_by_default(enabled=False).build()
    )

    assert metadata.enabled_by_default is False


def test_builder_severity() -> None:
    """Builder sets severity."""
    metadata = PluginSpecBuilder("test.plugin").severity("soft_fail").build()

    assert metadata.severity == "soft_fail"


def test_builder_input_from_type() -> None:
    """Builder creates input spec from type."""

    class MyConfig:
        """Test configuration class."""

    metadata = (
        PluginSpecBuilder("test.plugin")
        .input(MyConfig, required=True, source="config")
        .build()
    )

    assert len(metadata.inputs) == 1
    assert metadata.inputs[0].name == "MyConfig"
    assert metadata.inputs[0].required is True
    assert metadata.inputs[0].source == "config"


def test_builder_input_from_spec() -> None:
    """Builder accepts prebuilt PluginInputSpec."""
    input_spec = PluginInputSpec(
        name="custom_input",
        type_ref="CustomType",
        required=False,
        source="runtime",
    )

    metadata = PluginSpecBuilder("test.plugin").input(input_spec).build()

    assert len(metadata.inputs) == 1
    assert metadata.inputs[0] is input_spec


def test_builder_output_from_table_name() -> None:
    """Builder creates output spec from table name."""
    metadata = (
        PluginSpecBuilder("test.plugin")
        .output("analytics.test_table", min_rows=1, required_columns=["id", "name"])
        .build()
    )

    assert len(metadata.outputs) == 1
    assert metadata.outputs[0].name == "test_table"  # Without schema prefix
    assert metadata.outputs[0].tables == ("analytics.test_table",)
    assert metadata.outputs[0].min_rows == 1
    assert metadata.outputs[0].required_columns == ("id", "name")


def test_builder_output_from_spec() -> None:
    """Builder accepts prebuilt PluginOutputSpec."""
    output_spec = PluginOutputSpec(
        name="custom_output",
        tables=("analytics.custom",),
        min_rows=10,
    )

    metadata = PluginSpecBuilder("test.plugin").output(output_spec).build()

    assert len(metadata.outputs) == 1
    assert metadata.outputs[0] is output_spec


def test_builder_provides_from_strings() -> None:
    """Builder creates capabilities from strings."""
    metadata = (
        PluginSpecBuilder("test.plugin")
        .provides("capability.one", "capability.two")
        .build()
    )

    assert len(metadata.capabilities_provided) == EXPECTED_TWO_ITEMS
    cap_names = {cap.name for cap in metadata.capabilities_provided}
    assert "capability.one" in cap_names
    assert "capability.two" in cap_names


def test_builder_provides_from_capability() -> None:
    """Builder accepts PluginCapability instances."""
    cap = PluginCapability(name="custom.capability")

    metadata = PluginSpecBuilder("test.plugin").provides(cap).build()

    assert len(metadata.capabilities_provided) == 1
    assert metadata.capabilities_provided[0] is cap


def test_builder_requires_from_strings() -> None:
    """Builder creates required capabilities from strings."""
    metadata = (
        PluginSpecBuilder("test.plugin")
        .requires("required.one", "required.two")
        .build()
    )

    assert len(metadata.capabilities_required) == EXPECTED_TWO_ITEMS
    cap_names = {cap.name for cap in metadata.capabilities_required}
    assert "required.one" in cap_names
    assert "required.two" in cap_names


def test_builder_requires_from_capability() -> None:
    """Builder accepts PluginCapability instances for requires."""
    cap = PluginCapability(name="required.capability")

    metadata = PluginSpecBuilder("test.plugin").requires(cap).build()

    assert len(metadata.capabilities_required) == 1
    assert metadata.capabilities_required[0] is cap


def test_builder_depends_on() -> None:
    """Builder adds plugin dependencies."""
    metadata = (
        PluginSpecBuilder("test.plugin")
        .depends_on("plugin.one", "plugin.two")
        .build()
    )

    assert metadata.depends_on == ("plugin.one", "plugin.two")


def test_builder_resources() -> None:
    """Builder sets resource hints."""
    metadata = (
        PluginSpecBuilder("test.plugin")
        .resources(
            max_runtime_ms=MAX_RUNTIME_MS,
            max_memory_mb=MAX_MEMORY_MB,
            requires_gpu=True,
            priority=PRIORITY_HIGH,
        )
        .build()
    )

    hints = metadata.resource_hints
    assert hints is not None
    assert hints.max_runtime_ms == MAX_RUNTIME_MS
    assert hints.max_memory_mb == MAX_MEMORY_MB
    assert hints.requires_gpu is True
    assert hints.priority == PRIORITY_HIGH


def test_builder_isolate_process() -> None:
    """Builder sets process isolation."""
    metadata = PluginSpecBuilder("test.plugin").isolate(kind="process").build()

    assert metadata.requires_isolation is True
    assert metadata.isolation_kind == "process"


def test_builder_isolate_thread() -> None:
    """Builder sets thread isolation."""
    metadata = PluginSpecBuilder("test.plugin").isolate(kind="thread").build()

    assert metadata.requires_isolation is True
    assert metadata.isolation_kind == "thread"


def test_builder_tag() -> None:
    """Builder adds tags."""
    metadata = (
        PluginSpecBuilder("test.plugin").tag("function", "metrics", "core").build()
    )

    assert "function" in metadata.tags
    assert "metrics" in metadata.tags
    assert "core" in metadata.tags


def test_builder_chaining() -> None:
    """Builder methods can be chained."""
    metadata = (
        PluginSpecBuilder("chained.plugin")
        .description("A fully configured plugin")
        .stage("graph")
        .version("3.0.0")
        .enabled_by_default(enabled=True)
        .severity("fatal")
        .provides("output.capability")
        .requires("input.capability")
        .depends_on("base.plugin")
        .tag("chained", "test")
        .build()
    )

    assert metadata.name == "chained.plugin"
    assert metadata.description == "A fully configured plugin"
    assert metadata.stage == "graph"
    assert metadata.version == "3.0.0"
    assert len(metadata.capabilities_provided) == 1
    assert len(metadata.capabilities_required) == 1
    assert metadata.depends_on == ("base.plugin",)
    assert "chained" in metadata.tags


# =============================================================================
# PluginSpec Factory Tests
# =============================================================================


def test_plugin_spec_create() -> None:
    """PluginSpec.create returns a builder."""
    builder = PluginSpec.create("factory.plugin")

    assert isinstance(builder, PluginSpecBuilder)


def test_plugin_spec_create_builds_valid_metadata() -> None:
    """PluginSpec.create can build valid metadata."""
    metadata = (
        PluginSpec.create("factory.plugin")
        .description("Created via factory")
        .stage("function")
        .build()
    )

    assert metadata.name == "factory.plugin"
    assert metadata.description == "Created via factory"
    assert metadata.stage == "function"


def test_builder_multiple_inputs() -> None:
    """Builder can add multiple inputs."""

    class Config1:
        """First config."""

    class Config2:
        """Second config."""

    metadata = (
        PluginSpecBuilder("multi.input")
        .input(Config1, required=True)
        .input(Config2, required=False)
        .build()
    )

    assert len(metadata.inputs) == EXPECTED_TWO_ITEMS


def test_builder_multiple_outputs() -> None:
    """Builder can add multiple outputs."""
    metadata = (
        PluginSpecBuilder("multi.output")
        .output("analytics.table1")
        .output("analytics.table2")
        .output("analytics.table3")
        .build()
    )

    assert len(metadata.outputs) == EXPECTED_THREE_OUTPUTS


def test_builder_output_with_custom_name() -> None:
    """Builder output can have custom logical name."""
    metadata = (
        PluginSpecBuilder("named.output")
        .output("analytics.long_table_name", name="short")
        .build()
    )

    assert len(metadata.outputs) == 1
    assert metadata.outputs[0].name == "short"
    assert metadata.outputs[0].tables == ("analytics.long_table_name",)
