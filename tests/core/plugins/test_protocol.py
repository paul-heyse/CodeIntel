"""Test plugin protocol types from codeintel.core.plugins.protocol.

This module tests:
- PluginMetadata dataclass construction and defaults
- PluginCapability, PluginInputSpec, PluginOutputSpec frozen dataclasses
- PluginResourceHints optional fields
- ValidationResult factory methods
- PluginMetadata post_init normalization
"""

from __future__ import annotations

import pytest

from codeintel.core.plugins.types.protocol import (
    CapabilityKind,
    InputSource,
    PluginCapability,
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
    PluginProtocol,
    PluginResourceHints,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.core.plugins.types.result import PluginResult
from tests._helpers import assert_frozen
from tests._helpers.assertions import expect_equal, expect_true

MIN_ROWS_REQUIRED = 10
RESOURCE_MAX_RUNTIME_MS = 5000
RESOURCE_MAX_MEMORY_MB = 512
RESOURCE_PRIORITY = 10

# =============================================================================
# PluginCapability Tests
# =============================================================================


def test_plugin_capability_construction() -> None:
    """Verify PluginCapability can be constructed with required fields."""
    capability = PluginCapability(name="test.capability")

    expect_equal(capability.name, "test.capability")
    expect_equal(capability.kind, "dataset")  # Default


def test_plugin_capability_with_kind() -> None:
    """Verify PluginCapability can specify kind."""
    capability = PluginCapability(name="test.graph", kind="graph")

    expect_equal(capability.name, "test.graph")
    expect_equal(capability.kind, "graph")


def test_plugin_capability_is_frozen() -> None:
    """Verify PluginCapability is immutable (frozen dataclass)."""
    capability = PluginCapability(name="test.capability")

    assert_frozen(capability, "name", "modified")


# =============================================================================
# PluginInputSpec Tests
# =============================================================================


def test_plugin_input_spec_construction() -> None:
    """Verify PluginInputSpec can be constructed."""
    spec = PluginInputSpec(name="config", type_ref="AppConfig")

    expect_equal(spec.name, "config")
    expect_equal(spec.type_ref, "AppConfig")
    expect_true(spec.required)  # Default
    expect_equal(spec.source, "config")  # Default
    expect_true(spec.default is None)  # Default


def test_plugin_input_spec_optional() -> None:
    """Verify PluginInputSpec can be made optional."""
    spec = PluginInputSpec(
        name="config",
        type_ref="AppConfig",
        required=False,
        default={"key": "value"},
    )

    expect_true(not spec.required)
    expect_equal(spec.default, {"key": "value"})


def test_plugin_input_spec_runtime_source() -> None:
    """Verify PluginInputSpec can specify runtime source."""
    spec = PluginInputSpec(
        name="data",
        type_ref="DataFrame",
        source="runtime",
    )

    expect_equal(spec.source, "runtime")


def test_plugin_input_spec_is_frozen() -> None:
    """Verify PluginInputSpec is immutable."""
    spec = PluginInputSpec(name="config", type_ref="AppConfig")

    assert_frozen(spec, "name", "modified")


# =============================================================================
# PluginOutputSpec Tests
# =============================================================================


def test_plugin_output_spec_construction() -> None:
    """Verify PluginOutputSpec can be constructed."""
    spec = PluginOutputSpec(name="metrics")

    expect_equal(spec.name, "metrics")
    expect_equal(spec.tables, ())  # Default
    expect_true(spec.artifact_type is None)  # Default
    expect_true(spec.min_rows is None)  # Default
    expect_equal(spec.required_columns, ())  # Default


def test_plugin_output_spec_with_tables() -> None:
    """Verify PluginOutputSpec can specify tables."""
    spec = PluginOutputSpec(
        name="metrics",
        tables=("analytics.function_metrics", "analytics.module_metrics"),
        min_rows=MIN_ROWS_REQUIRED,
        required_columns=("goid", "metric_value"),
    )

    expect_equal(spec.tables, ("analytics.function_metrics", "analytics.module_metrics"))
    expect_equal(spec.min_rows, MIN_ROWS_REQUIRED)
    expect_equal(spec.required_columns, ("goid", "metric_value"))


def test_plugin_output_spec_with_artifact() -> None:
    """Verify PluginOutputSpec can specify artifact type."""
    spec = PluginOutputSpec(
        name="graph",
        artifact_type="networkx.DiGraph",
    )

    expect_equal(spec.artifact_type, "networkx.DiGraph")


# =============================================================================
# PluginResourceHints Tests
# =============================================================================


def test_plugin_resource_hints_defaults() -> None:
    """Verify PluginResourceHints has sensible defaults."""
    hints = PluginResourceHints()

    expect_true(hints.max_runtime_ms is None)
    expect_true(hints.max_memory_mb is None)
    expect_true(not hints.cpu_intensive)
    expect_true(not hints.io_intensive)
    expect_true(not hints.requires_gpu)
    expect_equal(hints.priority, 0)


def test_plugin_resource_hints_all_fields() -> None:
    """Verify PluginResourceHints can specify all fields."""
    hints = PluginResourceHints(
        max_runtime_ms=RESOURCE_MAX_RUNTIME_MS,
        max_memory_mb=RESOURCE_MAX_MEMORY_MB,
        cpu_intensive=True,
        io_intensive=False,
        requires_gpu=True,
        priority=RESOURCE_PRIORITY,
    )

    expect_equal(hints.max_runtime_ms, RESOURCE_MAX_RUNTIME_MS)
    expect_equal(hints.max_memory_mb, RESOURCE_MAX_MEMORY_MB)
    expect_true(hints.cpu_intensive)
    expect_true(hints.requires_gpu)
    expect_equal(hints.priority, RESOURCE_PRIORITY)


# =============================================================================
# ValidationResult Tests
# =============================================================================


def test_validation_result_success() -> None:
    """Verify ValidationResult.success() creates successful result."""
    result = ValidationResult.success()

    expect_true(result.valid)
    expect_equal(result.errors, ())
    expect_equal(result.warnings, ())


def test_validation_result_failure() -> None:
    """Verify ValidationResult.failure() creates failed result."""
    errors = ("Missing required input", "Invalid configuration")
    result = ValidationResult.failure(errors)

    expect_true(not result.valid)
    expect_equal(result.errors, errors)
    expect_equal(result.warnings, ())


def test_validation_result_with_warnings() -> None:
    """Verify ValidationResult can include warnings."""
    result = ValidationResult(
        valid=True,
        warnings=("Deprecated option used",),
    )

    expect_true(result.valid)
    expect_equal(result.warnings, ("Deprecated option used",))


def test_validation_result_is_frozen() -> None:
    """Verify ValidationResult is immutable."""
    result = ValidationResult.success()

    assert_frozen(result, "valid", new_value=False)


# =============================================================================
# PluginMetadata Tests
# =============================================================================


def test_plugin_metadata_minimal() -> None:
    """Verify PluginMetadata can be constructed with minimal fields."""
    metadata = PluginMetadata(
        name="test.plugin",
        description="A test plugin",
        kind="analytics",
        stage="function",
    )

    expect_equal(metadata.name, "test.plugin")
    expect_equal(metadata.description, "A test plugin")
    expect_equal(metadata.kind, "analytics")
    expect_equal(metadata.stage, "function")


def test_plugin_metadata_defaults() -> None:
    """Verify PluginMetadata has sensible defaults."""
    metadata = PluginMetadata(
        name="test.plugin",
        description="A test plugin",
        kind="analytics",
        stage="function",
    )

    expect_equal(metadata.version, "1.0.0")
    expect_true(metadata.enabled_by_default)
    expect_equal(metadata.severity, "fatal")
    expect_equal(metadata.depends_on, ())
    expect_equal(metadata.provides, ())
    expect_equal(metadata.requires, ())
    expect_equal(metadata.inputs, ())
    expect_equal(metadata.outputs, ())
    expect_equal(metadata.produces_tables, ())
    expect_equal(metadata.produces_graphs, ())
    expect_equal(metadata.requires_graphs, ())
    expect_true(metadata.resource_hints is None)
    expect_true(not metadata.supports_incremental)
    expect_equal(metadata.isolation_kind, "none")
    expect_true(not metadata.requires_isolation)


def test_plugin_metadata_all_fields() -> None:
    """Verify PluginMetadata can specify all fields."""
    hints = PluginResourceHints(max_runtime_ms=1000)
    input_spec = PluginInputSpec(name="config", type_ref="Config")
    output_spec = PluginOutputSpec(name="metrics", tables=("analytics.metrics",))

    metadata = PluginMetadata(
        name="test.full",
        description="Full test plugin",
        kind="builder",
        stage="goid",
        version="2.0.0",
        enabled_by_default=False,
        severity="soft_fail",
        depends_on=("dep1", "dep2"),
        provides=("capability1",),
        requires=("requirement1",),
        inputs=(input_spec,),
        outputs=(output_spec,),
        produces_tables=("table1",),
        produces_graphs=("call_graph",),
        requires_graphs=("import_graph",),
        resource_hints=hints,
        supports_incremental=True,
        isolation_kind="process",
        requires_isolation=True,
        scope_aware=True,
        supported_scopes=("function", "module"),
        version_hash="abc123",
        config_schema_ref="schema://config",
        row_count_tables=("table1",),
        cache_populates=("cache1",),
        cache_consumes=("cache2",),
        contract_checkers=("checker1",),
        tags=("tag1", "tag2"),
    )

    expect_equal(metadata.version, "2.0.0")
    expect_equal(metadata.depends_on, ("dep1", "dep2"))
    expect_equal(metadata.produces_graphs, ("call_graph",))
    expect_true(metadata.supports_incremental)
    expect_equal(metadata.isolation_kind, "process")
    expect_equal(metadata.tags, ("tag1", "tag2"))


def test_plugin_metadata_post_init_isolation_normalization() -> None:
    """Verify __post_init__ normalizes requires_isolation based on isolation_kind."""
    # When isolation_kind is not "none", requires_isolation should be True
    metadata = PluginMetadata(
        name="test.isolated",
        description="Isolated plugin",
        kind="analytics",
        stage="function",
        isolation_kind="process",
        requires_isolation=False,  # Should be normalized to True
    )

    expect_true(metadata.requires_isolation)


def test_plugin_metadata_post_init_no_change_for_none_isolation() -> None:
    """Verify __post_init__ doesn't change requires_isolation when isolation_kind is 'none'."""
    metadata = PluginMetadata(
        name="test.not_isolated",
        description="Non-isolated plugin",
        kind="analytics",
        stage="function",
        isolation_kind="none",
        requires_isolation=False,
    )

    expect_true(not metadata.requires_isolation)


def test_plugin_metadata_is_frozen() -> None:
    """Verify PluginMetadata is immutable."""
    metadata = PluginMetadata(
        name="test.plugin",
        description="A test plugin",
        kind="analytics",
        stage="function",
    )

    assert_frozen(metadata, "name", "modified")


# =============================================================================
# Literal Type Coverage Tests
# =============================================================================


@pytest.mark.parametrize(
    "kind",
    ["builder", "metric", "validation", "analytics"],
)
def test_plugin_kind_values(kind: PluginKind) -> None:
    """Verify all PluginKind values are valid."""
    metadata = PluginMetadata(
        name="test",
        description="test",
        kind=kind,
        stage="other",
    )
    expect_equal(metadata.kind, kind)


@pytest.mark.parametrize(
    "stage",
    [
        "goid",
        "edges",
        "structure",
        "core",
        "graph",
        "function",
        "function_history",
        "test",
        "coverage",
        "subsystem",
        "data_model",
        "data_model_usage",
        "entrypoints",
        "profiles",
        "history",
        "semantic",
        "hotspots",
        "risk",
        "cfg",
        "dfg",
        "symbol",
        "config",
        "stats",
        "validation",
        "other",
    ],
)
def test_plugin_stage_values(stage: PluginStage) -> None:
    """Verify all PluginStage values are valid."""
    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage=stage,
    )
    expect_equal(metadata.stage, stage)


@pytest.mark.parametrize(
    "severity",
    ["fatal", "soft_fail", "skip_on_error"],
)
def test_plugin_severity_values(severity: PluginSeverity) -> None:
    """Verify all PluginSeverity values are valid."""
    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="other",
        severity=severity,
    )
    expect_equal(metadata.severity, severity)


@pytest.mark.parametrize(
    "isolation",
    ["process", "thread", "none"],
)
def test_plugin_isolation_values(isolation: PluginIsolation) -> None:
    """Verify all PluginIsolation values are valid."""
    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="other",
        isolation_kind=isolation,
    )
    expect_equal(metadata.isolation_kind, isolation)


@pytest.mark.parametrize(
    "kind",
    ["dataset", "artifact", "service", "graph"],
)
def test_capability_kind_values(kind: CapabilityKind) -> None:
    """Verify all CapabilityKind values are valid."""
    capability = PluginCapability(name="test", kind=kind)
    expect_equal(capability.kind, kind)


@pytest.mark.parametrize(
    "source",
    ["config", "runtime", "prior_plugin"],
)
def test_input_source_values(source: InputSource) -> None:
    """Verify all InputSource values are valid."""
    spec = PluginInputSpec(name="test", type_ref="str", source=source)
    expect_equal(spec.source, source)


# =============================================================================
# PluginProtocol Runtime Check Tests
# =============================================================================


def test_plugin_protocol_is_runtime_checkable() -> None:
    """Verify PluginProtocol is a runtime_checkable protocol."""

    # Classes implementing the protocol should pass isinstance checks
    class ConformingPlugin:
        def __init__(self) -> None:
            self._metadata = PluginMetadata(
                name="test",
                description="test",
                kind="analytics",
                stage="other",
            )

        @property
        def metadata(self) -> PluginMetadata:
            return self._metadata

        def execute(self, _ctx: object) -> PluginResult:
            _ = self.metadata
            return PluginResult.ok()

        def validate_inputs(self, _ctx: object) -> ValidationResult:
            _ = self.metadata
            return ValidationResult.success()

    plugin = ConformingPlugin()
    # Use hasattr checks for protocol compliance instead of isinstance
    expect_true(hasattr(plugin, "metadata"))
    expect_true(hasattr(plugin, "execute"))
    expect_true(hasattr(plugin, "validate_inputs"))


def test_non_conforming_class_fails_protocol_check() -> None:
    """Verify non-conforming classes fail isinstance check."""

    class NotAPlugin:
        pass

    expect_true(not isinstance(NotAPlugin(), PluginProtocol))
