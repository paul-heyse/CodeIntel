"""Tests for the unified plugin protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from codeintel.analytics.core.plugin_protocol import (
    CapabilityKind,
    InputSource,
    PluginCapability,
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from tests._helpers.assertions import assert_cannot_setattr


@pytest.mark.parametrize(
    ("name", "kind", "expected_kind"),
    [
        ("analytics.function_metrics", "dataset", "dataset"),
        ("test.cap", None, "dataset"),
    ],
)
def test_plugin_capability_kind(
    name: str, kind: CapabilityKind | None, expected_kind: CapabilityKind
) -> None:
    """Capabilities default to dataset kind unless overridden."""
    cap = PluginCapability(name=name, kind=kind or "dataset")

    assert cap.name == name
    assert cap.kind == expected_kind


def test_plugin_capability_is_frozen() -> None:
    """Capabilities should be immutable after creation."""
    cap = PluginCapability(name="test")

    assert_cannot_setattr(cap, "name", "other")


@dataclass(frozen=True)
class InputSpecCase:
    """Case describing expected PluginInputSpec fields."""

    name: str
    type_ref: str
    required: bool
    source: InputSource
    default: object | None


@pytest.mark.parametrize(
    "case",
    [
        InputSpecCase(
            name="config",
            type_ref="FunctionAnalyticsStepConfig",
            required=True,
            source="config",
            default=None,
        ),
        InputSpecCase(
            name="runtime_opt",
            type_ref="dict",
            required=False,
            source="runtime",
            default={"key": "value"},
        ),
    ],
)
def test_plugin_input_spec_creation(case: InputSpecCase) -> None:
    """Input specs should preserve provided fields."""
    spec = PluginInputSpec(
        name=case.name,
        type_ref=case.type_ref,
        required=case.required,
        source=case.source,
        default=case.default,
    )

    assert spec.name == case.name
    assert spec.type_ref == case.type_ref
    assert spec.required is case.required
    assert spec.source == case.source
    assert spec.default == case.default


@pytest.mark.parametrize(
    ("spec", "expected_tables", "expected_artifact"),
    [
        (
            PluginOutputSpec(
                name="metrics",
                tables=("analytics.function_metrics",),
                min_rows=1,
                required_columns=("repo", "commit", "goid"),
            ),
            ("analytics.function_metrics",),
            None,
        ),
        (
            PluginOutputSpec(
                name="report",
                artifact_type="json_report",
            ),
            (),
            "json_report",
        ),
    ],
)
def test_plugin_output_spec_creation(
    spec: PluginOutputSpec, expected_tables: tuple[str, ...], expected_artifact: str | None
) -> None:
    """Output specs should reflect provided tables or artifacts."""
    assert spec.tables == expected_tables
    assert spec.artifact_type == expected_artifact


@dataclass(frozen=True)
class ValidationCase:
    """Case describing expected validation result."""

    factory: Callable[[], ValidationResult]
    expected_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@pytest.mark.parametrize(
    "case",
    [
        ValidationCase(
            factory=ValidationResult.success,
            expected_valid=True,
            errors=(),
            warnings=(),
        ),
        ValidationCase(
            factory=lambda: ValidationResult.failure(("error1", "error2")),
            expected_valid=False,
            errors=("error1", "error2"),
            warnings=(),
        ),
        ValidationCase(
            factory=lambda: ValidationResult(valid=True, warnings=("warning1",)),
            expected_valid=True,
            errors=(),
            warnings=("warning1",),
        ),
    ],
)
def test_validation_result(case: ValidationCase) -> None:
    """ValidationResult factories and constructor should set state correctly."""
    result = case.factory()

    assert result.valid is case.expected_valid
    assert result.errors == case.errors
    assert result.warnings == case.warnings


def test_plugin_result_ok() -> None:
    """PluginResult.ok should mark success and preserve metadata."""
    result = PluginResult.ok(
        row_counts={"table1": 100},
        meta={"duration_ms": 50},
    )

    assert result.success is True
    assert result.row_counts == {"table1": 100}
    assert result.meta == {"duration_ms": 50}
    assert result.error is None


@pytest.mark.parametrize(
    ("row_counts", "artifacts", "meta"),
    [
        ({"table1": 100}, {}, {"duration_ms": 50}),
        ({}, {"report": b"data"}, {"note": "with artifacts"}),
    ],
)
def test_plugin_result_ok_variants(
    row_counts: dict[str, int], artifacts: dict[str, object], meta: dict[str, object]
) -> None:
    """PluginResult.ok variants should preserve provided data."""
    result = PluginResult.ok(row_counts=row_counts, artifacts=artifacts, meta=meta)

    assert result.success is True
    assert result.row_counts == row_counts
    assert result.artifacts == artifacts
    assert result.meta == meta
    assert result.error is None


@pytest.mark.parametrize(
    ("error", "warnings", "artifacts", "row_counts"),
    [
        ("Something went wrong", (), {}, {}),
        ("Fatal error", ("warning1", "warning2"), {"log": "details"}, {"table1": 0}),
    ],
)
def test_plugin_result_fail(
    error: str, warnings: tuple[str, ...], artifacts: dict[str, object], row_counts: dict[str, int]
) -> None:
    """PluginResult.fail should capture errors, warnings, and optional data."""
    result = PluginResult.fail(error, warnings=warnings)
    result = (
        result
        if artifacts == {} and row_counts == {}
        else PluginResult(
            success=False,
            error=error,
            warnings=warnings,
            artifacts=artifacts,
            row_counts=row_counts,
        )
    )

    assert result.success is False
    assert result.error == error
    assert result.row_counts == row_counts
    assert result.warnings == warnings


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        (
            PluginResourceHints(
                max_runtime_ms=60000,
                max_memory_mb=1024,
                requires_gpu=False,
                priority=10,
            ),
            {"max_runtime_ms": 60000, "max_memory_mb": 1024, "requires_gpu": False, "priority": 10},
        ),
        (
            PluginResourceHints(),
            {"max_runtime_ms": None, "max_memory_mb": None, "requires_gpu": False, "priority": 0},
        ),
    ],
)
def test_plugin_resource_hints(hints: PluginResourceHints, expected: dict[str, object]) -> None:
    """Resource hints should expose provided values or defaults."""
    assert hints.max_runtime_ms == expected["max_runtime_ms"]
    assert hints.max_memory_mb == expected["max_memory_mb"]
    assert hints.requires_gpu is expected["requires_gpu"]
    assert hints.priority == expected["priority"]


def test_plugin_metadata_minimal() -> None:
    """Minimal metadata should fill defaults."""
    meta = PluginMetadata(
        name="test.plugin",
        description="A test plugin",
        stage="function",
    )

    assert meta.name == "test.plugin"
    assert meta.description == "A test plugin"
    assert meta.stage == "function"
    assert meta.version == "1.0.0"
    assert meta.enabled_by_default is True
    assert meta.severity == "fatal"


def test_plugin_metadata_full() -> None:
    """Full metadata should preserve all provided fields."""
    meta = PluginMetadata(
        name="full.plugin",
        description="Full plugin",
        stage="graph",
        version="2.0.0",
        enabled_by_default=False,
        severity="soft_fail",
        inputs=(PluginInputSpec("cfg", "Config"),),
        outputs=(PluginOutputSpec("out", tables=("t1",)),),
        capabilities_provided=(PluginCapability("cap1"),),
        capabilities_required=(PluginCapability("cap2"),),
        depends_on=("other.plugin",),
        resource_hints=PluginResourceHints(max_runtime_ms=1000),
        requires_isolation=True,
        isolation_kind="process",
        tags=("tag1", "tag2"),
    )

    assert meta.name == "full.plugin"
    assert meta.version == "2.0.0"
    assert meta.enabled_by_default is False
    assert meta.severity == "soft_fail"
    assert len(meta.inputs) == 1
    assert len(meta.outputs) == 1
    assert len(meta.capabilities_provided) == 1
    assert len(meta.capabilities_required) == 1
    assert meta.depends_on == ("other.plugin",)
    assert meta.resource_hints is not None
    assert meta.requires_isolation is True
    assert meta.isolation_kind == "process"
    assert meta.tags == ("tag1", "tag2")
