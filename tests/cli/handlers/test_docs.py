"""Tests for docs handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.cli.core.results import SerializableResult
from codeintel.cli.errors import ValidationError
from codeintel.cli.handlers.docs import (
    DocsDependencies,
    DocsExportResult,
    DocsValidateResult,
    ExportMode,
    docs_export_handler,
    docs_validate_handler,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from tests.cli.handlers.conftest import DocsHandlerHarness


def _result_to_dict(result: object) -> dict[str, object]:
    return cast("SerializableResult", result).to_dict()


def test_docs_export_result_to_dict() -> None:
    """Verify DocsExportResult.to_dict returns correct structure."""
    result = DocsExportResult(
        status="ok",
        validation="required",
        macro_requirement="require_normalized",
        datasets=["dataset1", "dataset2"],
        schemas=["schema1"],
        mode=ExportMode.BUILD_SYSTEM,
    )

    data = _result_to_dict(result)

    expect_equal(data["status"], "ok")
    expect_equal(data["validation"], "required")
    expect_equal(data["macro_requirement"], "require_normalized")
    expect_equal(data["datasets"], ["dataset1", "dataset2"])
    expect_equal(data["schemas"], ["schema1"])
    expect_equal(data["mode"], "build_system")


def test_docs_export_result_with_none_datasets() -> None:
    """Verify DocsExportResult.to_dict handles None datasets/schemas."""
    result = DocsExportResult(
        status="ok",
        validation="required",
        macro_requirement="require_normalized",
        datasets=None,
        schemas=None,
        mode=ExportMode.DIRECT,
    )

    data = _result_to_dict(result)

    expect_true("datasets" not in data)
    expect_true("schemas" not in data)


def test_docs_validate_result_to_dict() -> None:
    """Verify DocsValidateResult.to_dict returns correct structure."""
    result = DocsValidateResult(
        passed=True,
        issues=[],
    )

    data = _result_to_dict(result)

    expect_true(data["passed"])
    expect_equal(data["issues"], [])


def test_docs_validate_result_with_issues() -> None:
    """Verify DocsValidateResult.to_dict with issues returns correct structure."""
    result = DocsValidateResult(
        passed=False,
        issues=["Missing export file", "Invalid schema"],
    )

    data = _result_to_dict(result)

    expect_true(not data["passed"])
    issues = data["issues"]
    expect_true(isinstance(issues, list))
    if isinstance(issues, list):
        expect_equal(len(issues), 2)


def test_export_mode_values() -> None:
    """Verify ExportMode enum values."""
    expect_equal(ExportMode.BUILD_SYSTEM.value, "build_system")
    expect_equal(ExportMode.DIRECT.value, "direct")
    expect_equal(ExportMode.DRY_RUN.value, "dry_run")


def test_docs_export_handler_default_params(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler uses default parameters."""
    with docs_handler_harness_fixture.command_context({}) as ctx:
        result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.validation, "required")
        expect_equal(data.macro_requirement, "require_normalized")
        expect_equal(data.mode, ExportMode.BUILD_SYSTEM)


def test_docs_export_handler_dry_run_mode(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler handles dry_run parameter."""
    with docs_handler_harness_fixture.command_context({"dry_run": True}) as ctx:
        result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.status, "dry_run")
        expect_equal(data.mode, ExportMode.DRY_RUN)


def test_docs_export_handler_skip_prereqs(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler handles skip_prereqs parameter."""
    with docs_handler_harness_fixture.command_context({"skip_prereqs": True}) as ctx:
        result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.mode, ExportMode.DIRECT)


def test_docs_export_handler_with_datasets_filter(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler handles datasets parameter."""
    with docs_handler_harness_fixture.command_context(
        {"datasets": ["dataset1", "dataset2"]}
    ) as ctx:
        result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.datasets, ["dataset1", "dataset2"])


def test_docs_export_handler_with_schemas_filter(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler handles schemas parameter."""
    with docs_handler_harness_fixture.command_context({"schemas": ["schema1"]}) as ctx:
        result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.schemas, ["schema1"])


def test_docs_export_handler_custom_validation(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler handles custom validation mode."""
    with docs_handler_harness_fixture.command_context({"validation": "skip"}) as ctx:
        result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.validation, "skip")


def test_docs_validate_handler_success(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_validate_handler returns success."""
    with docs_handler_harness_fixture.command_context({}) as ctx:
        result = docs_validate_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_true(data.passed)
        expect_equal(data.issues, [])


def test_docs_export_handler_project_error(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_export_handler handles project errors."""
    msg = "Project not found"

    def _raise(_: object) -> object:
        raise ValidationError(msg)

    deps = DocsDependencies(runtime_builder=_raise)

    with docs_handler_harness_fixture.command_context({}) as ctx:
        result = docs_export_handler(ctx, deps=deps)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:docs:project-error")


def test_docs_validate_handler_project_error(
    docs_handler_harness_fixture: DocsHandlerHarness,
) -> None:
    """Verify docs_validate_handler handles project errors."""
    msg = "Project not found"

    def _raise(_: object) -> object:
        raise ValidationError(msg)

    deps = DocsDependencies(runtime_builder=_raise)

    with docs_handler_harness_fixture.command_context({}) as ctx:
        result = docs_validate_handler(ctx, deps=deps)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:docs:project-error")
