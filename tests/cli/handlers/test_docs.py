"""Tests for docs handlers."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

from codeintel.cli.errors import ValidationError
from codeintel.cli.handlers.docs import (
    DocsExportResult,
    DocsValidateResult,
    ExportMode,
    docs_export_handler,
    docs_validate_handler,
)
from codeintel.cli.handlers.protocol import EnhancedHandlerContext
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)


def _make_mock_context(params: dict[str, Any]) -> EnhancedHandlerContext:
    """Create a mock EnhancedHandlerContext for testing.

    Parameters
    ----------
    params
        Parameters to include in the context.

    Returns
    -------
    EnhancedHandlerContext
        Mock context with provided params.
    """
    ctx = MagicMock(spec=EnhancedHandlerContext)
    ctx.params = params
    ctx.logger = logging.getLogger("test")
    return ctx


def test_docs_export_result_to_dict() -> None:
    """Verify DocsExportResult.to_dict returns correct structure."""
    result = DocsExportResult(
        status="ok",
        validation="required",
        macro_requirement="require_normalized",
        datasets=["dataset1", "dataset2"],
        schemas=["schema1"],
        mode="build_system",
    )

    data = result.to_dict()

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
        mode="direct",
    )

    data = result.to_dict()

    expect_equal(data["datasets"], None)
    expect_equal(data["schemas"], None)


def test_docs_validate_result_to_dict() -> None:
    """Verify DocsValidateResult.to_dict returns correct structure."""
    result = DocsValidateResult(
        passed=True,
        issues=[],
    )

    data = result.to_dict()

    expect_true(data["passed"])
    expect_equal(data["issues"], [])


def test_docs_validate_result_with_issues() -> None:
    """Verify DocsValidateResult.to_dict with issues returns correct structure."""
    result = DocsValidateResult(
        passed=False,
        issues=["Missing export file", "Invalid schema"],
    )

    data = result.to_dict()

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


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_default_params(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler uses default parameters."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({})

    result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.validation, "required")
        expect_equal(data.macro_requirement, "require_normalized")
        expect_equal(data.mode, "build_system")


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_dry_run_mode(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler handles dry_run parameter."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({"dry_run": True})

    result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.status, "dry_run")
        expect_equal(data.mode, "dry_run")


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_skip_prereqs(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler handles skip_prereqs parameter."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({"skip_prereqs": True})

    result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.mode, "direct")


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_with_datasets_filter(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler handles datasets parameter."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({"datasets": ["dataset1", "dataset2"]})

    result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.datasets, ["dataset1", "dataset2"])


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_with_schemas_filter(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler handles schemas parameter."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({"schemas": ["schema1"]})

    result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.schemas, ["schema1"])


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_custom_validation(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler handles custom validation mode."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({"validation": "skip"})

    result = docs_export_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.validation, "skip")


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_validate_handler_success(mock_build_runtime: MagicMock) -> None:
    """Verify docs_validate_handler returns success."""
    mock_runtime = MagicMock()
    mock_build_runtime.return_value = mock_runtime

    ctx = _make_mock_context({})

    result = docs_validate_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_true(data.passed)
        expect_equal(data.issues, [])


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_export_handler_project_error(mock_build_runtime: MagicMock) -> None:
    """Verify docs_export_handler handles project errors."""
    mock_build_runtime.side_effect = ValidationError("Project not found")

    ctx = _make_mock_context({})

    result = docs_export_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:docs:project-error")


@patch("codeintel.cli.handlers.docs._build_runtime_from_ctx")
def test_docs_validate_handler_project_error(mock_build_runtime: MagicMock) -> None:
    """Verify docs_validate_handler handles project errors."""
    mock_build_runtime.side_effect = ValidationError("Project not found")

    ctx = _make_mock_context({})

    result = docs_validate_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:docs:project-error")
