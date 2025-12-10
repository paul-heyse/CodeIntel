"""Tests for UnifiedRenderer."""

from __future__ import annotations

import json
from io import StringIO

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.rendering import (
    ColumnSpec,
    OutputFormat,
    RenderContext,
    TableSpec,
    UnifiedRenderer,
)
from codeintel.cli.results import CliResult
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_length,
)


def test_render_message_text() -> None:
    """Verify render_message outputs text correctly."""
    ctx, out, _err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)

    renderer.render_message("Test message", level="info")

    expect_in("Test message", out.getvalue())


def test_render_message_json() -> None:
    """Verify render_message outputs JSON correctly."""
    out = StringIO()
    err = StringIO()
    ctx = RenderContext(
        format=OutputFormat.JSON,
        color=False,
        writer=out,
        err_writer=err,
        is_tty=False,
    )
    renderer = UnifiedRenderer(ctx)

    renderer.render_message("Test message", level="success")

    output = json.loads(out.getvalue())
    expect_equal(output["status"], "success")
    expect_equal(output["message"], "Test message")


def test_render_table_text() -> None:
    """Verify render_table outputs plain table correctly."""
    ctx, out, _err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)

    rows = [
        {"name": "foo", "count": 10},
        {"name": "bar", "count": 20},
    ]
    spec = TableSpec(
        columns=(
            ColumnSpec("name", "Name"),
            ColumnSpec("count", "Count"),
        ),
    )

    renderer.render_table(rows, spec)

    output = out.getvalue()
    expect_in("Name", output)
    expect_in("Count", output)
    expect_in("foo", output)
    expect_in("bar", output)


def test_render_table_json() -> None:
    """Verify render_table outputs JSON array correctly."""
    out = StringIO()
    err = StringIO()
    ctx = RenderContext(
        format=OutputFormat.JSON,
        color=False,
        writer=out,
        err_writer=err,
        is_tty=False,
    )
    renderer = UnifiedRenderer(ctx)

    expected_count = 2
    expected_second_count = 20

    rows = [
        {"name": "foo", "count": 10},
        {"name": "bar", "count": expected_second_count},
    ]
    spec = TableSpec(
        columns=(
            ColumnSpec("name", "Name"),
            ColumnSpec("count", "Count"),
        ),
    )

    renderer.render_table(rows, spec)

    output = json.loads(out.getvalue())
    expect_length(output, expected_count)
    expect_equal(output[0]["name"], "foo")
    expect_equal(output[1]["count"], expected_second_count)


def test_render_table_empty() -> None:
    """Verify render_table handles empty rows."""
    ctx, out, _err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)

    spec = TableSpec(
        columns=(ColumnSpec("name", "Name"),),
        empty_message="No items found.",
    )

    renderer.render_table([], spec)

    expect_in("No items found", out.getvalue())


def test_render_error_text() -> None:
    """Verify render_error outputs to stderr in text mode."""
    ctx, _out, err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)

    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        status=400,
        detail="Something went wrong",
    )

    renderer.render_error(error)

    expect_in("Test Error", err.getvalue())
    expect_in("Something went wrong", err.getvalue())


def test_render_error_json() -> None:
    """Verify render_error outputs JSON to stderr."""
    out = StringIO()
    err = StringIO()
    ctx = RenderContext(
        format=OutputFormat.JSON,
        color=False,
        writer=out,
        err_writer=err,
        is_tty=False,
    )
    renderer = UnifiedRenderer(ctx)

    expected_status = 400

    error = ProblemDetail(
        type="urn:test:error",
        title="Test Error",
        status=expected_status,
        detail="Something went wrong",
    )

    renderer.render_error(error)

    output = json.loads(err.getvalue())
    expect_equal(output["type"], "urn:test:error")
    expect_equal(output["title"], "Test Error")
    expect_equal(output["status"], expected_status)


def test_render_result_success() -> None:
    """Verify render_result handles success correctly."""
    out = StringIO()
    err = StringIO()
    ctx = RenderContext(
        format=OutputFormat.JSON,
        color=False,
        writer=out,
        err_writer=err,
        is_tty=False,
    )
    renderer = UnifiedRenderer(ctx)

    result: CliResult[dict[str, str]] = CliResult.ok({"key": "value"})

    exit_code = renderer.render_result(result)

    expect_equal(exit_code, 0)
    output = json.loads(out.getvalue())
    expect_equal(output["data"]["key"], "value")


def test_render_result_failure() -> None:
    """Verify render_result handles failure correctly."""
    ctx, _out, err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)

    error = ProblemDetail(
        type="urn:test:error",
        title="Failed",
        status=400,
    )
    result: CliResult[dict[str, str]] = CliResult.fail(error)

    exit_code = renderer.render_result(result)

    expect_equal(exit_code, 1)
    expect_in("Failed", err.getvalue())


def test_render_result_with_warnings() -> None:
    """Verify render_result emits warnings to stderr."""
    ctx, _out, err = RenderContext.for_testing()
    renderer = UnifiedRenderer(ctx)

    result: CliResult[dict[str, str]] = CliResult.ok({"key": "value"})
    result.warnings.append("Warning 1")
    result.warnings.append("Warning 2")

    renderer.render_result(result)

    expect_in("Warning 1", err.getvalue())
    expect_in("Warning 2", err.getvalue())
