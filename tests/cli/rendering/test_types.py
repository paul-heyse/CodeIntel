"""Tests for rendering types."""

from __future__ import annotations

import sys
from io import StringIO

from codeintel.cli.rendering import OutputFormat, RenderContext
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)


def test_output_format_values() -> None:
    """Validate OutputFormat has expected values."""
    expect_equal(OutputFormat.TEXT.value, "text")
    expect_equal(OutputFormat.JSON.value, "json")
    expect_equal(OutputFormat.JSONL.value, "jsonl")


def test_render_context_auto_detect_tty() -> None:
    """Verify auto_detect uses TEXT format for TTY."""
    mock_stdout = StringIO()
    mock_stdout.isatty = lambda: True  # type: ignore[method-assign]
    original_stdout = sys.stdout
    try:
        sys.stdout = mock_stdout  # type: ignore[assignment]
        ctx = RenderContext.auto_detect()
        expect_equal(ctx.format, OutputFormat.TEXT)
        expect_true(ctx.color)
        expect_true(ctx.is_tty)
    finally:
        sys.stdout = original_stdout


def test_render_context_auto_detect_non_tty() -> None:
    """Verify auto_detect uses JSON format for non-TTY."""
    mock_stdout = StringIO()
    mock_stdout.isatty = lambda: False  # type: ignore[method-assign]
    original_stdout = sys.stdout
    try:
        sys.stdout = mock_stdout  # type: ignore[assignment]
        ctx = RenderContext.auto_detect()
        expect_equal(ctx.format, OutputFormat.JSON)
        expect_true(not ctx.color)
        expect_true(not ctx.is_tty)
    finally:
        sys.stdout = original_stdout


def test_render_context_auto_detect_with_override() -> None:
    """Verify auto_detect respects overrides."""
    mock_stdout = StringIO()
    mock_stdout.isatty = lambda: True  # type: ignore[method-assign]
    original_stdout = sys.stdout
    try:
        sys.stdout = mock_stdout  # type: ignore[assignment]
        ctx = RenderContext.auto_detect(
            format_override=OutputFormat.JSON,
            color_override=False,
        )
        expect_equal(ctx.format, OutputFormat.JSON)
        expect_true(not ctx.color)
    finally:
        sys.stdout = original_stdout


def test_render_context_for_testing() -> None:
    """Verify for_testing returns captured streams."""
    ctx, out, err = RenderContext.for_testing()

    expect_equal(ctx.format, OutputFormat.TEXT)
    expect_true(not ctx.color)
    expect_true(not ctx.is_tty)
    expect_is_instance(out, StringIO)
    expect_is_instance(err, StringIO)

    # Verify streams are writable
    ctx.writer.write("test")
    ctx.err_writer.write("error")

    expect_equal(out.getvalue(), "test")
    expect_equal(err.getvalue(), "error")
