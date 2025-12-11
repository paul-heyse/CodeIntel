"""Tests for rendering types."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, TextIO

from codeintel.cli.rendering import OutputFormat, RenderContext
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class _TtyStringIO(StringIO):
    """StringIO with a configurable TTY flag."""

    def __init__(self, *, is_tty: bool) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        """
        Return whether this stream should behave like a TTY.

        Returns
        -------
        bool
            True if configured as a TTY, otherwise False.
        """
        return self._is_tty


def _swap_stdout(fake_stdout: TextIO) -> AbstractContextManager[TextIO]:
    """
    Temporarily replace sys.stdout for RenderContext detection.

    Returns
    -------
    ContextManager[TextIO]
        Context manager that restores stdout after use.
    """
    return redirect_stdout(fake_stdout)


def test_output_format_values() -> None:
    """Validate OutputFormat has expected values."""
    expect_equal(OutputFormat.TEXT.value, "text")
    expect_equal(OutputFormat.JSON.value, "json")
    expect_equal(OutputFormat.JSONL.value, "jsonl")


def test_render_context_auto_detect_tty() -> None:
    """Verify auto_detect uses TEXT format for TTY."""
    mock_stdout = _TtyStringIO(is_tty=True)
    with _swap_stdout(mock_stdout):
        ctx = RenderContext.auto_detect()
        expect_equal(ctx.format, OutputFormat.TEXT)
        expect_true(ctx.color)
        expect_true(ctx.is_tty)


def test_render_context_auto_detect_non_tty() -> None:
    """Verify auto_detect uses JSON format for non-TTY."""
    mock_stdout = _TtyStringIO(is_tty=False)
    with _swap_stdout(mock_stdout):
        ctx = RenderContext.auto_detect()
        expect_equal(ctx.format, OutputFormat.JSON)
        expect_true(not ctx.color)
        expect_true(not ctx.is_tty)


def test_render_context_auto_detect_with_override() -> None:
    """Verify auto_detect respects overrides."""
    mock_stdout = _TtyStringIO(is_tty=True)
    with _swap_stdout(mock_stdout):
        ctx = RenderContext.auto_detect(
            format_override=OutputFormat.JSON,
            color_override=False,
        )
        expect_equal(ctx.format, OutputFormat.JSON)
        expect_true(not ctx.color)


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
