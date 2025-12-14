"""Rendering type definitions.

This module defines the core types for CLI output rendering:

- OutputFormat: Canonical output formats
- RenderContext: Context for rendering operations
- JustifyMethod: Text justification options
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import StrEnum
from io import StringIO
from typing import Literal, TextIO


class OutputFormat(StrEnum):
    """Canonical output formats for CLI commands.

    Attributes
    ----------
    TEXT
        Human-readable output, may include colors.
    JSON
        Structured JSON object.
    JSONL
        JSON Lines for streaming.
    """

    TEXT = "text"
    JSON = "json"
    JSONL = "jsonl"


JustifyMethod = Literal["left", "center", "right", "full", "default"]


def _default_stdout() -> TextIO:
    """Return sys.stdout as default writer.

    Returns
    -------
    TextIO
        Standard output stream.
    """
    return sys.stdout


def _default_stderr() -> TextIO:
    """Return sys.stderr as default error writer.

    Returns
    -------
    TextIO
        Standard error stream.
    """
    return sys.stderr


@dataclass(frozen=True)
class RenderContext:
    """Context for rendering operations.

    Determines how output is formatted based on environment and user preference.

    Parameters
    ----------
    format
        Output format preference.
    color
        Whether to use ANSI color codes.
    writer
        Primary output stream (stdout).
    err_writer
        Error/warning stream (stderr).
    is_tty
        Whether output is a terminal (affects defaults).

    Examples
    --------
    >>> ctx = RenderContext.auto_detect()
    >>> ctx.format
    <OutputFormat.TEXT: 'text'>

    >>> ctx, out, err = RenderContext.for_testing()
    >>> ctx.color
    False
    """

    format: OutputFormat
    color: bool
    writer: TextIO = field(default_factory=_default_stdout)
    err_writer: TextIO = field(default_factory=_default_stderr)
    is_tty: bool = field(default=True)

    @classmethod
    def auto_detect(
        cls,
        format_override: OutputFormat | None = None,
        *,
        color_override: bool | None = None,
    ) -> RenderContext:
        """Create context with auto-detection.

        TTY detection determines color default. JSON format disables color.

        Parameters
        ----------
        format_override
            Explicit format override.
        color_override
            Explicit color override.

        Returns
        -------
        RenderContext
            Context with appropriate defaults.
        """
        is_tty = sys.stdout.isatty()
        fmt = format_override or (OutputFormat.TEXT if is_tty else OutputFormat.JSON)
        color = (
            color_override if color_override is not None else (is_tty and fmt == OutputFormat.TEXT)
        )

        return cls(format=fmt, color=color, is_tty=is_tty)

    @classmethod
    def for_testing(cls) -> tuple[RenderContext, StringIO, StringIO]:
        """Create context with captured output for testing.

        Returns
        -------
        tuple[RenderContext, StringIO, StringIO]
            Context, captured stdout, captured stderr.

        Examples
        --------
        >>> ctx, out, err = RenderContext.for_testing()
        >>> ctx.writer.write("test")
        4
        >>> out.getvalue()
        'test'
        """
        out = StringIO()
        err = StringIO()
        return (
            cls(
                format=OutputFormat.TEXT,
                color=False,
                writer=out,
                err_writer=err,
                is_tty=False,
            ),
            out,
            err,
        )


__all__ = [
    "JustifyMethod",
    "OutputFormat",
    "RenderContext",
]
