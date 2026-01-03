"""Unified rendering service for CLI output.

This module provides the UnifiedRenderer which consolidates all rendering logic:

- Format negotiation (TEXT/JSON/JSONL)
- TTY detection and graceful degradation
- Error rendering with RFC 9457 Problem Details
- Warning and metadata handling

Factory Functions
-----------------
- ``get_renderer``: Create renderer with auto-detection
- ``render_cli_result``: Convenience function to render results

Classes
-------
- ``UnifiedRenderer``: Main renderer implementation
- ``RenderingService``: Protocol for renderer interface
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol, TextIO, TypeVar

from rich.console import Console
from rich.theme import Theme

from codeintel.cli.core.result_types import TabularResult
from codeintel.cli.rendering.types import OutputFormat, RenderContext
from codeintel.core.columnar.ipc import write_ipc_stream
from codeintel.core.columnar.stream import ColumnarStream
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.query_results import iter_records_from_arrow_reader, records_from_arrow_reader
from codeintel.core.serialization.msgspec import (
    encode_json_line_text,
    encode_json_text,
    to_builtins,
)

if TYPE_CHECKING:
    from codeintel.cli.core import CliResult
    from codeintel.cli.errors import ProblemDetail

T = TypeVar("T")


# Default theme for CodeIntel CLI
CODEINTEL_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "heading": "bold magenta",
        "key": "bold blue",
        "value": "white",
        "muted": "dim",
    }
)

# HTTP status code threshold for internal server errors (5xx)
_INTERNAL_ERROR_STATUS_THRESHOLD = 500


class RenderingService(Protocol):
    """Protocol for CLI output rendering.

    All handlers delegate output to this protocol. Implementations handle
    format negotiation, TTY detection, and consistent formatting.
    """

    def render_result(self, result: CliResult[T]) -> int:
        """Render a CLI result and return exit code.

        Parameters
        ----------
        result
            CLI result to render.

        Returns
        -------
        int
            Exit code: 0 for success, non-zero for failure.
        """
        ...

    def render_error(self, error: ProblemDetail) -> None:
        """Render error with RFC 9457 Problem Details.

        Parameters
        ----------
        error
            Problem detail to render.
        """
        ...

    def render_message(self, message: str, *, level: str = "info") -> None:
        """Render a simple message.

        Parameters
        ----------
        message
            Message text.
        level
            Message level: "info", "success", "warning", "error".
        """
        ...


class UnifiedRenderer:
    """Single implementation of RenderingService.

    Consolidates:

    - RichRenderer from cli_render.py
    - PlainRenderer from cli_render.py
    - CliResult.render() logic from results.py

    Parameters
    ----------
    ctx
        Render context with format, color, and stream settings.

    Examples
    --------
    >>> ctx = RenderContext.auto_detect()  # doctest: +SKIP
    >>> renderer = UnifiedRenderer(ctx)  # doctest: +SKIP
    >>> renderer.render_message("Operation complete", level="success")  # doctest: +SKIP
    """

    def __init__(self, ctx: RenderContext) -> None:
        """Initialize renderer with context."""
        self._ctx = ctx
        self._console: Console | None = (
            Console(file=ctx.writer, theme=CODEINTEL_THEME) if ctx.color else None
        )
        self._err_console: Console | None = (
            Console(file=ctx.err_writer, theme=CODEINTEL_THEME) if ctx.color else None
        )

    @property
    def context(self) -> RenderContext:
        """Get the render context.

        Returns
        -------
        RenderContext
            Current render context.
        """
        return self._ctx

    def render_result(self, result: CliResult[T]) -> int:
        """Render a CLI result and return exit code.

        Handle warning emission to stderr, error rendering with Problem Details,
        data rendering in appropriate format, and metadata inclusion in JSON output.

        Parameters
        ----------
        result
            CLI result to render.

        Returns
        -------
        int
            Exit code: 0 for success, non-zero for failure.
        """
        # 1. Emit warnings to stderr
        for warning in result.warnings:
            self._emit_warning(warning)

        # 2. Handle failure case
        if not result.success:
            if result.error:
                self.render_error(result.error)
            return self._exit_code_for_error(result.error)

        # 3. Handle success case
        if result.data is not None:
            self._render_data(result.data, result.metadata)

        return 0

    def render_error(self, error: ProblemDetail) -> None:
        """Render error with RFC 9457 Problem Details.

        Always writes to err_writer (stderr).

        Parameters
        ----------
        error
            Problem detail to render.
        """
        if self._ctx.format in {OutputFormat.JSON, OutputFormat.JSONL, OutputFormat.ARROW_IPC}:
            self._ctx.err_writer.write(encode_json_text(error.to_dict(), indent=2, newline=True))
        elif self._err_console is not None:
            self._err_console.print(f"[error]Error:[/error] {error.title}")
            if error.detail:
                self._err_console.print(f"[muted]{error.detail}[/muted]")
        else:
            self._ctx.err_writer.write(f"Error: {error.title}\n")
            if error.detail:
                self._ctx.err_writer.write(f"{error.detail}\n")

    def render_message(self, message: str, *, level: str = "info") -> None:
        """Render a simple message.

        Parameters
        ----------
        message
            Message text.
        level
            Message level: "info", "success", "warning", "error".
        """
        if self._ctx.format == OutputFormat.JSON:
            self._write_json({"status": level, "message": message})
            return
        if self._ctx.format == OutputFormat.ARROW_IPC:
            self._ctx.err_writer.write(
                encode_json_text({"status": level, "message": message}, indent=2, newline=True)
            )
            return

        if self._console is not None:
            style_map = {
                "info": "[info]i[/info]",
                "success": "[success]v[/success]",
                "warning": "[warning]![/warning]",
                "error": "[error]x[/error]",
            }
            prefix = style_map.get(level, "")
            self._console.print(f"{prefix} {message}")
        else:
            prefix_map = {
                "info": "INFO:",
                "success": "OK:",
                "warning": "WARNING:",
                "error": "ERROR:",
            }
            prefix = prefix_map.get(level, "")
            writer = self._ctx.err_writer if level in {"warning", "error"} else self._ctx.writer
            writer.write(f"{prefix} {message}\n")

    def emit_progress(
        self,
        current: int,
        total: int,
        message: str | None = None,
    ) -> None:
        """Emit progress update.

        In TEXT format: No-op (use Rich progress bar separately)
        In JSONL format: Emits progress JSON object
        In JSON format: No-op (batch result includes progress)

        Parameters
        ----------
        current
            Current progress count.
        total
            Total count.
        message
            Optional progress message.
        """
        if self._ctx.format == OutputFormat.JSONL:
            progress_obj: dict[str, object] = {
                "type": "progress",
                "current": current,
                "total": total,
            }
            if message:
                progress_obj["message"] = message
            self._write_jsonl(progress_obj)

    # --- Private Methods ---

    def _emit_warning(self, warning: str) -> None:
        """Emit a warning to stderr."""
        if self._ctx.format == OutputFormat.JSON:
            # Warnings will be included in result JSON
            return
        if self._ctx.format == OutputFormat.ARROW_IPC:
            return
        if self._err_console is not None:
            self._err_console.print(f"[warning]Warning:[/warning] {warning}")
        else:
            self._ctx.err_writer.write(f"Warning: {warning}\n")

    def _render_data(self, data: object, metadata: dict[str, object]) -> None:
        """Render data payload."""
        if isinstance(data, TabularResult):
            self._render_stream(data.stream, metadata=data.metadata or metadata)
            return
        if isinstance(data, ColumnarStream):
            self._render_stream(data, metadata=metadata)
            return
        if self._ctx.format == OutputFormat.JSON:
            self._render_json_payload(data, metadata=metadata)
            return
        self._render_text_payload(data)

    def _render_json_payload(self, data: object, *, metadata: dict[str, object]) -> None:
        output: dict[str, object] = {"data": self._serialize(data)}
        if metadata:
            output["metadata"] = metadata
        self._write_json(output)

    def _render_text_payload(self, data: object) -> None:
        if isinstance(data, str):
            self._ctx.writer.write(data)
            if not data.endswith("\n"):
                self._ctx.writer.write("\n")
            return
        if isinstance(data, dict):
            self._render_dict(data)
            return
        if isinstance(data, list):
            for item in data:
                self._ctx.writer.write(f"{item}\n")
            return
        # Try to convert objects with to_dict to dict for text rendering
        serialized = self._serialize(data)
        if isinstance(serialized, dict):
            self._render_dict(serialized)
        else:
            self._ctx.writer.write(f"{data}\n")

    def _render_stream(self, stream: ColumnarStream, *, metadata: dict[str, object]) -> None:
        reader = stream.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
        if self._ctx.format == OutputFormat.ARROW_IPC:
            binary_writer = getattr(self._ctx.writer, "buffer", None)
            if binary_writer is None:
                msg = "Arrow IPC output requires a binary writer"
                raise RuntimeError(msg)
            write_ipc_stream(reader, binary_writer)
            return
        if self._ctx.format == OutputFormat.JSONL:
            for record in iter_records_from_arrow_reader(reader):
                self._write_jsonl(record)
            return
        records = records_from_arrow_reader(reader)
        if self._ctx.format == OutputFormat.JSON:
            output: dict[str, object] = {"data": records}
            if metadata:
                output["metadata"] = metadata
            self._write_json(output)
            return
        for record in records:
            self._ctx.writer.write(f"{record}\n")

    def _render_dict(self, data: dict[str, object]) -> None:
        """Render a dictionary."""
        if self._console is not None:
            for key, value in data.items():
                self._console.print(f"[key]{key}:[/key] [value]{value}[/value]")
        else:
            for key, value in data.items():
                self._ctx.writer.write(f"{key}: {value}\n")

    def _write_json(self, obj: object) -> None:
        """Write JSON to stdout with pretty formatting."""
        self._ctx.writer.write(encode_json_text(obj, indent=2, newline=True))

    def _write_jsonl(self, obj: object) -> None:
        """Write JSON to stdout as single line (for JSONL)."""
        self._ctx.writer.write(encode_json_line_text(obj))

    @staticmethod
    def _serialize(data: object) -> object:
        """Serialize data for JSON output.

        Returns
        -------
        object
            Serialized data.
        """
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        try:
            return to_builtins(data)
        except TypeError:
            if hasattr(data, "__dict__") and not isinstance(data, type):
                return data.__dict__
            return data

    @staticmethod
    def _exit_code_for_error(error: ProblemDetail | None) -> int:
        """Determine exit code from error.

        Returns
        -------
        int
            Exit code: 1 for user error, 2 for internal error.
        """
        if error is None:
            return 1
        if error.status >= _INTERNAL_ERROR_STATUS_THRESHOLD:
            return 2  # Internal error
        return 1  # User error


def get_renderer(
    output_format: OutputFormat = OutputFormat.TEXT,
    *,
    color: bool | None = None,
    writer: TextIO | None = None,
    err_writer: TextIO | None = None,
) -> UnifiedRenderer:
    """Get a renderer for the specified output format.

    Factory function that creates UnifiedRenderer instances with appropriate
    settings based on output format, environment, and TTY detection.

    Parameters
    ----------
    output_format
        Desired output format (TEXT, JSON, or JSONL).
    color
        Override color detection. If None, auto-detect based on TTY.
    writer
        Output stream (defaults to sys.stdout).
    err_writer
        Error stream (defaults to sys.stderr).

    Returns
    -------
    UnifiedRenderer
        Configured renderer instance.

    Examples
    --------
    >>> renderer = get_renderer(OutputFormat.JSON)
    >>> renderer.context.format
    <OutputFormat.JSON: 'json'>

    >>> renderer = get_renderer(color=False)
    >>> renderer.context.color
    False
    """
    # If custom streams are provided, construct context directly
    if writer is not None or err_writer is not None:
        is_tty = (writer or sys.stdout).isatty()
        use_color = color if color is not None else (is_tty and output_format == OutputFormat.TEXT)
        ctx = RenderContext(
            format=output_format,
            color=use_color,
            writer=writer or sys.stdout,
            err_writer=err_writer or sys.stderr,
            is_tty=is_tty,
        )
    else:
        ctx = RenderContext.auto_detect(
            format_override=output_format,
            color_override=color,
        )
    return UnifiedRenderer(ctx)


def render_cli_result[T](
    result: CliResult[T],
    renderer: UnifiedRenderer | None = None,
    *,
    output_format: OutputFormat = OutputFormat.TEXT,
) -> int:
    """Render a CliResult and return exit code.

    Convenience function that creates a renderer if not provided and renders
    the result appropriately. Supports optional table rendering for list data.

    Parameters
    ----------
    result
        CLI result to render.
    renderer
        Optional renderer. If None, creates one based on output_format.
    output_format
        Output format (used if renderer is None).

    Returns
    -------
    int
        Exit code: 0 for success, non-zero for failure.

    Examples
    --------
    >>> result = CliResult.ok({"status": "done"})
    >>> exit_code = render_cli_result(result)
    >>> exit_code
    0
    """
    if renderer is None:
        renderer = get_renderer(output_format)

    return renderer.render_result(result)


__all__ = [
    "CODEINTEL_THEME",
    "RenderingService",
    "UnifiedRenderer",
    "get_renderer",
    "render_cli_result",
]
