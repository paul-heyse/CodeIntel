"""Unified rendering service for CLI output.

This module provides the UnifiedRenderer which consolidates all rendering logic:

- Format negotiation (TEXT/JSON/JSONL)
- TTY detection and graceful degradation
- Table rendering with Rich or plain text
- Error rendering with RFC 9457 Problem Details
- Warning and metadata handling
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, TypeVar

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from codeintel.cli.rendering.table import TableSpec
from codeintel.cli.rendering.types import OutputFormat, RenderContext

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def render_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render tabular data.

        Parameters
        ----------
        rows
            Row data as dictionaries.
        spec
            Table specification.
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
    - StreamingRenderer from pipelines.py
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

    def render_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render tabular data with format negotiation.

        Parameters
        ----------
        rows
            Row data as dictionaries.
        spec
            Table specification.
        """
        if not rows:
            self.render_message(spec.empty_message, level="info")
            return

        if self._ctx.format == OutputFormat.JSON:
            self._write_json([dict(row) for row in rows])
        elif self._ctx.format == OutputFormat.JSONL:
            for row in rows:
                self._write_jsonl(dict(row))
        elif self._console is not None:
            self._render_rich_table(rows, spec)
        else:
            self._render_plain_table(rows, spec)

    def render_error(self, error: ProblemDetail) -> None:
        """Render error with RFC 9457 Problem Details.

        Always writes to err_writer (stderr).

        Parameters
        ----------
        error
            Problem detail to render.
        """
        if self._ctx.format in {OutputFormat.JSON, OutputFormat.JSONL}:
            self._ctx.err_writer.write(json.dumps(error.to_dict(), indent=2))
            self._ctx.err_writer.write("\n")
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
        if self._err_console is not None:
            self._err_console.print(f"[warning]Warning:[/warning] {warning}")
        else:
            self._ctx.err_writer.write(f"Warning: {warning}\n")

    def _render_data(self, data: object, metadata: dict[str, object]) -> None:
        """Render data payload."""
        if self._ctx.format == OutputFormat.JSON:
            output: dict[str, object] = {"data": self._serialize(data)}
            if metadata:
                output["metadata"] = metadata
            self._write_json(output)
        elif isinstance(data, str):
            self._ctx.writer.write(data)
            if not data.endswith("\n"):
                self._ctx.writer.write("\n")
        elif isinstance(data, dict):
            self._render_dict(data)
        elif isinstance(data, list):
            for item in data:
                self._ctx.writer.write(f"{item}\n")
        else:
            # Try to convert objects with to_dict to dict for text rendering
            serialized = self._serialize(data)
            if isinstance(serialized, dict):
                self._render_dict(serialized)
            else:
                self._ctx.writer.write(f"{data}\n")

    def _render_dict(self, data: dict[str, object]) -> None:
        """Render a dictionary."""
        if self._console is not None:
            for key, value in data.items():
                self._console.print(f"[key]{key}:[/key] [value]{value}[/value]")
        else:
            for key, value in data.items():
                self._ctx.writer.write(f"{key}: {value}\n")

    def _render_rich_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render table using Rich."""
        if self._console is None:
            self._render_plain_table(rows, spec)
            return

        table = Table(
            title=spec.title,
            caption=spec.caption,
            show_header=True,
            header_style="bold",
        )

        if spec.show_row_numbers:
            table.add_column("#", style="muted", width=4)

        for col in spec.columns:
            table.add_column(
                col.header,
                style=col.style,
                justify=col.justify,
                width=col.width,
            )

        for i, row in enumerate(rows, 1):
            values = [str(row.get(col.key, "")) for col in spec.columns]
            if spec.show_row_numbers:
                values = [str(i), *values]
            table.add_row(*values)

        self._console.print(table)

    def _render_plain_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render table as plain text."""
        # Calculate column widths
        widths = {
            col.key: max(
                len(col.header),
                max((len(str(row.get(col.key, ""))) for row in rows), default=0),
            )
            for col in spec.columns
        }

        # Header
        header = " | ".join(col.header.ljust(widths[col.key]) for col in spec.columns)
        self._ctx.writer.write(header + "\n")
        self._ctx.writer.write("-" * len(header) + "\n")

        # Rows
        for row in rows:
            line = " | ".join(
                str(row.get(col.key, "")).ljust(widths[col.key]) for col in spec.columns
            )
            self._ctx.writer.write(line + "\n")

    def _write_json(self, obj: object) -> None:
        """Write JSON to stdout with pretty formatting."""
        self._ctx.writer.write(json.dumps(obj, indent=2, default=str))
        self._ctx.writer.write("\n")

    def _write_jsonl(self, obj: object) -> None:
        """Write JSON to stdout as single line (for JSONL)."""
        self._ctx.writer.write(json.dumps(obj, default=str))
        self._ctx.writer.write("\n")

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


__all__ = [
    "CODEINTEL_THEME",
    "RenderingService",
    "UnifiedRenderer",
]
