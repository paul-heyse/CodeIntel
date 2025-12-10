"""Unified output rendering for CLI commands.

This module provides a consistent rendering layer that handles:
- Table formatting with rich
- JSON serialization with consistent conventions
- Error rendering with RFC 9457 Problem Details
- TTY detection and graceful degradation
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from codeintel.cli.core import CliResult
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.cli.errors import ProblemDetail

# Type alias for table column justification
JustifyMethod = Literal["default", "left", "center", "right", "full"]


class RenderMode(Enum):
    """Output rendering mode."""

    AUTO = "auto"
    RICH = "rich"
    PLAIN = "plain"
    JSON = "json"
    JSONL = "jsonl"


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a table column.

    Parameters
    ----------
    key
        Dictionary key to extract from row data.
    header
        Column header text.
    style
        Rich style for the column (e.g., "bold", "cyan").
    justify
        Text justification ("left", "center", "right", "full", "default").
    width
        Fixed column width (None for auto).
    """

    key: str
    header: str
    style: str | None = None
    justify: JustifyMethod = "left"
    width: int | None = None


@dataclass(frozen=True)
class TableSpec:
    """Specification for rendering a table.

    Parameters
    ----------
    columns
        Column specifications.
    title
        Optional table title.
    caption
        Optional table caption.
    show_row_numbers
        Whether to show row numbers.
    """

    columns: tuple[ColumnSpec, ...]
    title: str | None = None
    caption: str | None = None
    show_row_numbers: bool = False


class OutputRenderer(Protocol):
    """Protocol for output rendering implementations."""

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

    def render_object(self, obj: object) -> None:
        """Render a single object (dict, dataclass, etc.).

        Parameters
        ----------
        obj
            Object to render.
        """
        ...

    def render_error(self, error: ProblemDetail) -> None:
        """Render an error with RFC 9457 Problem Details.

        Parameters
        ----------
        error
            Problem detail to render.
        """
        ...

    def render_success(self, message: str) -> None:
        """Render a success message.

        Parameters
        ----------
        message
            Success message text.
        """
        ...

    def render_warning(self, message: str) -> None:
        """Render a warning message.

        Parameters
        ----------
        message
            Warning message text.
        """
        ...


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


def _create_themed_console() -> Console:
    """Create a console with the CodeIntel theme.

    Returns
    -------
    Console
        Configured console instance.
    """
    return Console(theme=CODEINTEL_THEME)


def _render_json_to_stdout(obj: object) -> None:
    """Render object as JSON to stdout.

    Parameters
    ----------
    obj
        Object to serialize.
    """
    data = obj
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()  # type: ignore[union-attr]
    sys.stdout.write(json.dumps(data, indent=2, default=str))
    sys.stdout.write("\n")


@dataclass
class RichRenderer:
    """Rich-based renderer for TTY environments.

    Parameters
    ----------
    console
        Rich console instance.
    output_format
        Output format preference.
    """

    console: Console = field(default_factory=_create_themed_console)
    output_format: OutputFormat = OutputFormat.TEXT

    def render_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render tabular data using rich tables.

        Parameters
        ----------
        rows
            Row data as dictionaries.
        spec
            Table specification.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout([dict(row) for row in rows])
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

        self.console.print(table)

    def render_object(self, obj: object) -> None:
        """Render a single object.

        Parameters
        ----------
        obj
            Object to render.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout(obj)
            return

        data: object
        if hasattr(obj, "to_dict"):
            data = obj.to_dict()  # type: ignore[union-attr]
        elif hasattr(obj, "__dict__"):
            data = obj.__dict__
        else:
            data = obj

        if isinstance(data, dict):
            for key, value in data.items():
                self.console.print(f"[key]{key}:[/key] [value]{value}[/value]")
        else:
            self.console.print(data)

    def render_error(self, error: ProblemDetail) -> None:
        """Render an error with RFC 9457 Problem Details.

        Parameters
        ----------
        error
            Problem detail to render.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout(error.to_dict())
            return

        self.console.print(f"[error]Error:[/error] {error.title}")
        if error.detail:
            self.console.print(f"[muted]{error.detail}[/muted]")
        if error.instance:
            self.console.print(f"[muted]Instance: {error.instance}[/muted]")

    def render_success(self, message: str) -> None:
        """Render a success message.

        Parameters
        ----------
        message
            Success message text.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout({"status": "success", "message": message})
            return
        self.console.print(f"[success]✓[/success] {message}")

    def render_warning(self, message: str) -> None:
        """Render a warning message.

        Parameters
        ----------
        message
            Warning message text.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout({"status": "warning", "message": message})
            return
        self.console.print(f"[warning]⚠[/warning] {message}")


@dataclass
class PlainRenderer:
    """Plain text renderer for non-TTY environments.

    Parameters
    ----------
    output_format
        Output format preference.
    """

    output_format: OutputFormat = OutputFormat.TEXT

    def render_table(
        self,
        rows: Sequence[dict[str, object]],
        spec: TableSpec,
    ) -> None:
        """Render tabular data as plain text.

        Parameters
        ----------
        rows
            Row data as dictionaries.
        spec
            Table specification.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout([dict(row) for row in rows])
            return

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
        sys.stdout.write(header)
        sys.stdout.write("\n")
        sys.stdout.write("-" * len(header))
        sys.stdout.write("\n")

        # Rows
        for row in rows:
            line = " | ".join(
                str(row.get(col.key, "")).ljust(widths[col.key]) for col in spec.columns
            )
            sys.stdout.write(line)
            sys.stdout.write("\n")

    def render_object(self, obj: object) -> None:
        """Render a single object as plain text.

        Parameters
        ----------
        obj
            Object to render.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout(obj)
            return

        data: object
        if hasattr(obj, "to_dict"):
            data = obj.to_dict()  # type: ignore[union-attr]
        elif hasattr(obj, "__dict__"):
            data = obj.__dict__
        else:
            data = obj

        if isinstance(data, dict):
            for key, value in data.items():
                sys.stdout.write(f"{key}: {value}\n")
        else:
            sys.stdout.write(f"{data}\n")

    def render_error(self, error: ProblemDetail) -> None:
        """Render an error as plain text.

        Parameters
        ----------
        error
            Problem detail to render.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout(error.to_dict())
            return

        sys.stderr.write(f"Error: {error.title}\n")
        if error.detail:
            sys.stderr.write(f"{error.detail}\n")

    def render_success(self, message: str) -> None:
        """Render a success message.

        Parameters
        ----------
        message
            Success message text.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout({"status": "success", "message": message})
            return
        sys.stdout.write(f"OK: {message}\n")

    def render_warning(self, message: str) -> None:
        """Render a warning message.

        Parameters
        ----------
        message
            Warning message text.
        """
        if self.output_format == OutputFormat.JSON:
            _render_json_to_stdout({"status": "warning", "message": message})
            return
        sys.stderr.write(f"Warning: {message}\n")


def get_renderer(
    output_format: OutputFormat = OutputFormat.TEXT,
    *,
    force_mode: RenderMode = RenderMode.AUTO,
) -> OutputRenderer:
    """Get an appropriate renderer based on environment.

    Parameters
    ----------
    output_format
        Desired output format.
    force_mode
        Force a specific render mode.

    Returns
    -------
    OutputRenderer
        Configured renderer instance.
    """
    if force_mode == RenderMode.JSON:
        return PlainRenderer(output_format=OutputFormat.JSON)

    if force_mode == RenderMode.PLAIN:
        return PlainRenderer(output_format=output_format)

    if force_mode == RenderMode.RICH:
        return RichRenderer(output_format=output_format)

    # Auto-detect based on TTY
    if sys.stdout.isatty():
        return RichRenderer(output_format=output_format)
    return PlainRenderer(output_format=output_format)


def render_cli_result[T](
    result: CliResult[T],
    renderer: OutputRenderer,
    *,
    table_spec: TableSpec | None = None,
) -> int:
    """Render a CliResult and return exit code.

    Parameters
    ----------
    result
        CLI result to render.
    renderer
        Renderer to use.
    table_spec
        Table spec if result contains tabular data.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).
    """
    if not result.success:
        if result.error:
            renderer.render_error(result.error)
        return 1

    if result.data is None:
        return 0

    # Handle tabular data
    if table_spec and isinstance(result.data, list):
        renderer.render_table(result.data, table_spec)
        return 0

    # Handle objects with to_dict
    renderer.render_object(result.data)
    return 0


# Pre-built table specs for common outputs
OPERATION_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("id", "Operation ID", style="cyan"),
        ColumnSpec("summary", "Summary"),
        ColumnSpec("tags", "Tags", style="muted"),
    ),
    title="Available Operations",
)

DATASET_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("table_key", "Table Key", style="cyan"),
        ColumnSpec("name", "Name"),
        ColumnSpec("description", "Description", style="muted"),
    ),
    title="Available Datasets",
)

BUILD_TARGET_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("name", "Target", style="cyan"),
        ColumnSpec("status", "Status"),
        ColumnSpec("last_run", "Last Run", style="muted"),
    ),
    title="Build Targets",
)


__all__ = [
    "BUILD_TARGET_TABLE_SPEC",
    "CODEINTEL_THEME",
    "DATASET_TABLE_SPEC",
    "OPERATION_TABLE_SPEC",
    "ColumnSpec",
    "OutputRenderer",
    "PlainRenderer",
    "RenderMode",
    "RichRenderer",
    "TableSpec",
    "get_renderer",
    "render_cli_result",
]
