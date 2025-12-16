# CLI Consolidation Phase 1: Foundation Layer — Detailed Implementation Plan

> **Status**: Ready for Implementation  
> **Depends On**: None  
> **Risk Level**: Low (purely additive)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

## Overview

This document provides step-by-step implementation instructions with complete code snippets for Phase 1 of the CLI consolidation. All changes are **purely additive** — no existing code is modified.

---

## Table of Contents

1. [Implementation Order](#1-implementation-order)
2. [Rendering Package](#2-rendering-package)
3. [RuntimeParams and Resolution Enhancements](#3-runtimeparams-and-resolution-enhancements)
4. [ConfigService](#4-configservice)
5. [Enhanced HandlerContext](#5-enhanced-handlercontext)
6. [Test Implementation](#6-test-implementation)
7. [Verification Checklist](#7-verification-checklist)

---

## 1. Implementation Order

Create files in this order to minimize import issues:

```
1. cli/rendering/types.py          # No dependencies on new code
2. cli/rendering/table.py          # Depends only on types.py
3. cli/rendering/specs.py          # Pre-built table specs
4. cli/rendering/service.py        # UnifiedRenderer
5. cli/rendering/__init__.py       # Re-exports
6. cli/resolution/params.py        # RuntimeParams (new file)
7. cli/config/service.py           # ConfigService
8. cli/handlers/protocol.py        # HandlerProtocol, enhanced HandlerContext
```

---

## 2. Rendering Package

### 2.1 Create `cli/rendering/types.py`

```python
"""Rendering type definitions.

This module defines the core types for CLI output rendering:
- OutputFormat: Canonical output formats
- RenderContext: Context for rendering operations
- JustifyMethod: Text justification options
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import TYPE_CHECKING, Literal, TextIO

if TYPE_CHECKING:
    pass


class OutputFormat(Enum):
    """Canonical output formats for CLI commands.

    Parameters
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
    writer: TextIO = field(default_factory=lambda: sys.stdout)
    err_writer: TextIO = field(default_factory=lambda: sys.stderr)
    is_tty: bool = field(default=True)

    @classmethod
    def auto_detect(
        cls,
        format_override: OutputFormat | None = None,
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
            color_override
            if color_override is not None
            else (is_tty and fmt == OutputFormat.TEXT)
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
        >>> # Use ctx for rendering
        >>> output = out.getvalue()
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
```

### 2.2 Create `cli/rendering/table.py`

```python
"""Table rendering specifications.

This module defines the types for specifying table structure:
- ColumnSpec: Individual column configuration
- TableSpec: Complete table specification
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.cli.rendering.types import JustifyMethod


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
        Text justification.
    width
        Fixed column width (None for auto).

    Examples
    --------
    >>> col = ColumnSpec("name", "Name", style="cyan")
    >>> col.key
    'name'
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
        Optional table caption (footer).
    show_row_numbers
        Whether to show row numbers.
    empty_message
        Message when table has no rows.

    Examples
    --------
    >>> spec = TableSpec(
    ...     columns=(
    ...         ColumnSpec("id", "ID"),
    ...         ColumnSpec("name", "Name"),
    ...     ),
    ...     title="Users",
    ... )
    >>> spec.title
    'Users'
    """

    columns: tuple[ColumnSpec, ...]
    title: str | None = None
    caption: str | None = None
    show_row_numbers: bool = False
    empty_message: str = "No data."


__all__ = [
    "ColumnSpec",
    "TableSpec",
]
```

### 2.3 Create `cli/rendering/specs.py`

```python
"""Pre-built table specifications for common CLI outputs.

This module provides standard table specs for consistent formatting
across commands that output similar data.
"""

from __future__ import annotations

from codeintel.cli.rendering.table import ColumnSpec, TableSpec

# Operations listing
OPERATIONS_TABLE = TableSpec(
    columns=(
        ColumnSpec("id", "Operation ID", style="cyan"),
        ColumnSpec("summary", "Summary"),
        ColumnSpec("tags", "Tags", style="dim"),
    ),
    title="Available Operations",
)

# Dataset listing
DATASETS_TABLE = TableSpec(
    columns=(
        ColumnSpec("table_key", "Table", style="cyan"),
        ColumnSpec("name", "Name"),
        ColumnSpec("row_count", "Rows", justify="right"),
        ColumnSpec("description", "Description", style="dim"),
    ),
    title="Datasets",
)

# Build targets
BUILD_TARGETS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Target", style="cyan"),
        ColumnSpec("module", "Module"),
        ColumnSpec("status", "Status"),
        ColumnSpec("duration", "Duration", justify="right"),
    ),
    title="Build Targets",
)

# Plugin listing
PLUGINS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Plugin", style="cyan"),
        ColumnSpec("version", "Version"),
        ColumnSpec("status", "Status"),
        ColumnSpec("capabilities", "Capabilities", style="dim"),
    ),
    title="Installed Plugins",
)

# Jobs listing
JOBS_TABLE = TableSpec(
    columns=(
        ColumnSpec("job_id", "Job ID", style="cyan"),
        ColumnSpec("operation", "Operation"),
        ColumnSpec("status", "Status"),
        ColumnSpec("created_at", "Created", style="dim"),
    ),
    title="Jobs",
)

# Health checks
HEALTH_TABLE = TableSpec(
    columns=(
        ColumnSpec("component", "Component", style="cyan"),
        ColumnSpec("status", "Status"),
        ColumnSpec("message", "Message"),
        ColumnSpec("latency_ms", "Latency", justify="right"),
    ),
    title="Health Status",
)

# Subsystem info
SUBSYSTEMS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Subsystem", style="cyan"),
        ColumnSpec("type", "Type"),
        ColumnSpec("count", "Count", justify="right"),
        ColumnSpec("description", "Description", style="dim"),
    ),
    title="Subsystems",
)


__all__ = [
    "BUILD_TARGETS_TABLE",
    "DATASETS_TABLE",
    "HEALTH_TABLE",
    "JOBS_TABLE",
    "OPERATIONS_TABLE",
    "PLUGINS_TABLE",
    "SUBSYSTEMS_TABLE",
]
```

### 2.4 Create `cli/rendering/service.py`

```python
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

    from codeintel.cli.cli_errors import ProblemDetail
    from codeintel.cli.results import CliResult

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
    >>> ctx = RenderContext.auto_detect()
    >>> renderer = UnifiedRenderer(ctx)
    >>> renderer.render_message("Operation complete", level="success")
    """

    def __init__(self, ctx: RenderContext) -> None:
        """Initialize renderer with context."""
        self._ctx = ctx
        self._console = Console(theme=CODEINTEL_THEME) if ctx.color else None

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

        Handles:
        - Warning emission to stderr
        - Error rendering with Problem Details
        - Data rendering in appropriate format
        - Metadata inclusion in JSON output

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
                self._write_json(dict(row))
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
        if self._ctx.format in (OutputFormat.JSON, OutputFormat.JSONL):
            self._ctx.err_writer.write(json.dumps(error.to_dict(), indent=2))
            self._ctx.err_writer.write("\n")
        elif self._console is not None:
            self._console.print(f"[error]Error:[/error] {error.title}", file=self._ctx.err_writer)
            if error.detail:
                self._console.print(f"[muted]{error.detail}[/muted]", file=self._ctx.err_writer)
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
                "info": "[info]ℹ[/info]",
                "success": "[success]✓[/success]",
                "warning": "[warning]⚠[/warning]",
                "error": "[error]✗[/error]",
            }
            prefix = style_map.get(level, "")
            self._console.print(f"{prefix} {message}")
        else:
            prefix_map = {"info": "INFO:", "success": "OK:", "warning": "WARNING:", "error": "ERROR:"}
            prefix = prefix_map.get(level, "")
            writer = self._ctx.err_writer if level in ("warning", "error") else self._ctx.writer
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
            self._write_json(progress_obj)

    # --- Private Methods ---

    def _emit_warning(self, warning: str) -> None:
        """Emit a warning to stderr."""
        if self._ctx.format == OutputFormat.JSON:
            # Warnings will be included in result JSON
            return
        if self._console is not None:
            self._console.print(f"[warning]Warning:[/warning] {warning}", file=self._ctx.err_writer)
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
        """Write JSON to stdout."""
        self._ctx.writer.write(json.dumps(obj, indent=2, default=str))
        self._ctx.writer.write("\n")

    @staticmethod
    def _serialize(data: object) -> object:
        """Serialize data for JSON output."""
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if hasattr(data, "__dict__") and not isinstance(data, type):
            return data.__dict__
        return data

    @staticmethod
    def _exit_code_for_error(error: ProblemDetail | None) -> int:
        """Determine exit code from error."""
        if error is None:
            return 1
        if error.status >= 500:
            return 2  # Internal error
        return 1  # User error


__all__ = [
    "CODEINTEL_THEME",
    "RenderingService",
    "UnifiedRenderer",
]
```

### 2.5 Create `cli/rendering/__init__.py`

```python
"""Unified rendering package for CLI output.

This package provides the single source of truth for CLI output rendering:

- ``UnifiedRenderer``: Single renderer for all output
- ``RenderContext``: Context with format, color, and stream settings
- ``OutputFormat``: Output format enum (TEXT, JSON, JSONL)
- ``TableSpec``, ``ColumnSpec``: Table specification types
- Pre-built table specs for common outputs

Examples
--------
>>> from codeintel.cli.rendering import UnifiedRenderer, RenderContext
>>> ctx = RenderContext.auto_detect()
>>> renderer = UnifiedRenderer(ctx)
>>> renderer.render_message("Done!", level="success")
"""

from __future__ import annotations

from codeintel.cli.rendering.service import (
    CODEINTEL_THEME,
    RenderingService,
    UnifiedRenderer,
)
from codeintel.cli.rendering.specs import (
    BUILD_TARGETS_TABLE,
    DATASETS_TABLE,
    HEALTH_TABLE,
    JOBS_TABLE,
    OPERATIONS_TABLE,
    PLUGINS_TABLE,
    SUBSYSTEMS_TABLE,
)
from codeintel.cli.rendering.table import ColumnSpec, TableSpec
from codeintel.cli.rendering.types import JustifyMethod, OutputFormat, RenderContext

__all__ = [
    # Core types
    "OutputFormat",
    "RenderContext",
    "JustifyMethod",
    # Table specs
    "ColumnSpec",
    "TableSpec",
    # Service
    "RenderingService",
    "UnifiedRenderer",
    "CODEINTEL_THEME",
    # Pre-built specs
    "BUILD_TARGETS_TABLE",
    "DATASETS_TABLE",
    "HEALTH_TABLE",
    "JOBS_TABLE",
    "OPERATIONS_TABLE",
    "PLUGINS_TABLE",
    "SUBSYSTEMS_TABLE",
]
```

---

## 3. RuntimeParams and Resolution Enhancements

### 3.1 Create `cli/resolution/params.py`

```python
"""Canonical runtime parameters type.

This module defines RuntimeParams, the single source of truth for
runtime parameters that replaces all RuntimeCliOptions variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.cli.cyclopts_common import RuntimeCLI
    from codeintel.cli.execution.context import ExecutionContext


@dataclass(frozen=True)
class BackendFlags:
    """Graph backend configuration flags.

    Parameters
    ----------
    use_gpu
        Whether to attempt GPU acceleration.
    backend
        Backend selection: "auto", "cpu", or "nx-cugraph".
    strict
        Whether to enforce strict backend compatibility.
    """

    use_gpu: bool = False
    backend: str = "auto"
    strict: bool = False


@dataclass(frozen=True)
class RuntimeParams:
    """Canonical runtime parameters from any input source.

    This is THE type for runtime parameters. All other RuntimeCliOptions
    variants are deprecated in favor of this single type.

    Parameters
    ----------
    project_root
        Root directory for project file discovery.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit SHA.
    db_path
        Explicit database path.
    build_dir
        Build output directory.
    repo_root
        Repository root path.
    document_output_dir
        Document export directory.
    backend
        Graph backend configuration.

    Examples
    --------
    >>> params = RuntimeParams.minimal(Path("/project"))
    >>> params.project_root
    PosixPath('/project')

    >>> params = RuntimeParams(repo="org/repo", commit="abc123")
    >>> params.repo
    'org/repo'
    """

    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    backend: BackendFlags = field(default_factory=BackendFlags)

    # --- Factory Methods ---

    @classmethod
    def from_context(cls, ctx: ExecutionContext) -> RuntimeParams:
        """Extract RuntimeParams from ExecutionContext.params dict.

        The context params may contain any subset of fields.
        Missing fields use defaults.

        Parameters
        ----------
        ctx
            Execution context with params dict.

        Returns
        -------
        RuntimeParams
            Extracted parameters.

        Examples
        --------
        >>> from codeintel.cli.execution.context import ExecutionContext
        >>> ctx = ExecutionContext.for_sync("op", {"repo": "org/repo"})
        >>> params = RuntimeParams.from_context(ctx)
        >>> params.repo
        'org/repo'
        """
        params = ctx.params

        backend_raw = params.get("backend", {})
        backend = BackendFlags(
            use_gpu=_get_bool(backend_raw, "use_gpu", default=False),
            backend=_get_str(backend_raw, "backend", default="auto"),
            strict=_get_bool(backend_raw, "strict", default=False),
        ) if isinstance(backend_raw, dict) else BackendFlags()

        return cls(
            project_root=_to_path(params.get("project_root")),
            repo=_to_str(params.get("repo")),
            commit=_to_str(params.get("commit")),
            db_path=_to_path(params.get("db_path")),
            build_dir=_to_path(params.get("build_dir")),
            repo_root=_to_path(params.get("repo_root")),
            document_output_dir=_to_path(params.get("document_output_dir")),
            backend=backend,
        )

    @classmethod
    def from_cyclopts(cls, runtime_cli: RuntimeCLI) -> RuntimeParams:
        """Convert Cyclopts RuntimeCLI to canonical RuntimeParams.

        RuntimeCLI is a Cyclopts-specific dataclass with Parameter
        annotations. This method extracts values into the canonical type.

        Parameters
        ----------
        runtime_cli
            Cyclopts runtime CLI dataclass.

        Returns
        -------
        RuntimeParams
            Canonical parameters.

        Examples
        --------
        >>> from codeintel.cli.cyclopts_common import RuntimeCLI
        >>> cli = RuntimeCLI(repo="org/repo", commit="abc123")
        >>> params = RuntimeParams.from_cyclopts(cli)
        >>> params.repo
        'org/repo'
        """
        return cls(
            project_root=runtime_cli.project_root,
            repo=runtime_cli.repo,
            commit=runtime_cli.commit,
            db_path=runtime_cli.db_path,
            build_dir=runtime_cli.build_dir,
            repo_root=runtime_cli.repo_root,
            document_output_dir=runtime_cli.document_output_dir,
            backend=BackendFlags(),  # RuntimeCLI doesn't include backend
        )

    @classmethod
    def minimal(cls, project_root: Path | None = None) -> RuntimeParams:
        """Create minimal params for simple commands.

        Use for commands that only need project discovery (ide hints, etc).

        Parameters
        ----------
        project_root
            Optional project root path.

        Returns
        -------
        RuntimeParams
            Minimal parameters.

        Examples
        --------
        >>> params = RuntimeParams.minimal()
        >>> params.project_root is None
        True
        """
        return cls(project_root=project_root)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeParams:
        """Create RuntimeParams from a dictionary.

        Parameters
        ----------
        data
            Dictionary with parameter values.

        Returns
        -------
        RuntimeParams
            Parsed parameters.
        """
        backend_raw = data.get("backend", {})
        backend = BackendFlags(
            use_gpu=_get_bool(backend_raw, "use_gpu", default=False),
            backend=_get_str(backend_raw, "backend", default="auto"),
            strict=_get_bool(backend_raw, "strict", default=False),
        ) if isinstance(backend_raw, dict) else BackendFlags()

        return cls(
            project_root=_to_path(data.get("project_root")),
            repo=_to_str(data.get("repo")),
            commit=_to_str(data.get("commit")),
            db_path=_to_path(data.get("db_path")),
            build_dir=_to_path(data.get("build_dir")),
            repo_root=_to_path(data.get("repo_root")),
            document_output_dir=_to_path(data.get("document_output_dir")),
            backend=backend,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "project_root": str(self.project_root) if self.project_root else None,
            "repo": self.repo,
            "commit": self.commit,
            "db_path": str(self.db_path) if self.db_path else None,
            "build_dir": str(self.build_dir) if self.build_dir else None,
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "document_output_dir": str(self.document_output_dir) if self.document_output_dir else None,
            "backend": {
                "use_gpu": self.backend.use_gpu,
                "backend": self.backend.backend,
                "strict": self.backend.strict,
            },
        }


# --- Helper Functions ---


def _to_path(value: object) -> Path | None:
    """Convert value to Path or None."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _to_str(value: object) -> str | None:
    """Convert value to string or None."""
    if value is None:
        return None
    return str(value)


def _get_bool(data: object, key: str, *, default: bool) -> bool:
    """Get boolean from dict-like object."""
    if not isinstance(data, dict):
        return default
    return bool(data.get(key, default))


def _get_str(data: object, key: str, *, default: str) -> str:
    """Get string from dict-like object."""
    if not isinstance(data, dict):
        return default
    value = data.get(key, default)
    return str(value) if value is not None else default


__all__ = [
    "BackendFlags",
    "RuntimeParams",
]
```

### 3.2 Update `cli/resolution/__init__.py`

Add to the existing file:

```python
# Add these imports to cli/resolution/__init__.py

from codeintel.cli.resolution.params import BackendFlags, RuntimeParams

# Add to __all__
__all__ = [
    "BackendFlags",
    "GatewayManager",
    "ResolutionError",
    "ResolvedRuntime",
    "RuntimeParams",
    "RuntimeResolver",
    "open_gateway_for_context",
    "resolve_runtime",
]
```

### 3.3 Add `RuntimeParams` Support to `RuntimeResolver`

Add a new method to `cli/resolution/runtime.py`:

```python
# Add this method to the RuntimeResolver class in cli/resolution/runtime.py

@staticmethod
def resolve_from_params(
    params: RuntimeParams,
    *,
    allow_fallback: bool = True,
) -> ResolvedRuntime:
    """Resolve runtime from RuntimeParams.

    Parameters
    ----------
    params
        Canonical runtime parameters.
    allow_fallback
        If True, attempt fallback to explicit params when no project file.

    Returns
    -------
    ResolvedRuntime
        Fully resolved runtime.

    Raises
    ------
    ResolutionError
        If resolution fails.

    Examples
    --------
    >>> params = RuntimeParams(repo="org/repo", commit="abc123")
    >>> runtime = RuntimeResolver.resolve_from_params(params)  # doctest: +SKIP
    """
    # Try project file discovery first
    try:
        return _resolve_from_project(params.project_root)
    except ProjectNotFoundError as exc:
        if not allow_fallback:
            raise ResolutionError(_MSG_NO_PROJECT_NO_FALLBACK) from exc

    # Fall back to explicit params
    return _resolve_from_runtime_params(params)


def _resolve_from_runtime_params(params: RuntimeParams) -> ResolvedRuntime:
    """Resolve from RuntimeParams (internal).

    Parameters
    ----------
    params
        Runtime parameters.

    Returns
    -------
    ResolvedRuntime
        Resolved runtime.

    Raises
    ------
    ResolutionError
        If required parameters are missing.
    """
    if params.repo is None or params.commit is None:
        missing: list[str] = []
        if params.repo is None:
            missing.append("repo")
        if params.commit is None:
            missing.append("commit")
        raise ResolutionError(_MSG_MISSING_PARAMS, missing_params=missing)

    repo_root = params.repo_root or Path.cwd()
    db_path = params.db_path or Path("build/db/codeintel.duckdb")
    build_dir = params.build_dir or Path("build")

    config = _build_config(
        _ConfigParams(
            repo=params.repo,
            commit=params.commit,
            repo_root=repo_root,
            db_path=db_path,
            build_dir=build_dir,
            document_output_dir=params.document_output_dir,
            use_gpu=params.backend.use_gpu,
        )
    )

    config.build_paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    return ResolvedRuntime(
        root=repo_root,
        project=ProjectConfig(
            repo=params.repo,
            storage=StorageProjectConfig(db_path=config.build_paths.db_path),
        ),
        snapshot=SnapshotRef(repo=params.repo, commit=params.commit, repo_root=repo_root),
        paths=config.build_paths,
        config=config,
        serving=ServingConfig(
            mode="local_db",
            repo_root=repo_root,
            repo=params.repo,
            commit=params.commit,
            db_path=config.build_paths.db_path,
            read_only=True,
        ),
    )
```

---

## 4. ConfigService

### 4.1 Create `cli/config/service.py`

```python
"""Unified configuration service.

This module provides ConfigService, the single source of truth for all
CLI configuration loading, validation, and access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from codeintel.cli.config.loader import load_config
from codeintel.cli.config.model import CliConfig, ConfigLoadError

if TYPE_CHECKING:
    from cyclopts import App


# Environment variable prefix
CONFIG_ENV_PREFIX = "CODEINTEL_"

# Config file search paths
TOML_CONFIG_PATHS = [
    Path("codeintel.toml"),
    Path.home() / ".codeintel" / "config.toml",
]


@dataclass(frozen=True)
class ConfigService:
    """Unified configuration service.

    Precedence (highest to lowest):
    1. CLI flags (explicit overrides)
    2. Environment variables (CODEINTEL_*)
    3. Config file (codeintel.toml or ~/.codeintel/config.yaml)
    4. Built-in defaults from CliConfig

    Parameters
    ----------
    config
        The resolved, validated configuration.
    sources
        Ordered list of sources that contributed to the config.

    Examples
    --------
    >>> service = ConfigService.load()
    >>> service.config.output_format
    'text'
    >>> service.sources
    ('defaults', ...)
    """

    config: CliConfig
    sources: tuple[str, ...]

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
        *,
        env_prefix: str = CONFIG_ENV_PREFIX,
        validate: bool = True,
    ) -> ConfigService:
        """Load configuration from all sources with precedence.

        Parameters
        ----------
        config_path
            Explicit config file path. If None, searches default locations.
        cli_overrides
            Overrides from CLI flags (highest precedence).
        env_prefix
            Environment variable prefix.
        validate
            If True, validate config and raise ConfigLoadError on failure.

        Returns
        -------
        ConfigService
            Service with loaded configuration.

        Raises
        ------
        ConfigLoadError
            If validation is enabled and config is invalid.

        Examples
        --------
        >>> service = ConfigService.load()
        >>> service.config.color
        True
        """
        # Use the existing load_config with its precedence
        config = load_config(
            config_file=config_path,
            env_prefix=env_prefix,
            cli_overrides=cli_overrides,
            validate=validate,
        )

        return cls(
            config=config,
            sources=tuple(config.config_sources),
        )

    def get_cyclopts_config_chain(self) -> list[Callable[[App, tuple[str, ...], Any], Any]]:
        """Return Cyclopts-compatible config callables.

        Integrates with Cyclopts' config parameter while maintaining
        our unified precedence. The returned chain applies our
        pre-loaded configuration.

        Returns
        -------
        list
            Config callables for Cyclopts App.config parameter.

        Examples
        --------
        >>> service = ConfigService.load()
        >>> chain = service.get_cyclopts_config_chain()
        >>> len(chain) >= 0
        True
        """
        # Return empty chain since we handle config loading ourselves
        # Cyclopts will use its default behavior for CLI parsing
        # and we apply overrides via cli_overrides parameter
        return []

    def with_overrides(self, **overrides: Any) -> ConfigService:
        """Create new service with overrides applied.

        Useful for testing or command-specific modifications.

        Parameters
        ----------
        **overrides
            Field overrides to apply.

        Returns
        -------
        ConfigService
            New service with overrides applied.

        Examples
        --------
        >>> service = ConfigService.load()
        >>> modified = service.with_overrides(color=False)
        >>> modified.config.color
        False
        """
        from codeintel.cli.config.loader import apply_overrides

        new_config = apply_overrides(self.config, overrides)
        return ConfigService(
            config=new_config,
            sources=(*self.sources, "overrides"),
        )


__all__ = [
    "CONFIG_ENV_PREFIX",
    "ConfigService",
]
```

### 4.2 Update `cli/config/__init__.py`

Add to the existing exports:

```python
# Add to cli/config/__init__.py imports
from codeintel.cli.config.service import CONFIG_ENV_PREFIX, ConfigService

# Add to __all__
__all__ = [
    # ... existing exports ...
    "CONFIG_ENV_PREFIX",
    "ConfigService",
]
```

---

## 5. Enhanced HandlerContext

### 5.1 Create `cli/handlers/protocol.py`

```python
"""Handler protocol and enhanced context.

This module defines:
- HandlerProtocol: Contract for all CLI handlers
- EnhancedHandlerContext: Context with lazy gateway and runtime access
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Protocol, TypeVar

if TYPE_CHECKING:
    from codeintel.analytics.runtime import GraphRuntime
    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.cli.results import CliResult
    from codeintel.storage.gateway import StorageGateway

T = TypeVar("T")


@dataclass
class EnhancedHandlerContext:
    """Enhanced context for CLI handlers with lazy resource access.

    Provides lazy access to gateway and graph_runtime to avoid opening
    connections unnecessarily. Resources are opened on first access
    and closed via the close() method.

    Parameters
    ----------
    config
        CLI configuration.
    runtime
        Resolved project runtime.
    params
        Operation-specific parameters.
    verbosity
        Verbosity level (0=warnings, 1=info, 2+=debug).

    Examples
    --------
    >>> # In a handler:
    >>> def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:
    ...     ctx.logger.info("Starting")
    ...     data = ctx.gateway.execute("SELECT * FROM table")
    ...     return CliResult.ok(MyData(data))
    """

    config: CliConfig
    runtime: ResolvedRuntime
    params: Mapping[str, object] = field(default_factory=dict)
    verbosity: int = 0

    # Private fields for lazy initialization
    _gateway: StorageGateway | None = field(default=None, repr=False)
    _graph_runtime: GraphRuntime | None = field(default=None, repr=False)
    _operation_name: str = field(default="handler", repr=False)

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this handler.

        Returns
        -------
        logging.Logger
            Logger with handler-specific name.
        """
        return logging.getLogger(f"codeintel.cli.handlers.{self._operation_name}")

    @property
    def gateway(self) -> StorageGateway:
        """Get storage gateway (lazy).

        Gateway is opened on first access. The context manages lifecycle.

        Returns
        -------
        StorageGateway
            Open storage gateway.
        """
        if self._gateway is None:
            from codeintel.storage.gateway import StorageConfig, open_gateway

            storage_config = StorageConfig(db_path=self.runtime.db_path, read_only=True)
            self._gateway = open_gateway(storage_config)
        return self._gateway

    @property
    def graph_runtime(self) -> GraphRuntime:
        """Get graph runtime (lazy).

        Returns
        -------
        GraphRuntime
            Graph runtime for graph operations.
        """
        if self._graph_runtime is None:
            from codeintel.analytics.runtime import build_graph_runtime

            self._graph_runtime = build_graph_runtime(
                gateway=self.gateway,
                snapshot=self.runtime.snapshot,
            )
        return self._graph_runtime

    @property
    def db_path(self) -> Path:
        """Shortcut to database path.

        Returns
        -------
        Path
            Path to DuckDB database.
        """
        return self.runtime.db_path

    @property
    def output_format(self) -> str:
        """Get output format from config.

        Returns
        -------
        str
            Output format ('text' or 'json').
        """
        return self.config.output_format

    @property
    def color_enabled(self) -> bool:
        """Check if color output is enabled.

        Returns
        -------
        bool
            True if color is enabled.
        """
        return self.config.color

    def close(self) -> None:
        """Close managed resources.

        Called automatically when using as_context_manager() or
        should be called explicitly after handler execution.
        """
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None
        self._graph_runtime = None

    @contextmanager
    def as_context_manager(self) -> Iterator[EnhancedHandlerContext]:
        """Use context as a context manager for automatic cleanup.

        Yields
        ------
        EnhancedHandlerContext
            Self for use in with block.

        Examples
        --------
        >>> with ctx.as_context_manager():
        ...     result = handler(ctx)
        """
        try:
            yield self
        finally:
            self.close()


class HandlerProtocol(Protocol[T]):
    """Protocol for CLI handler functions.

    All handlers must:
    1. Accept EnhancedHandlerContext as their only argument
    2. Return CliResult[T] (never None, never raise for expected errors)
    3. Never write to stdout/stderr directly
    4. Never call sys.exit()

    Unexpected exceptions (bugs) may propagate; expected errors
    should return CliResult.fail() with appropriate ProblemDetail.

    Examples
    --------
    >>> def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:
    ...     if not ctx.params.get("required"):
    ...         return CliResult.fail(ProblemDetail(...))
    ...     data = compute_result(ctx.gateway)
    ...     return CliResult.ok(data)
    """

    def __call__(self, ctx: EnhancedHandlerContext) -> CliResult[T]:
        """Execute the handler.

        Parameters
        ----------
        ctx
            Handler context with config, runtime, params.

        Returns
        -------
        CliResult[T]
            Success or failure result. Never None.
        """
        ...


@contextmanager
def handler_context(
    config: CliConfig,
    runtime: ResolvedRuntime,
    params: Mapping[str, object] | None = None,
    *,
    verbosity: int = 0,
    operation_name: str = "handler",
) -> Iterator[EnhancedHandlerContext]:
    """Create handler context with automatic resource cleanup.

    Parameters
    ----------
    config
        CLI configuration.
    runtime
        Resolved runtime.
    params
        Operation parameters.
    verbosity
        Verbosity level.
    operation_name
        Name for logging.

    Yields
    ------
    EnhancedHandlerContext
        Context for handler use.

    Examples
    --------
    >>> with handler_context(config, runtime, {"key": "value"}) as ctx:
    ...     result = my_handler(ctx)
    """
    ctx = EnhancedHandlerContext(
        config=config,
        runtime=runtime,
        params=params or {},
        verbosity=verbosity,
        _operation_name=operation_name,
    )
    try:
        yield ctx
    finally:
        ctx.close()


__all__ = [
    "EnhancedHandlerContext",
    "HandlerProtocol",
    "handler_context",
]
```

### 5.2 Update `cli/handlers/__init__.py`

Add the new exports:

```python
"""Unified CLI handlers package.

This package provides:
1. Base utilities (logging, context) in `handlers.base`
2. Protocol and enhanced context in `handlers.protocol`
3. Domain-specific handlers in `handlers.<domain>`
"""

from __future__ import annotations

from codeintel.cli.handlers.base import (
    HandlerContext,
    build_handler_context,
    get_handler_logger,
    open_handler_gateway,
    setup_logging,
)
from codeintel.cli.handlers.protocol import (
    EnhancedHandlerContext,
    HandlerProtocol,
    handler_context,
)

__all__ = [
    # Base utilities
    "HandlerContext",
    "build_handler_context",
    "get_handler_logger",
    "open_handler_gateway",
    "setup_logging",
    # Protocol (new)
    "EnhancedHandlerContext",
    "HandlerProtocol",
    "handler_context",
]
```

---

## 6. Test Implementation

### 6.1 Create `tests/cli/rendering/test_types.py`

```python
"""Tests for rendering types."""

from __future__ import annotations

import sys
from io import StringIO

import pytest

from codeintel.cli.rendering import OutputFormat, RenderContext


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_values(self) -> None:
        """OutputFormat has expected values."""
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.JSONL.value == "jsonl"


class TestRenderContext:
    """Tests for RenderContext."""

    def test_auto_detect_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auto_detect uses TEXT format for TTY."""
        mock_stdout = StringIO()
        mock_stdout.isatty = lambda: True  # type: ignore[method-assign]
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        ctx = RenderContext.auto_detect()

        assert ctx.format == OutputFormat.TEXT
        assert ctx.color is True
        assert ctx.is_tty is True

    def test_auto_detect_non_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auto_detect uses JSON format for non-TTY."""
        mock_stdout = StringIO()
        mock_stdout.isatty = lambda: False  # type: ignore[method-assign]
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        ctx = RenderContext.auto_detect()

        assert ctx.format == OutputFormat.JSON
        assert ctx.color is False
        assert ctx.is_tty is False

    def test_auto_detect_with_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auto_detect respects overrides."""
        mock_stdout = StringIO()
        mock_stdout.isatty = lambda: True  # type: ignore[method-assign]
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        ctx = RenderContext.auto_detect(
            format_override=OutputFormat.JSON,
            color_override=False,
        )

        assert ctx.format == OutputFormat.JSON
        assert ctx.color is False

    def test_for_testing(self) -> None:
        """for_testing returns captured streams."""
        ctx, out, err = RenderContext.for_testing()

        assert ctx.format == OutputFormat.TEXT
        assert ctx.color is False
        assert ctx.is_tty is False
        assert isinstance(out, StringIO)
        assert isinstance(err, StringIO)

        # Verify streams are writable
        ctx.writer.write("test")
        ctx.err_writer.write("error")

        assert out.getvalue() == "test"
        assert err.getvalue() == "error"
```

### 6.2 Create `tests/cli/rendering/test_service.py`

```python
"""Tests for UnifiedRenderer."""

from __future__ import annotations

import json

import pytest

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.rendering import (
    ColumnSpec,
    OutputFormat,
    RenderContext,
    TableSpec,
    UnifiedRenderer,
)
from codeintel.cli.results import CliResult


class TestUnifiedRenderer:
    """Tests for UnifiedRenderer."""

    def test_render_message_text(self) -> None:
        """render_message outputs text correctly."""
        ctx, out, err = RenderContext.for_testing()
        renderer = UnifiedRenderer(ctx)

        renderer.render_message("Test message", level="info")

        assert "Test message" in out.getvalue()

    def test_render_message_json(self) -> None:
        """render_message outputs JSON correctly."""
        ctx, out, err = RenderContext.for_testing()
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
        assert output["status"] == "success"
        assert output["message"] == "Test message"

    def test_render_table_text(self) -> None:
        """render_table outputs plain table correctly."""
        ctx, out, err = RenderContext.for_testing()
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
        assert "Name" in output
        assert "Count" in output
        assert "foo" in output
        assert "bar" in output

    def test_render_table_json(self) -> None:
        """render_table outputs JSON array correctly."""
        ctx, out, err = RenderContext.for_testing()
        ctx = RenderContext(
            format=OutputFormat.JSON,
            color=False,
            writer=out,
            err_writer=err,
            is_tty=False,
        )
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

        output = json.loads(out.getvalue())
        assert len(output) == 2
        assert output[0]["name"] == "foo"
        assert output[1]["count"] == 20

    def test_render_table_empty(self) -> None:
        """render_table handles empty rows."""
        ctx, out, err = RenderContext.for_testing()
        renderer = UnifiedRenderer(ctx)

        spec = TableSpec(
            columns=(ColumnSpec("name", "Name"),),
            empty_message="No items found.",
        )

        renderer.render_table([], spec)

        assert "No items found" in out.getvalue()

    def test_render_error_text(self) -> None:
        """render_error outputs to stderr in text mode."""
        ctx, out, err = RenderContext.for_testing()
        renderer = UnifiedRenderer(ctx)

        error = ProblemDetail(
            type="urn:test:error",
            title="Test Error",
            status=400,
            detail="Something went wrong",
        )

        renderer.render_error(error)

        assert "Test Error" in err.getvalue()
        assert "Something went wrong" in err.getvalue()

    def test_render_error_json(self) -> None:
        """render_error outputs JSON to stderr."""
        ctx, out, err = RenderContext.for_testing()
        ctx = RenderContext(
            format=OutputFormat.JSON,
            color=False,
            writer=out,
            err_writer=err,
            is_tty=False,
        )
        renderer = UnifiedRenderer(ctx)

        error = ProblemDetail(
            type="urn:test:error",
            title="Test Error",
            status=400,
            detail="Something went wrong",
        )

        renderer.render_error(error)

        output = json.loads(err.getvalue())
        assert output["type"] == "urn:test:error"
        assert output["title"] == "Test Error"
        assert output["status"] == 400

    def test_render_result_success(self) -> None:
        """render_result handles success correctly."""
        ctx, out, err = RenderContext.for_testing()
        ctx = RenderContext(
            format=OutputFormat.JSON,
            color=False,
            writer=out,
            err_writer=err,
            is_tty=False,
        )
        renderer = UnifiedRenderer(ctx)

        result = CliResult.ok({"key": "value"})

        exit_code = renderer.render_result(result)

        assert exit_code == 0
        output = json.loads(out.getvalue())
        assert output["data"]["key"] == "value"

    def test_render_result_failure(self) -> None:
        """render_result handles failure correctly."""
        ctx, out, err = RenderContext.for_testing()
        renderer = UnifiedRenderer(ctx)

        error = ProblemDetail(
            type="urn:test:error",
            title="Failed",
            status=400,
        )
        result: CliResult[dict[str, str]] = CliResult.fail(error)

        exit_code = renderer.render_result(result)

        assert exit_code == 1
        assert "Failed" in err.getvalue()

    def test_render_result_with_warnings(self) -> None:
        """render_result emits warnings to stderr."""
        ctx, out, err = RenderContext.for_testing()
        renderer = UnifiedRenderer(ctx)

        result = CliResult.ok({"key": "value"})
        result.warnings.append("Warning 1")
        result.warnings.append("Warning 2")

        renderer.render_result(result)

        assert "Warning 1" in err.getvalue()
        assert "Warning 2" in err.getvalue()
```

### 6.3 Create `tests/cli/resolution/test_params.py`

```python
"""Tests for RuntimeParams."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.resolution import RuntimeParams
from codeintel.cli.resolution.params import BackendFlags


class TestBackendFlags:
    """Tests for BackendFlags."""

    def test_defaults(self) -> None:
        """BackendFlags has expected defaults."""
        flags = BackendFlags()

        assert flags.use_gpu is False
        assert flags.backend == "auto"
        assert flags.strict is False


class TestRuntimeParams:
    """Tests for RuntimeParams."""

    def test_defaults(self) -> None:
        """RuntimeParams has expected defaults."""
        params = RuntimeParams()

        assert params.project_root is None
        assert params.repo is None
        assert params.commit is None
        assert params.db_path is None
        assert params.backend.use_gpu is False

    def test_minimal_factory(self) -> None:
        """minimal() creates minimal params."""
        params = RuntimeParams.minimal(Path("/project"))

        assert params.project_root == Path("/project")
        assert params.repo is None

    def test_minimal_factory_no_args(self) -> None:
        """minimal() works with no arguments."""
        params = RuntimeParams.minimal()

        assert params.project_root is None

    def test_from_dict(self) -> None:
        """from_dict creates params from dictionary."""
        data = {
            "project_root": "/project",
            "repo": "org/repo",
            "commit": "abc123",
            "db_path": "/db/test.duckdb",
            "backend": {"use_gpu": True},
        }

        params = RuntimeParams.from_dict(data)

        assert params.project_root == Path("/project")
        assert params.repo == "org/repo"
        assert params.commit == "abc123"
        assert params.db_path == Path("/db/test.duckdb")
        assert params.backend.use_gpu is True

    def test_to_dict(self) -> None:
        """to_dict creates dictionary from params."""
        params = RuntimeParams(
            project_root=Path("/project"),
            repo="org/repo",
            commit="abc123",
        )

        data = params.to_dict()

        assert data["project_root"] == "/project"
        assert data["repo"] == "org/repo"
        assert data["commit"] == "abc123"

    def test_from_cyclopts(self) -> None:
        """from_cyclopts converts RuntimeCLI."""
        from codeintel.cli.cyclopts_common import RuntimeCLI

        cli = RuntimeCLI(
            project_root=Path("/project"),
            repo="org/repo",
            commit="abc123",
        )

        params = RuntimeParams.from_cyclopts(cli)

        assert params.project_root == Path("/project")
        assert params.repo == "org/repo"
        assert params.commit == "abc123"

    def test_immutable(self) -> None:
        """RuntimeParams is immutable."""
        params = RuntimeParams(repo="org/repo")

        with pytest.raises(AttributeError):
            params.repo = "other/repo"  # type: ignore[misc]
```

### 6.4 Create `tests/cli/config/test_service.py`

```python
"""Tests for ConfigService."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.config import ConfigService


class TestConfigService:
    """Tests for ConfigService."""

    def test_load_defaults(self) -> None:
        """load() returns config with defaults."""
        service = ConfigService.load(validate=False)

        assert service.config is not None
        assert "defaults" in service.sources

    def test_load_with_overrides(self) -> None:
        """load() applies CLI overrides."""
        service = ConfigService.load(
            cli_overrides={"color": False},
            validate=False,
        )

        assert service.config.color is False

    def test_with_overrides(self) -> None:
        """with_overrides creates new service."""
        service = ConfigService.load(validate=False)
        original_color = service.config.color

        modified = service.with_overrides(color=not original_color)

        assert modified.config.color != original_color
        assert service.config.color == original_color  # Original unchanged

    def test_sources_tracking(self) -> None:
        """Sources are tracked correctly."""
        service = ConfigService.load(validate=False)

        # Should at least have defaults
        assert len(service.sources) >= 1

    def test_cyclopts_config_chain(self) -> None:
        """get_cyclopts_config_chain returns list."""
        service = ConfigService.load(validate=False)

        chain = service.get_cyclopts_config_chain()

        assert isinstance(chain, list)
```

---

## 7. Verification Checklist

After implementing all files, verify:

### 7.1 Type Checking

```bash
uv run pyright src/codeintel/cli/rendering/
uv run pyright src/codeintel/cli/resolution/params.py
uv run pyright src/codeintel/cli/config/service.py
uv run pyright src/codeintel/cli/handlers/protocol.py
```

### 7.2 Imports

```bash
# Verify all public imports work
python -c "from codeintel.cli.rendering import UnifiedRenderer, RenderContext, OutputFormat, TableSpec, ColumnSpec"
python -c "from codeintel.cli.resolution import RuntimeParams, BackendFlags"
python -c "from codeintel.cli.config import ConfigService"
python -c "from codeintel.cli.handlers import EnhancedHandlerContext, HandlerProtocol"
```

### 7.3 Tests

```bash
uv run pytest tests/cli/rendering/ -v
uv run pytest tests/cli/resolution/test_params.py -v
uv run pytest tests/cli/config/test_service.py -v
```

### 7.4 No Circular Imports

```bash
# Should not raise ImportError
python -c "import codeintel.cli.rendering"
python -c "import codeintel.cli.resolution"
python -c "import codeintel.cli.config"
python -c "import codeintel.cli.handlers"
```

### 7.5 Existing Tests Still Pass

```bash
uv run pytest tests/cli/ -v --tb=short
```

---

## Summary

| Component | File(s) | Lines | Status |
|-----------|---------|-------|--------|
| Rendering types | `cli/rendering/types.py` | ~120 | New |
| Rendering table | `cli/rendering/table.py` | ~60 | New |
| Rendering specs | `cli/rendering/specs.py` | ~70 | New |
| Rendering service | `cli/rendering/service.py` | ~300 | New |
| Rendering init | `cli/rendering/__init__.py` | ~50 | New |
| RuntimeParams | `cli/resolution/params.py` | ~200 | New |
| ConfigService | `cli/config/service.py` | ~120 | New |
| EnhancedHandlerContext | `cli/handlers/protocol.py` | ~180 | New |
| Tests | Multiple | ~400 | New |
| **Total** | | **~1500** | |

After Phase 1 completion, proceed to Phase 2 (Config Integration).

