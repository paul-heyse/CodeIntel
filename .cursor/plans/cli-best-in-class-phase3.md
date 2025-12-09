# CLI Best-in-Class Implementation Plan (Phase 3)

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Created**: 2025-12-09  
> **Depends On**: Phase 2 (Completed)

---

## Executive Summary

This document outlines the next evolution of the CodeIntel CLI toward a best-in-class developer experience. Building on the foundational infrastructure from Phase 2 (unified types, stdin support, CliResult pattern, dry-run mode, middleware, progress reporting, and validation), Phase 3 focuses on **completeness**, **resilience**, and **operational excellence**.

The six phases below address:
1. **User Experience** — Consistent, beautiful output across all commands
2. **Code Quality** — Complete migration to structured patterns
3. **Robustness** — Validated configuration and resilient operations
4. **Testability** — Charter-compliant test infrastructure
5. **Discoverability** — Shell completions for faster workflows
6. **Reliability** — Retry logic for transient failures

---

## Table of Contents

1. [Phase 3.1: Unified Output Rendering](#phase-31-unified-output-rendering)
2. [Phase 3.2: Complete Handler Migration](#phase-32-complete-handler-migration)
3. [Phase 3.3: Configuration Validation Integration](#phase-33-configuration-validation-integration)
4. [Phase 3.4: Test Infrastructure Refactoring](#phase-34-test-infrastructure-refactoring)
5. [Phase 3.5: Shell Completion Generation](#phase-35-shell-completion-generation)
6. [Phase 3.6: Resilience and Retry Layer](#phase-36-resilience-and-retry-layer)
7. [Implementation Timeline](#implementation-timeline)
8. [Success Metrics](#success-metrics)

---

## Phase 3.1: Unified Output Rendering

### Value Proposition

Currently, CLI output formatting is scattered across handlers with inconsistent patterns:
- Some handlers use `rich.table.Table` directly
- Some use `rich.console.Console.print`
- Some use `json.dumps` with varying indent/sort settings
- Error rendering varies between plain text and structured formats

A unified rendering layer provides:
- **Consistency** — Every command looks and feels the same
- **Accessibility** — Proper support for non-TTY environments (CI/CD pipelines)
- **Extensibility** — Easy to add new output formats (YAML, CSV, etc.)
- **Testability** — Output can be captured and verified without mocking

### Functional Objectives

1. Create a `CliRenderer` protocol and default implementation
2. Support text, JSON, and table output modes
3. Integrate RFC 9457 Problem Details for error rendering
4. Auto-detect TTY vs non-TTY environments
5. Support theming and color customization

### Implementation

#### File: `src/codeintel/cli/cli_render.py`

```python
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, Self

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class RenderMode(Enum):
    """Output rendering mode."""

    AUTO = "auto"  # Detect based on TTY
    RICH = "rich"  # Force rich formatting
    PLAIN = "plain"  # Plain text, no colors
    JSON = "json"  # JSON output
    JSONL = "jsonl"  # JSON Lines (one object per line)


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
        Text justification ("left", "center", "right").
    width
        Fixed column width (None for auto).
    """

    key: str
    header: str
    style: str | None = None
    justify: str = "left"
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

    console: Console = field(default_factory=lambda: Console(theme=CODEINTEL_THEME))
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
            self._render_json([dict(row) for row in rows])
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
            self._render_json(obj)
            return

        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
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
            self._render_json(error.to_dict())
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
            self._render_json({"status": "success", "message": message})
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
            self._render_json({"status": "warning", "message": message})
            return
        self.console.print(f"[warning]⚠[/warning] {message}")

    def _render_json(self, obj: object) -> None:
        """Render object as JSON.

        Parameters
        ----------
        obj
            Object to serialize.
        """
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        print(json.dumps(obj, indent=2, default=str))


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
            self._render_json([dict(row) for row in rows])
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
        header = " | ".join(
            col.header.ljust(widths[col.key]) for col in spec.columns
        )
        print(header)
        print("-" * len(header))

        # Rows
        for row in rows:
            line = " | ".join(
                str(row.get(col.key, "")).ljust(widths[col.key])
                for col in spec.columns
            )
            print(line)

    def render_object(self, obj: object) -> None:
        """Render a single object as plain text.

        Parameters
        ----------
        obj
            Object to render.
        """
        if self.output_format == OutputFormat.JSON:
            self._render_json(obj)
            return

        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
        elif hasattr(obj, "__dict__"):
            data = obj.__dict__
        else:
            data = obj

        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print(data)

    def render_error(self, error: ProblemDetail) -> None:
        """Render an error as plain text.

        Parameters
        ----------
        error
            Problem detail to render.
        """
        if self.output_format == OutputFormat.JSON:
            self._render_json(error.to_dict())
            return

        print(f"Error: {error.title}", file=sys.stderr)
        if error.detail:
            print(error.detail, file=sys.stderr)

    def render_success(self, message: str) -> None:
        """Render a success message.

        Parameters
        ----------
        message
            Success message text.
        """
        if self.output_format == OutputFormat.JSON:
            self._render_json({"status": "success", "message": message})
            return
        print(f"OK: {message}")

    def render_warning(self, message: str) -> None:
        """Render a warning message.

        Parameters
        ----------
        message
            Warning message text.
        """
        if self.output_format == OutputFormat.JSON:
            self._render_json({"status": "warning", "message": message})
            return
        print(f"Warning: {message}", file=sys.stderr)

    def _render_json(self, obj: object) -> None:
        """Render object as JSON.

        Parameters
        ----------
        obj
            Object to serialize.
        """
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        print(json.dumps(obj, indent=2, default=str))


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
        return result.exit_code

    if result.value is None:
        return 0

    # Handle tabular data
    if table_spec and isinstance(result.value, list):
        renderer.render_table(result.value, table_spec)
        return 0

    # Handle objects with to_dict
    renderer.render_object(result.value)
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
    "ColumnSpec",
    "DATASET_TABLE_SPEC",
    "OPERATION_TABLE_SPEC",
    "OutputRenderer",
    "PlainRenderer",
    "RenderMode",
    "RichRenderer",
    "TableSpec",
    "get_renderer",
    "render_cli_result",
]
```

### Usage Example

```python
# In a handler:
from codeintel.cli.cli_render import (
    get_renderer,
    render_cli_result,
    OPERATION_TABLE_SPEC,
)

def op_list_command(output_format: OutputFormat = OutputFormat.TEXT) -> int:
    """List available operations."""
    result = op_list_handler_structured()
    renderer = get_renderer(output_format)
    return render_cli_result(result, renderer, table_spec=OPERATION_TABLE_SPEC)
```

---

## Phase 3.2: Complete Handler Migration

### Value Proposition

Currently, most handlers in the codebase still use direct printing, which creates:
- **Testing friction** — Tests must capture stdout/stderr
- **Composition barriers** — Handlers can't easily call other handlers
- **Inconsistent error handling** — Some throw, some print, some return codes

Complete migration to `CliResult` provides:
- **Uniform interface** — Every handler returns structured data
- **Composability** — Build complex workflows from simple handlers
- **Testability** — Assert on returned values, not captured output
- **Error propagation** — Errors bubble up with full context

### Functional Objectives

1. Migrate all handlers in `build_handlers.py` (largest file, 882 lines)
2. Migrate all handlers in `graphs_handlers.py`
3. Migrate remaining handlers across `cyclopts_*.py` modules
4. Create result types for all handler outputs
5. Update command wrappers to use `render_cli_result`

### Implementation Strategy

#### Step 1: Expand Result Types

Add to `src/codeintel/cli/result_types.py`:

```python
# Build-related result types
@dataclass(frozen=True)
class BuildTargetInfo:
    """Information about a build target.

    Parameters
    ----------
    name
        Target name.
    status
        Current status (fresh, stale, error).
    last_run
        Timestamp of last successful run.
    dependencies
        List of dependency target names.
    outputs
        List of output paths.
    """

    name: str
    status: str
    last_run: str | None
    dependencies: list[str]
    outputs: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "last_run": self.last_run,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
        }


@dataclass(frozen=True)
class BuildExecutionResult:
    """Result from executing a build.

    Parameters
    ----------
    targets_executed
        List of targets that were executed.
    targets_skipped
        List of targets that were skipped (already fresh).
    targets_failed
        List of targets that failed.
    total_duration_seconds
        Total execution time.
    """

    targets_executed: list[str]
    targets_skipped: list[str]
    targets_failed: list[str]
    total_duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "targets_executed": self.targets_executed,
            "targets_skipped": self.targets_skipped,
            "targets_failed": self.targets_failed,
            "total_duration_seconds": self.total_duration_seconds,
            "success": len(self.targets_failed) == 0,
        }


# Graph-related result types
@dataclass(frozen=True)
class GraphStatsResult:
    """Statistics about a graph.

    Parameters
    ----------
    node_count
        Number of nodes.
    edge_count
        Number of edges.
    density
        Graph density (0.0 to 1.0).
    components
        Number of connected components.
    avg_degree
        Average node degree.
    """

    node_count: int
    edge_count: int
    density: float
    components: int
    avg_degree: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "components": self.components,
            "avg_degree": self.avg_degree,
        }


@dataclass(frozen=True)
class GraphQueryResult:
    """Result from a graph query.

    Parameters
    ----------
    nodes
        Matching nodes with their attributes.
    edges
        Edges between matching nodes.
    query
        The query that was executed.
    """

    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    query: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "query": self.query,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }
```

#### Step 2: Migration Pattern

For each handler, follow this pattern:

**Before (direct printing):**
```python
def build_status_handler(
    runtime: RuntimeCliOptions,
    output_format: OutputFormat,
) -> None:
    """Show build target status."""
    targets = get_build_targets(runtime)
    
    if output_format == OutputFormat.JSON:
        print(json.dumps([t.to_dict() for t in targets], indent=2))
        return
    
    table = Table(title="Build Targets")
    table.add_column("Target")
    table.add_column("Status")
    for target in targets:
        table.add_row(target.name, target.status)
    console.print(table)
```

**After (structured result):**
```python
def build_status_handler_structured(
    runtime: RuntimeCliOptions,
) -> CliResult[BuildStatusResult]:
    """Show build target status.

    Parameters
    ----------
    runtime
        Runtime configuration options.

    Returns
    -------
    CliResult[BuildStatusResult]
        Structured result with target information.
    """
    try:
        targets = get_build_targets(runtime)
        target_dicts = [
            {
                "name": t.name,
                "status": t.status,
                "last_run": t.last_run,
            }
            for t in targets
        ]
        return CliResult.ok(
            BuildStatusResult(
                targets=target_dicts,
                stale_count=sum(1 for t in targets if t.status == "stale"),
                fresh_count=sum(1 for t in targets if t.status == "fresh"),
            )
        )
    except StorageError as e:
        return CliResult.error(
            ProblemDetail(
                type_uri="urn:codeintel:cli:storage-error",
                title="Storage Error",
                detail=str(e),
                status=500,
            )
        )
```

#### Step 3: Command Wrapper Update

```python
@build_app.command()
def status(
    output_format: Annotated[
        OutputFormat, Parameter(help="Output format")
    ] = OutputFormat.TEXT,
) -> None:
    """Show build target status."""
    runtime = resolve_runtime_options()
    result = build_status_handler_structured(runtime)
    renderer = get_renderer(output_format)
    exit_code = render_cli_result(result, renderer, table_spec=BUILD_TARGET_TABLE_SPEC)
    if exit_code != 0:
        raise SystemExit(exit_code)
```

### Migration Checklist

| File | Handlers | Priority | Estimated Effort |
|------|----------|----------|------------------|
| `build_handlers.py` | 12 handlers | High | 4 hours |
| `graphs_handlers.py` | 8 handlers | High | 3 hours |
| `docs_handlers.py` | 5 handlers | Medium | 2 hours |
| `storage_handlers.py` | 6 handlers | Medium | 2 hours |
| `config_handlers.py` | 4 handlers | Low | 1 hour |

---

## Phase 3.3: Configuration Validation Integration

### Value Proposition

The validation layer from Phase 2 (`cli_validation.py`) provides composable validators, but they're not yet integrated into:
- Configuration file parsing
- Operation parameter validation
- Runtime option validation

Integration provides:
- **Fail-fast behavior** — Invalid config detected at startup
- **Helpful error messages** — Clear guidance on fixing issues
- **Schema documentation** — Validation rules serve as documentation
- **Consistency** — Same validation logic for files and CLI args

### Functional Objectives

1. Create configuration schema definitions
2. Validate configuration files at load time
3. Validate operation parameters before execution
4. Provide detailed error messages with fix suggestions

### Implementation

#### File: `src/codeintel/cli/cli_config_schema.py`

```python
"""Configuration schema definitions and validation.

This module defines the expected structure of configuration files
and provides validation using the cli_validation infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.cli.cli_validation import (
    IntValidator,
    PathValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    ValidationSchema,
)


# String validators for common config fields
REPO_NAME_VALIDATOR = StringValidator(
    min_length=1,
    max_length=200,
    pattern=r"^[a-zA-Z0-9_\-\.\/]+$",
)

COMMIT_SHA_VALIDATOR = StringValidator(
    min_length=7,
    max_length=40,
    pattern=r"^[a-fA-F0-9]+$",
)

LOG_LEVEL_VALIDATOR = StringValidator(
    allowed_values={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
)


@dataclass
class ConfigSection:
    """Base class for configuration sections.

    Subclasses define the schema for specific config sections
    and provide validation logic.
    """

    @classmethod
    def schema(cls) -> ValidationSchema:
        """Get the validation schema for this section.

        Returns
        -------
        ValidationSchema
            Schema for validating this section.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[ConfigSection]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[ConfigSection]
            Validated configuration or errors.
        """
        raise NotImplementedError


@dataclass
class StorageConfig(ConfigSection):
    """Storage configuration section.

    Parameters
    ----------
    db_path
        Path to DuckDB database.
    cache_dir
        Directory for cached data.
    max_connections
        Maximum database connections.
    """

    db_path: Path
    cache_dir: Path
    max_connections: int = 5

    @classmethod
    def schema(cls) -> ValidationSchema:
        """Get the validation schema."""
        return (
            ValidationSchema()
            .add("db_path", PathValidator())
            .add("cache_dir", PathValidator())
            .add("max_connections", IntValidator(min_value=1, max_value=100))
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[StorageConfig]:
        """Create from dictionary with validation."""
        schema = cls.schema()
        
        # Add defaults for optional fields
        data_with_defaults = {
            "max_connections": 5,
            **data,
        }
        
        result = schema.validate(data_with_defaults)
        if not result.is_valid:
            return ValidationResult.fail(result.errors)
        
        validated = result.value
        assert validated is not None
        
        return ValidationResult.ok(
            cls(
                db_path=Path(validated["db_path"]),
                cache_dir=Path(validated["cache_dir"]),
                max_connections=int(validated["max_connections"]),
            )
        )


@dataclass
class ProjectConfig(ConfigSection):
    """Project configuration section.

    Parameters
    ----------
    name
        Project name.
    repo
        Repository identifier.
    commit
        Current commit SHA.
    root
        Project root directory.
    """

    name: str
    repo: str
    commit: str | None
    root: Path

    @classmethod
    def schema(cls) -> ValidationSchema:
        """Get the validation schema."""
        return (
            ValidationSchema()
            .add("name", StringValidator(min_length=1, max_length=100))
            .add("repo", REPO_NAME_VALIDATOR)
            .add("root", PathValidator(must_exist=True, must_be_dir=True))
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[ProjectConfig]:
        """Create from dictionary with validation."""
        errors: list[ValidationError] = []
        
        # Validate required fields
        schema = cls.schema()
        required_result = schema.validate({
            k: v for k, v in data.items() if k in ("name", "repo", "root")
        })
        
        if not required_result.is_valid:
            errors.extend(required_result.errors)
        
        # Validate optional commit if present
        if "commit" in data and data["commit"] is not None:
            commit_result = COMMIT_SHA_VALIDATOR.validate(data["commit"], "commit")
            if not commit_result.is_valid:
                errors.extend(commit_result.errors)
        
        if errors:
            return ValidationResult.fail(errors)
        
        validated = required_result.value
        assert validated is not None
        
        return ValidationResult.ok(
            cls(
                name=str(validated["name"]),
                repo=str(validated["repo"]),
                commit=data.get("commit"),
                root=Path(validated["root"]),
            )
        )


@dataclass
class FullConfig:
    """Complete configuration with all sections.

    Parameters
    ----------
    project
        Project configuration.
    storage
        Storage configuration.
    log_level
        Logging level.
    """

    project: ProjectConfig
    storage: StorageConfig
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[FullConfig]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[FullConfig]
            Validated configuration or errors.
        """
        errors: list[ValidationError] = []
        
        # Validate project section
        if "project" not in data:
            errors.append(
                ValidationError(
                    field="project",
                    message="Required section 'project' is missing",
                    code="missing_section",
                )
            )
            project = None
        else:
            project_result = ProjectConfig.from_dict(data["project"])
            if not project_result.is_valid:
                errors.extend(project_result.errors)
                project = None
            else:
                project = project_result.value
        
        # Validate storage section
        if "storage" not in data:
            errors.append(
                ValidationError(
                    field="storage",
                    message="Required section 'storage' is missing",
                    code="missing_section",
                )
            )
            storage = None
        else:
            storage_result = StorageConfig.from_dict(data["storage"])
            if not storage_result.is_valid:
                errors.extend(storage_result.errors)
                storage = None
            else:
                storage = storage_result.value
        
        # Validate log_level if present
        if "log_level" in data:
            log_result = LOG_LEVEL_VALIDATOR.validate(data["log_level"], "log_level")
            if not log_result.is_valid:
                errors.extend(log_result.errors)
        
        if errors:
            return ValidationResult.fail(errors)
        
        assert project is not None
        assert storage is not None
        
        return ValidationResult.ok(
            cls(
                project=project,
                storage=storage,
                log_level=data.get("log_level", "INFO"),
            )
        )


def validate_config_file(path: Path) -> ValidationResult[FullConfig]:
    """Validate a configuration file.

    Parameters
    ----------
    path
        Path to configuration file (YAML or JSON).

    Returns
    -------
    ValidationResult[FullConfig]
        Validated configuration or errors.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    """
    import json

    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    content = path.read_text(encoding="utf-8")
    
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)
    
    return FullConfig.from_dict(data)


def format_validation_errors(errors: list[ValidationError]) -> str:
    """Format validation errors for display.

    Parameters
    ----------
    errors
        List of validation errors.

    Returns
    -------
    str
        Formatted error message.
    """
    lines = ["Configuration validation failed:"]
    for error in errors:
        lines.append(f"  • {error.field}: {error.message}")
        if error.value:
            lines.append(f"    Got: {error.value}")
    return "\n".join(lines)


__all__ = [
    "COMMIT_SHA_VALIDATOR",
    "ConfigSection",
    "FullConfig",
    "LOG_LEVEL_VALIDATOR",
    "ProjectConfig",
    "REPO_NAME_VALIDATOR",
    "StorageConfig",
    "format_validation_errors",
    "validate_config_file",
]
```

### Integration Points

1. **Startup validation** in `cyclopts_app.py`:
```python
def _validate_config_on_startup() -> None:
    """Validate configuration at CLI startup."""
    config_path = find_config_file()
    if config_path:
        result = validate_config_file(config_path)
        if not result.is_valid:
            console.print(format_validation_errors(result.errors), style="error")
            raise SystemExit(1)
```

2. **Operation parameter validation** in `cyclopts_ops.py`:
```python
def _validate_operation_params(op_id: str, params: dict[str, Any]) -> ValidationResult:
    """Validate operation parameters against schema."""
    operation = get_operation(op_id)
    if operation.param_schema:
        return validate_against_schema(params, operation.param_schema)
    return ValidationResult.ok(params)
```

---

## Phase 3.4: Test Infrastructure Refactoring

### Value Proposition

The current CLI tests violate the testing charter:
- Use of `monkeypatch` fixture (forbidden)
- Runtime patching of production code
- Tests not safe for parallel execution

Refactoring provides:
- **Charter compliance** — Tests follow hexagonal architecture
- **Speed** — Parallel test execution enabled
- **Reliability** — No flaky tests from shared state
- **Maintainability** — Clear separation of test doubles

### Functional Objectives

1. Create protocol-based test doubles for CLI dependencies
2. Implement dependency injection in handlers
3. Remove all `monkeypatch` usage from CLI tests
4. Enable parallel test execution

### Implementation

#### File: `tests/cli/_doubles/__init__.py`

```python
"""Test doubles for CLI testing.

These are protocol-compliant implementations that can be injected
during tests without monkeypatching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.cli.results import CliResult


@dataclass
class FakeStorageGateway:
    """Test double for StorageGateway.

    Provides in-memory storage that can be pre-populated
    with test data and inspected after test execution.

    Parameters
    ----------
    tables
        Pre-populated table data.
    queries_executed
        Record of queries executed (for assertions).
    """

    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    queries_executed: list[str] = field(default_factory=list)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a query against fake storage.

        Parameters
        ----------
        sql
            SQL query string.

        Returns
        -------
        list[dict[str, Any]]
            Query results.
        """
        self.queries_executed.append(sql)
        # Simple table name extraction for basic queries
        for table_name, data in self.tables.items():
            if table_name in sql:
                return data
        return []

    def insert(self, table: str, rows: list[dict[str, Any]]) -> int:
        """Insert rows into fake storage.

        Parameters
        ----------
        table
            Table name.
        rows
            Rows to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if table not in self.tables:
            self.tables[table] = []
        self.tables[table].extend(rows)
        return len(rows)


@dataclass
class FakeOperationCatalog:
    """Test double for operation catalog.

    Parameters
    ----------
    operations
        Pre-registered operations.
    """

    operations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_operation(self, op_id: str) -> dict[str, Any] | None:
        """Get operation by ID.

        Parameters
        ----------
        op_id
            Operation identifier.

        Returns
        -------
        dict[str, Any] | None
            Operation metadata or None.
        """
        return self.operations.get(op_id)

    def list_operations(self) -> list[dict[str, Any]]:
        """List all operations.

        Returns
        -------
        list[dict[str, Any]]
            All registered operations.
        """
        return list(self.operations.values())

    def invoke(self, op_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Invoke an operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Operation result.

        Raises
        ------
        KeyError
            If operation not found.
        """
        if op_id not in self.operations:
            raise KeyError(f"Operation not found: {op_id}")
        # Return a mock result
        return {"op_id": op_id, "params": params, "status": "success"}


@dataclass
class FakeConsole:
    """Test double for rich Console.

    Captures all output for assertion.

    Parameters
    ----------
    output
        Captured output lines.
    """

    output: list[str] = field(default_factory=list)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Capture print output.

        Parameters
        ----------
        *args
            Print arguments.
        **kwargs
            Print keyword arguments.
        """
        self.output.append(" ".join(str(arg) for arg in args))

    def clear(self) -> None:
        """Clear captured output."""
        self.output.clear()


@dataclass
class FakeFileSystem:
    """Test double for file system operations.

    Parameters
    ----------
    files
        Pre-populated file contents.
    directories
        Pre-existing directories.
    """

    files: dict[str, str] = field(default_factory=dict)
    directories: set[str] = field(default_factory=set)

    def read_text(self, path: Path) -> str:
        """Read file content.

        Parameters
        ----------
        path
            File path.

        Returns
        -------
        str
            File content.

        Raises
        ------
        FileNotFoundError
            If file doesn't exist.
        """
        key = str(path)
        if key not in self.files:
            raise FileNotFoundError(f"No such file: {path}")
        return self.files[key]

    def write_text(self, path: Path, content: str) -> None:
        """Write file content.

        Parameters
        ----------
        path
            File path.
        content
            Content to write.
        """
        self.files[str(path)] = content

    def exists(self, path: Path) -> bool:
        """Check if path exists.

        Parameters
        ----------
        path
            Path to check.

        Returns
        -------
        bool
            True if exists.
        """
        key = str(path)
        return key in self.files or key in self.directories

    def is_dir(self, path: Path) -> bool:
        """Check if path is a directory.

        Parameters
        ----------
        path
            Path to check.

        Returns
        -------
        bool
            True if directory.
        """
        return str(path) in self.directories


__all__ = [
    "FakeConsole",
    "FakeFileSystem",
    "FakeOperationCatalog",
    "FakeStorageGateway",
]
```

#### File: `tests/cli/_doubles/contexts.py`

```python
"""Test context builders for CLI testing.

Provides fluent builders for constructing test scenarios
with the appropriate doubles injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from tests.cli._doubles import (
    FakeConsole,
    FakeFileSystem,
    FakeOperationCatalog,
    FakeStorageGateway,
)


@dataclass
class CliTestContext:
    """Context for CLI handler tests.

    Provides all dependencies needed by handlers,
    allowing tests to inject test doubles.

    Parameters
    ----------
    storage
        Storage gateway (real or fake).
    operations
        Operation catalog (real or fake).
    console
        Console for output (real or fake).
    filesystem
        File system adapter (real or fake).
    """

    storage: FakeStorageGateway = field(default_factory=FakeStorageGateway)
    operations: FakeOperationCatalog = field(default_factory=FakeOperationCatalog)
    console: FakeConsole = field(default_factory=FakeConsole)
    filesystem: FakeFileSystem = field(default_factory=FakeFileSystem)


class CliTestContextBuilder:
    """Fluent builder for CliTestContext.

    Example
    -------
    ```python
    ctx = (
        CliTestContextBuilder()
        .with_operation("analyze.functions", summary="Analyze functions")
        .with_table("analytics.functions", [{"name": "foo", "loc": 10}])
        .build()
    )
    ```
    """

    def __init__(self) -> None:
        """Initialize builder with empty context."""
        self._storage = FakeStorageGateway()
        self._operations = FakeOperationCatalog()
        self._console = FakeConsole()
        self._filesystem = FakeFileSystem()

    def with_operation(
        self,
        op_id: str,
        *,
        summary: str = "",
        tags: list[str] | None = None,
    ) -> Self:
        """Add an operation to the catalog.

        Parameters
        ----------
        op_id
            Operation identifier.
        summary
            Operation summary.
        tags
            Operation tags.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._operations.operations[op_id] = {
            "id": op_id,
            "summary": summary,
            "tags": tags or [],
        }
        return self

    def with_table(
        self,
        table_key: str,
        rows: list[dict[str, Any]],
    ) -> Self:
        """Add table data to storage.

        Parameters
        ----------
        table_key
            Table key.
        rows
            Table rows.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._storage.tables[table_key] = rows
        return self

    def with_file(self, path: str, content: str) -> Self:
        """Add a file to the filesystem.

        Parameters
        ----------
        path
            File path.
        content
            File content.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._filesystem.files[path] = content
        return self

    def with_directory(self, path: str) -> Self:
        """Add a directory to the filesystem.

        Parameters
        ----------
        path
            Directory path.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._filesystem.directories.add(path)
        return self

    def build(self) -> CliTestContext:
        """Build the test context.

        Returns
        -------
        CliTestContext
            Configured test context.
        """
        return CliTestContext(
            storage=self._storage,
            operations=self._operations,
            console=self._console,
            filesystem=self._filesystem,
        )


__all__ = [
    "CliTestContext",
    "CliTestContextBuilder",
]
```

#### Example Test Migration

**Before (with monkeypatch):**
```python
def test_op_list(monkeypatch):
    """Test op list command."""
    operations = [{"id": "test.op", "summary": "Test"}]
    monkeypatch.setattr("codeintel.serving.operations.catalog.list_operations", lambda: operations)
    
    result = runner.invoke(app, ["op", "list"])
    assert result.exit_code == 0
    assert "test.op" in result.output
```

**After (with test doubles):**
```python
def test_op_list():
    """Test op list command returns structured result."""
    ctx = (
        CliTestContextBuilder()
        .with_operation("test.op", summary="Test operation")
        .build()
    )
    
    result = op_list_handler_structured(ctx.operations)
    
    assert result.success
    assert result.value is not None
    assert result.value.count == 1
    assert result.value.operations[0]["id"] == "test.op"
```

---

## Phase 3.5: Shell Completion Generation

### Value Proposition

Shell completions dramatically improve CLI usability:
- **Discoverability** — Users find commands without documentation
- **Speed** — Tab completion is faster than typing
- **Accuracy** — Fewer typos and invalid inputs
- **Professionalism** — Expected feature of mature CLIs

### Functional Objectives

1. Generate Bash, Zsh, and Fish completions
2. Dynamic completion for operation IDs
3. Dynamic completion for table keys
4. Path completion for file arguments
5. Install command for completions

### Implementation

#### File: `src/codeintel/cli/cli_completions.py`

```python
"""Shell completion generation for CodeIntel CLI.

Provides completion scripts for Bash, Zsh, and Fish shells,
including dynamic completions for operations and datasets.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path


class Shell(Enum):
    """Supported shells for completion."""

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"


# Bash completion script template
BASH_COMPLETION_TEMPLATE = '''
# CodeIntel CLI completion for Bash
# Generated automatically - do not edit

_codeintel_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Top-level commands
    local commands="op dataset build graph docs config storage"
    
    # Subcommands by parent
    local op_commands="list call"
    local dataset_commands="list describe verify"
    local build_commands="status run clean"
    local graph_commands="stats query export"
    local docs_commands="build serve"
    local config_commands="show set"
    local storage_commands="init migrate status"
    
    # Dynamic completions
    case "${prev}" in
        codeintel)
            COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
            return 0
            ;;
        op)
            COMPREPLY=( $(compgen -W "${op_commands}" -- ${cur}) )
            return 0
            ;;
        call)
            # Complete operation IDs dynamically
            local ops=$(codeintel op list --format=json 2>/dev/null | jq -r '.[].id' 2>/dev/null)
            COMPREPLY=( $(compgen -W "${ops}" -- ${cur}) )
            return 0
            ;;
        dataset)
            COMPREPLY=( $(compgen -W "${dataset_commands}" -- ${cur}) )
            return 0
            ;;
        describe|verify)
            # Complete table keys dynamically
            local tables=$(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)
            COMPREPLY=( $(compgen -W "${tables}" -- ${cur}) )
            return 0
            ;;
        build)
            COMPREPLY=( $(compgen -W "${build_commands}" -- ${cur}) )
            return 0
            ;;
        graph)
            COMPREPLY=( $(compgen -W "${graph_commands}" -- ${cur}) )
            return 0
            ;;
        docs)
            COMPREPLY=( $(compgen -W "${docs_commands}" -- ${cur}) )
            return 0
            ;;
        config)
            COMPREPLY=( $(compgen -W "${config_commands}" -- ${cur}) )
            return 0
            ;;
        storage)
            COMPREPLY=( $(compgen -W "${storage_commands}" -- ${cur}) )
            return 0
            ;;
        --db-path|--repo-root|--build-dir)
            # Path completion
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "text json" -- ${cur}) )
            return 0
            ;;
    esac
    
    # Global options
    local global_opts="--help --version --format --verbose --quiet"
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${global_opts}" -- ${cur}) )
        return 0
    fi
}

complete -F _codeintel_completions codeintel
'''

# Zsh completion script template
ZSH_COMPLETION_TEMPLATE = '''
#compdef codeintel

# CodeIntel CLI completion for Zsh
# Generated automatically - do not edit

_codeintel() {
    local -a commands
    commands=(
        'op:Operation management'
        'dataset:Dataset inspection'
        'build:Build management'
        'graph:Graph operations'
        'docs:Documentation'
        'config:Configuration'
        'storage:Storage management'
    )
    
    local -a op_commands
    op_commands=(
        'list:List available operations'
        'call:Call an operation'
    )
    
    local -a dataset_commands
    dataset_commands=(
        'list:List available datasets'
        'describe:Describe a dataset'
        'verify:Verify dataset integrity'
    )
    
    local -a build_commands
    build_commands=(
        'status:Show build status'
        'run:Run build targets'
        'clean:Clean build artifacts'
    )
    
    _arguments -C \\
        '1: :->command' \\
        '2: :->subcommand' \\
        '3: :->argument' \\
        '--help[Show help]' \\
        '--version[Show version]' \\
        '--format[Output format]:format:(text json)' \\
        '--verbose[Verbose output]' \\
        '--quiet[Quiet output]'
    
    case $state in
        command)
            _describe -t commands 'command' commands
            ;;
        subcommand)
            case $words[2] in
                op)
                    _describe -t op_commands 'op command' op_commands
                    ;;
                dataset)
                    _describe -t dataset_commands 'dataset command' dataset_commands
                    ;;
                build)
                    _describe -t build_commands 'build command' build_commands
                    ;;
            esac
            ;;
        argument)
            case $words[2]:$words[3] in
                op:call)
                    # Dynamic operation completion
                    local -a operations
                    operations=(${(f)"$(codeintel op list --format=json 2>/dev/null | jq -r '.[].id' 2>/dev/null)"})
                    _describe -t operations 'operation' operations
                    ;;
                dataset:describe|dataset:verify)
                    # Dynamic table completion
                    local -a tables
                    tables=(${(f)"$(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)"})
                    _describe -t tables 'table' tables
                    ;;
            esac
            ;;
    esac
}

_codeintel "$@"
'''

# Fish completion script template
FISH_COMPLETION_TEMPLATE = '''
# CodeIntel CLI completion for Fish
# Generated automatically - do not edit

# Disable file completion by default
complete -c codeintel -f

# Top-level commands
complete -c codeintel -n "__fish_use_subcommand" -a "op" -d "Operation management"
complete -c codeintel -n "__fish_use_subcommand" -a "dataset" -d "Dataset inspection"
complete -c codeintel -n "__fish_use_subcommand" -a "build" -d "Build management"
complete -c codeintel -n "__fish_use_subcommand" -a "graph" -d "Graph operations"
complete -c codeintel -n "__fish_use_subcommand" -a "docs" -d "Documentation"
complete -c codeintel -n "__fish_use_subcommand" -a "config" -d "Configuration"
complete -c codeintel -n "__fish_use_subcommand" -a "storage" -d "Storage management"

# Op subcommands
complete -c codeintel -n "__fish_seen_subcommand_from op" -a "list" -d "List operations"
complete -c codeintel -n "__fish_seen_subcommand_from op" -a "call" -d "Call an operation"

# Dataset subcommands
complete -c codeintel -n "__fish_seen_subcommand_from dataset" -a "list" -d "List datasets"
complete -c codeintel -n "__fish_seen_subcommand_from dataset" -a "describe" -d "Describe dataset"
complete -c codeintel -n "__fish_seen_subcommand_from dataset" -a "verify" -d "Verify dataset"

# Build subcommands
complete -c codeintel -n "__fish_seen_subcommand_from build" -a "status" -d "Build status"
complete -c codeintel -n "__fish_seen_subcommand_from build" -a "run" -d "Run build"
complete -c codeintel -n "__fish_seen_subcommand_from build" -a "clean" -d "Clean build"

# Dynamic operation completion for 'op call'
complete -c codeintel -n "__fish_seen_subcommand_from call" -a "(codeintel op list --format=json 2>/dev/null | jq -r '.[].id' 2>/dev/null)"

# Dynamic table completion for 'dataset describe/verify'
complete -c codeintel -n "__fish_seen_subcommand_from describe" -a "(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)"
complete -c codeintel -n "__fish_seen_subcommand_from verify" -a "(codeintel dataset list --format=json 2>/dev/null | jq -r '.[].table_key' 2>/dev/null)"

# Global options
complete -c codeintel -l help -d "Show help"
complete -c codeintel -l version -d "Show version"
complete -c codeintel -l format -d "Output format" -a "text json"
complete -c codeintel -l verbose -d "Verbose output"
complete -c codeintel -l quiet -d "Quiet output"
'''


def generate_completion(shell: Shell) -> str:
    """Generate completion script for a shell.

    Parameters
    ----------
    shell
        Target shell.

    Returns
    -------
    str
        Completion script content.
    """
    templates = {
        Shell.BASH: BASH_COMPLETION_TEMPLATE,
        Shell.ZSH: ZSH_COMPLETION_TEMPLATE,
        Shell.FISH: FISH_COMPLETION_TEMPLATE,
    }
    return templates[shell].strip()


def get_completion_install_path(shell: Shell) -> Path:
    """Get the recommended installation path for completions.

    Parameters
    ----------
    shell
        Target shell.

    Returns
    -------
    Path
        Recommended installation path.
    """
    home = Path.home()
    
    paths = {
        Shell.BASH: home / ".bash_completion.d" / "codeintel",
        Shell.ZSH: home / ".zsh" / "completions" / "_codeintel",
        Shell.FISH: home / ".config" / "fish" / "completions" / "codeintel.fish",
    }
    return paths[shell]


def install_completion(shell: Shell, *, force: bool = False) -> Path:
    """Install completion script for a shell.

    Parameters
    ----------
    shell
        Target shell.
    force
        Overwrite existing file.

    Returns
    -------
    Path
        Path where completion was installed.

    Raises
    ------
    FileExistsError
        If completion exists and force is False.
    """
    path = get_completion_install_path(shell)
    
    if path.exists() and not force:
        raise FileExistsError(f"Completion already exists: {path}")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_completion(shell))
    
    return path


def detect_shell() -> Shell | None:
    """Detect the current shell.

    Returns
    -------
    Shell | None
        Detected shell or None if unknown.
    """
    import os
    
    shell_path = os.environ.get("SHELL", "")
    
    if "bash" in shell_path:
        return Shell.BASH
    if "zsh" in shell_path:
        return Shell.ZSH
    if "fish" in shell_path:
        return Shell.FISH
    
    return None


__all__ = [
    "Shell",
    "detect_shell",
    "generate_completion",
    "get_completion_install_path",
    "install_completion",
]
```

#### CLI Commands for Completions

Add to `cyclopts_app.py`:

```python
@app.command()
def completion(
    shell: Annotated[
        str | None,
        Parameter(help="Shell type (bash, zsh, fish). Auto-detected if not specified."),
    ] = None,
    install: Annotated[
        bool,
        Parameter(help="Install completion to standard location"),
    ] = False,
    force: Annotated[
        bool,
        Parameter(help="Overwrite existing completion file"),
    ] = False,
) -> None:
    """Generate or install shell completions.

    Examples
    --------
    # Print completion script for current shell
    codeintel completion

    # Print completion for specific shell
    codeintel completion --shell=zsh

    # Install completion
    codeintel completion --install

    # Force reinstall
    codeintel completion --install --force
    """
    from codeintel.cli.cli_completions import (
        Shell,
        detect_shell,
        generate_completion,
        install_completion,
    )

    # Determine shell
    if shell:
        target_shell = Shell(shell)
    else:
        target_shell = detect_shell()
        if target_shell is None:
            console.print("Could not detect shell. Use --shell to specify.", style="error")
            raise SystemExit(1)

    if install:
        try:
            path = install_completion(target_shell, force=force)
            console.print(f"Completion installed to: {path}", style="success")
            console.print(f"\nTo activate, add to your shell config:")
            if target_shell == Shell.BASH:
                console.print(f"  source {path}")
            elif target_shell == Shell.ZSH:
                console.print(f"  fpath=({path.parent} $fpath)")
                console.print("  autoload -Uz compinit && compinit")
        except FileExistsError as e:
            console.print(str(e), style="error")
            console.print("Use --force to overwrite.")
            raise SystemExit(1)
    else:
        # Print completion script to stdout
        print(generate_completion(target_shell))
```

---

## Phase 3.6: Resilience and Retry Layer

### Value Proposition

CLI operations that interact with external services can fail transiently:
- Network timeouts
- Service temporarily unavailable
- Rate limiting

A resilience layer provides:
- **Automatic recovery** — Retry transient failures automatically
- **Graceful degradation** — Fallback to cached data when possible
- **User feedback** — Clear communication about retries
- **Configurability** — Adjustable retry policies

### Functional Objectives

1. Create configurable retry policies
2. Implement exponential backoff with jitter
3. Integrate with middleware pattern
4. Provide circuit breaker for persistent failures
5. Cache fallback for read operations

### Implementation

#### File: `src/codeintel/cli/cli_resilience.py`

```python
"""Resilience and retry infrastructure for CLI operations.

Provides retry logic, circuit breakers, and fallback mechanisms
for operations that may experience transient failures.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

LOG = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""

    pass


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open.

    Parameters
    ----------
    message
        Error message.
    retry_after
        Seconds until circuit may close.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        """Initialize circuit open error."""
        super().__init__(message)
        self.retry_after = retry_after


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

    Parameters
    ----------
    max_attempts
        Maximum number of attempts (including initial).
    initial_delay
        Initial delay between retries in seconds.
    max_delay
        Maximum delay between retries in seconds.
    backoff_factor
        Multiplier for exponential backoff.
    jitter
        Random jitter factor (0.0 to 1.0).
    retryable_exceptions
        Exception types that should trigger retry.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] = (RetryableError,)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Parameters
        ----------
        attempt
            Attempt number (0-indexed).

        Returns
        -------
        float
            Delay in seconds.
        """
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing repeated failures.

    Parameters
    ----------
    failure_threshold
        Number of failures before opening circuit.
    recovery_timeout
        Seconds before attempting recovery.
    half_open_max_calls
        Max calls in half-open state before deciding.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                # Recovered
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        
        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery, reopen
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns
        -------
        bool
            True if request is allowed.

        Raises
        ------
        CircuitOpenError
            If circuit is open.
        """
        state = self.state  # This may transition from OPEN to HALF_OPEN
        
        if state == CircuitState.OPEN:
            retry_after = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
            raise CircuitOpenError(
                "Circuit breaker is open",
                retry_after=max(0, retry_after),
            )
        
        return True


@dataclass
class RetryContext:
    """Context passed to retry callbacks.

    Parameters
    ----------
    attempt
        Current attempt number (0-indexed).
    exception
        Exception that triggered retry (if any).
    delay
        Delay before next attempt.
    """

    attempt: int
    exception: Exception | None
    delay: float


def with_retry(
    policy: RetryPolicy | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    on_retry: Callable[[RetryContext], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add retry logic to a function.

    Parameters
    ----------
    policy
        Retry policy (uses default if None).
    circuit_breaker
        Optional circuit breaker.
    on_retry
        Callback invoked before each retry.

    Returns
    -------
    Callable
        Decorated function.

    Example
    -------
    ```python
    @with_retry(RetryPolicy(max_attempts=3))
    def fetch_data(url: str) -> dict:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    ```
    """
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Check circuit breaker
            if circuit_breaker:
                circuit_breaker.allow_request()

            last_exception: Exception | None = None
            
            for attempt in range(policy.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                    
                except policy.retryable_exceptions as e:
                    last_exception = e
                    
                    # Record failure
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    # Check if we have retries left
                    if attempt < policy.max_attempts - 1:
                        delay = policy.calculate_delay(attempt)
                        
                        # Invoke callback
                        if on_retry:
                            ctx = RetryContext(
                                attempt=attempt,
                                exception=e,
                                delay=delay,
                            )
                            on_retry(ctx)
                        
                        LOG.warning(
                            "Retrying after error",
                            extra={
                                "attempt": attempt + 1,
                                "max_attempts": policy.max_attempts,
                                "delay": delay,
                                "error": str(e),
                            },
                        )
                        
                        time.sleep(delay)
                    else:
                        LOG.error(
                            "Max retries exceeded",
                            extra={
                                "attempts": policy.max_attempts,
                                "error": str(e),
                            },
                        )

            # All retries exhausted
            assert last_exception is not None
            raise last_exception

        return wrapper
    return decorator


class RetryMiddleware:
    """Middleware that adds retry logic to operations.

    Parameters
    ----------
    policy
        Default retry policy.
    circuit_breaker
        Shared circuit breaker.
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize retry middleware."""
        self._policy = policy or RetryPolicy()
        self._circuit_breaker = circuit_breaker

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Check circuit breaker before operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context for after_invoke.
        """
        if self._circuit_breaker:
            self._circuit_breaker.allow_request()
        return {"op_id": op_id, "start_time": time.monotonic()}

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Record success after operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        del result  # Unused
        if self._circuit_breaker:
            self._circuit_breaker.record_success()

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record failure after operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        del exc  # Unused
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()


# Default policies for common scenarios
DEFAULT_NETWORK_POLICY = RetryPolicy(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, RetryableError),
)

DEFAULT_STORAGE_POLICY = RetryPolicy(
    max_attempts=2,
    initial_delay=0.5,
    max_delay=5.0,
    backoff_factor=2.0,
    retryable_exceptions=(RetryableError,),
)


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "DEFAULT_NETWORK_POLICY",
    "DEFAULT_STORAGE_POLICY",
    "RetryContext",
    "RetryMiddleware",
    "RetryPolicy",
    "RetryableError",
    "with_retry",
]
```

### Usage Examples

```python
# Decorator usage
@with_retry(DEFAULT_NETWORK_POLICY)
def fetch_remote_config(url: str) -> dict[str, Any]:
    """Fetch configuration from remote server."""
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()

# Middleware usage in cyclopts_ops.py
from codeintel.cli.cli_resilience import RetryMiddleware, CircuitBreaker

# Create shared circuit breaker for external service
_EXTERNAL_SERVICE_CIRCUIT = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
)

# Add to middleware stack
stack = get_middleware_stack()
stack.add(RetryMiddleware(circuit_breaker=_EXTERNAL_SERVICE_CIRCUIT))
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Priority |
|-------|----------|--------------|----------|
| 3.1 Unified Rendering | 2-3 days | None | High |
| 3.2 Handler Migration | 5-7 days | 3.1 | High |
| 3.3 Config Validation | 2-3 days | None | Medium |
| 3.4 Test Refactoring | 4-5 days | 3.2 | High |
| 3.5 Shell Completions | 1-2 days | None | Low |
| 3.6 Resilience Layer | 2-3 days | None | Medium |

**Total estimated time: 16-23 days**

### Recommended Order

1. **Phase 3.1** (Unified Rendering) — Foundation for consistent output
2. **Phase 3.3** (Config Validation) — Can proceed in parallel
3. **Phase 3.2** (Handler Migration) — Largest effort, depends on 3.1
4. **Phase 3.4** (Test Refactoring) — Depends on 3.2
5. **Phase 3.5** (Shell Completions) — Independent, quick win
6. **Phase 3.6** (Resilience Layer) — Independent, adds robustness

---

## Success Metrics

### Code Quality

- [ ] All handlers return `CliResult` (100% migration)
- [ ] Zero `monkeypatch` usage in CLI tests
- [ ] Configuration validated at startup
- [ ] All tests pass in parallel mode

### User Experience

- [ ] Shell completions available for Bash, Zsh, Fish
- [ ] Consistent output formatting across all commands
- [ ] Clear error messages with fix suggestions
- [ ] Retry feedback for transient failures

### Operational Excellence

- [ ] Circuit breaker prevents cascading failures
- [ ] Structured logging with correlation IDs
- [ ] Graceful degradation when services unavailable
- [ ] Sub-second startup time maintained

---

## Appendix: File Manifest

### New Files

| File | Purpose |
|------|---------|
| `src/codeintel/cli/cli_render.py` | Unified output rendering |
| `src/codeintel/cli/cli_config_schema.py` | Configuration validation |
| `src/codeintel/cli/cli_completions.py` | Shell completion generation |
| `src/codeintel/cli/cli_resilience.py` | Retry and circuit breaker |
| `tests/cli/_doubles/__init__.py` | Test doubles |
| `tests/cli/_doubles/contexts.py` | Test context builders |

### Modified Files

| File | Changes |
|------|---------|
| `src/codeintel/cli/result_types.py` | Add build/graph result types |
| `src/codeintel/cli/build_handlers.py` | Migrate to CliResult |
| `src/codeintel/cli/graphs_handlers.py` | Migrate to CliResult |
| `src/codeintel/cli/cyclopts_app.py` | Add completion command |
| `src/codeintel/cli/cyclopts_ops.py` | Add resilience middleware |
| `tests/cli/conftest.py` | Remove monkeypatch, add doubles |

---

*End of Phase 3 Implementation Plan*

