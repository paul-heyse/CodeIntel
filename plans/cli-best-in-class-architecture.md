# CLI Best-in-Class Architecture

> **Status**: Proposed  
> **Author**: Architecture Review  
> **Date**: 2025-12-10  
> **Scope**: `src/codeintel/cli/`

---

## Executive Summary

This document specifies a best-in-class CLI architecture that maximizes type safety, minimizes boilerplate, and provides clean extensibility for future plugin and build system integration. The design is evolutionary—it can be implemented incrementally alongside the existing codebase.

**Core Insight**: Commands should be the single source of truth for both input types (CLI arguments) AND output types (results). The current architecture loses type safety when extracting parameters from `HandlerContext.param_*()` methods.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Package Structure](#3-package-structure)
4. [Core Types (`core/`)](#4-core-types-core)
5. [Commands (`commands/`)](#5-commands-commands)
6. [Dependencies (`deps/`)](#6-dependencies-deps)
7. [Execution (`execution/`)](#7-execution-execution)
8. [Rendering (`rendering/`)](#8-rendering-rendering)
9. [Errors (`errors/`)](#9-errors-errors)
10. [Testing Patterns](#10-testing-patterns)
11. [Migration Strategy](#11-migration-strategy)
12. [Extension Points for Plugins/Build](#12-extension-points-for-pluginsbuild)

---

## 1. Design Principles

### 1.1 Type Safety End-to-End

Parameters flow from CLI parsing through execution without losing type information:

```
CLI Args → Command Dataclass → Handler Method → Result Type → Renderer
   ↓              ↓                  ↓               ↓
 strings     typed fields      self.field      auto-serialized
```

**Anti-pattern** (current):
```python
def handler(ctx: HandlerContext) -> CliResult[JobsListResult]:
    limit = ctx.param_int("limt", 20)  # Typo compiles! Returns 20 silently.
```

**Pattern** (proposed):
```python
@dataclass(frozen=True)
class ListJobs(Command[JobsListResult]):
    limit: int = 20
    
    def execute(self, deps: Deps) -> CliResult[JobsListResult]:
        return self._list(self.limit)  # self.limit is int, typos are compile errors
```

### 1.2 Explicit Dependencies

Dependencies are declared, not discovered at runtime:

```python
# Bad: Hidden dependencies, hard to test
def handler(ctx: HandlerContext):
    manager = get_job_manager()  # Global singleton!
    gateway = ctx.gateway  # Lazy-loaded, might fail

# Good: Explicit dependencies, easy to test  
def execute(self, deps: Deps) -> CliResult[T]:
    jobs = deps.jobs.list(limit=self.limit)  # deps.jobs is injected
```

### 1.3 Commands as Self-Contained Units

A command class contains everything needed to understand its behavior:
- Input schema (dataclass fields)
- Output type (generic parameter)
- Execution logic (method)
- Metadata (docstring, decorators)

### 1.4 Automatic Serialization

Result types should not require manual `to_dict()` implementations:

```python
# Bad: 50+ lines of boilerplate per result type
@dataclass
class JobStatusResult:
    job_id: str
    status: str
    created_at: str | None = None
    
    def to_dict(self) -> dict[str, object]:
        result = {"job_id": self.job_id, "status": self.status}
        if self.created_at is not None:
            result["created_at"] = self.created_at
        return result

# Good: Automatic, zero boilerplate
@result_type
@dataclass(frozen=True)
class JobStatus:
    job_id: str
    status: str
    created_at: str | None = None
    # to_dict() auto-generated, None fields omitted
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Entry                                   │
│                         (cyclopts App routing)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           @cli_command                                   │
│         (decorator: validates, registers, generates __call__)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Command[TResult] Dataclass                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │  Input Fields   │  │   Output Type    │  │    execute(deps)       │  │
│  │  (CLI args)     │  │   (TResult)      │  │    -> CliResult[T]     │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Deps                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ storage  │  │  config  │  │  logger  │  │   jobs   │  │ serving  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CliResult[TResult]                               │
│              (success/failure + data + RFC 9457 errors)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          UnifiedRenderer                                 │
│              (format negotiation, TTY detection, output)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Package Structure

```
src/codeintel/cli/
├── __init__.py
├── core/                      # Core types and protocols
│   ├── __init__.py
│   ├── command.py             # Command[T] base, @cli_command
│   ├── results.py             # CliResult[T], @result_type
│   ├── result_types.py        # Generic result types (ListResult, etc.)
│   └── serialization.py       # Auto-serialization logic
│
├── deps/                      # Dependency injection
│   ├── __init__.py
│   ├── protocols.py           # StorageAccess, RuntimeAccess, etc.
│   ├── container.py           # Deps dataclass, DepsBuilder
│   └── providers.py           # Lazy providers for resources
│
├── commands/                  # Command implementations (one file per group)
│   ├── __init__.py
│   ├── _common.py             # SharedFlags, common field types
│   ├── jobs.py                # jobs.list, jobs.status, etc.
│   ├── build.py               # build.run, build.status, etc.
│   ├── datasets.py            # datasets.list, datasets.describe, etc.
│   └── ...
│
├── execution/                 # Execution infrastructure
│   ├── __init__.py
│   ├── registry.py            # OperationSpec, OperationRegistry
│   ├── bootstrap.py           # CLI initialization
│   └── middleware.py          # Logging, metrics, error handling
│
├── rendering/                 # Output rendering
│   ├── __init__.py
│   ├── service.py             # UnifiedRenderer
│   ├── table.py               # TableSpec, column definitions
│   └── types.py               # OutputFormat, RenderContext
│
├── errors/                    # Error handling
│   ├── __init__.py
│   ├── problem_detail.py      # RFC 9457 ProblemDetail
│   ├── taxonomy.py            # Error code enums
│   └── factory.py             # fail_* convenience functions
│
├── config/                    # Configuration (unchanged)
├── completions/               # Shell completions (unchanged)
├── introspection/             # Help/discovery (unchanged)
└── plugins/                   # Plugin system (unchanged)
```

---

## 4. Core Types (`core/`)

### 4.1 Command Base Class (`core/command.py`)

The `Command[T]` base establishes the contract for all CLI commands:

```python
"""Command base class and @cli_command decorator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from codeintel.cli.core.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.deps import Deps

T = TypeVar("T")


class Command(ABC, Generic[T]):
    """Base class for CLI commands.
    
    Subclasses define:
    - Input fields as dataclass attributes
    - Output type as the generic parameter T
    - Execution logic in execute()
    
    Example
    -------
    @cli_command("jobs.list")
    @dataclass(frozen=True)
    class ListJobs(Command[JobsListResult]):
        status: JobStatus | None = None
        limit: int = 20
        
        def execute(self, deps: Deps) -> CliResult[JobsListResult]:
            jobs = deps.jobs.list(status=self.status, limit=self.limit)
            return CliResult.ok(JobsListResult.from_models(jobs))
    """
    
    # Class-level metadata (populated by @cli_command)
    __operation_id__: ClassVar[str]
    __require_storage__: ClassVar[bool] = False
    __require_runtime__: ClassVar[bool] = False
    
    @abstractmethod
    def execute(self, deps: Deps) -> CliResult[T]:
        """Execute the command with provided dependencies.
        
        Parameters
        ----------
        deps
            Injected dependencies (storage, config, etc.).
            
        Returns
        -------
        CliResult[T]
            Success with result data or failure with ProblemDetail.
        """
        ...
```

### 4.2 The `@cli_command` Decorator (`core/command.py` continued)

```python
from codeintel.cli.execution.registry import OperationSpec, get_registry


def cli_command(
    operation_id: str,
    *,
    require_storage: bool = False,
    require_runtime: bool = False,
    hidden: bool = False,
    tags: tuple[str, ...] = (),
) -> Callable[[type[Command[T]]], type[Command[T]]]:
    """Decorate a Command subclass to register it and generate __call__.
    
    The decorator:
    1. Validates the class is a frozen dataclass and Command subclass
    2. Sets class-level metadata (__operation_id__, etc.)
    3. Generates __call__ that builds Deps and invokes execute()
    4. Registers the operation with OperationRegistry
    
    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "jobs.list", "build.run").
    require_storage
        Whether execute() needs storage access.
    require_runtime
        Whether execute() needs runtime resolution.
    hidden
        If True, hide from help output.
    tags
        Optional tags for filtering/categorization.
        
    Example
    -------
    @cli_command("jobs.list", require_storage=False)
    @dataclass(frozen=True)
    class ListJobs(Command[JobsListResult]):
        limit: int = 20
        
        def execute(self, deps: Deps) -> CliResult[JobsListResult]:
            ...
    """
    def decorator(cls: type[Command[T]]) -> type[Command[T]]:
        # Validate
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        if not issubclass(cls, Command):
            raise TypeError(f"{cls.__name__} must subclass Command")
        
        # Set metadata
        cls.__operation_id__ = operation_id
        cls.__require_storage__ = require_storage
        cls.__require_runtime__ = require_runtime
        
        # Generate __call__
        def __call__(self: Command[T]) -> None:
            _execute_command(self)
        
        cls.__call__ = __call__  # type: ignore[attr-defined]
        
        # Register operation
        _register_command(cls, operation_id, hidden=hidden, tags=tags)
        
        return cls
    
    return decorator


def _execute_command(cmd: Command[T]) -> None:
    """Execute a command instance (generated __call__ body)."""
    from codeintel.cli.deps import DepsBuilder
    from codeintel.cli.rendering import get_renderer
    
    # Build dependencies based on command requirements
    builder = DepsBuilder()
    if cmd.__require_storage__:
        builder.with_storage()
    if cmd.__require_runtime__:
        builder.with_runtime()
    
    # Extract shared flags if present
    output_format = _extract_output_format(cmd)
    
    with builder.build() as deps:
        result = cmd.execute(deps)
    
    # Render and exit
    renderer = get_renderer(output_format)
    exit_code = renderer.render_result(result)
    if exit_code != 0:
        sys.exit(exit_code)
```

### 4.3 Auto-Serializing Results (`core/results.py`)

```python
"""Result types with automatic serialization."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from codeintel.cli.errors import ProblemDetail


@runtime_checkable
class SerializableResult(Protocol):
    """Protocol for result types that can serialize to dict."""
    
    def to_dict(self) -> dict[str, object]: ...


def result_type(cls: type[T]) -> type[T]:
    """Decorator that adds automatic to_dict() to a dataclass.
    
    The generated to_dict():
    - Includes all non-None fields
    - Recursively serializes nested result types
    - Handles lists and dicts of result types
    
    Example
    -------
    @result_type
    @dataclass(frozen=True)
    class JobStatus:
        job_id: str
        status: str
        created_at: str | None = None  # Omitted from dict if None
        
    >>> JobStatus("j1", "running").to_dict()
    {"job_id": "j1", "status": "running"}
    
    >>> JobStatus("j1", "done", "2025-01-01").to_dict()
    {"job_id": "j1", "status": "done", "created_at": "2025-01-01"}
    """
    if not is_dataclass(cls):
        raise TypeError(f"@result_type requires a dataclass, got {cls}")
    
    def to_dict(self: object) -> dict[str, object]:
        result: dict[str, object] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if value is None:
                continue  # Omit None fields
            result[f.name] = _serialize_value(value)
        return result
    
    cls.to_dict = to_dict  # type: ignore[attr-defined]
    return cls


def _serialize_value(value: object) -> object:
    """Recursively serialize a value for JSON output."""
    if isinstance(value, SerializableResult):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize_value(getattr(value, f.name)) 
                for f in fields(value)}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)  # Fallback for enums, paths, etc.


@dataclass
class CliResult[T]:
    """Structured result from a CLI command.
    
    Unchanged from current implementation - this type is already excellent.
    """
    
    success: bool
    data: T | None = None
    error: ProblemDetail | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: T, *, metadata: dict[str, object] | None = None) -> Self:
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata or {})
    
    @classmethod
    def fail(cls, error: ProblemDetail, *, warnings: list[str] | None = None) -> Self:
        """Create failed result."""
        return cls(success=False, error=error, warnings=warnings or [])
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for JSON output."""
        result: dict[str, object] = {"success": self.success}
        if self.data is not None:
            result["data"] = _serialize_value(self.data)
        if self.error is not None:
            result["error"] = self.error.to_dict()
        if self.warnings:
            result["warnings"] = self.warnings
        if self.metadata:
            result["metadata"] = self.metadata
        return result
```

### 4.4 Generic Result Types (`core/result_types.py`)

```python
"""Reusable generic result types.

These cover ~70% of command output patterns, reducing boilerplate significantly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from codeintel.cli.core.results import result_type

T = TypeVar("T")


@result_type
@dataclass(frozen=True)
class ListResult(Generic[T]):
    """Generic list result with count.
    
    Use for: jobs.list, datasets.list, plugins.list, operations.list, etc.
    """
    items: list[T]
    count: int
    
    @classmethod
    def from_items(cls, items: list[T]) -> ListResult[T]:
        """Create from item list, auto-computing count."""
        return cls(items=items, count=len(items))


@result_type
@dataclass(frozen=True)
class StatusResult:
    """Generic status result.
    
    Use for: health checks, connectivity checks, etc.
    """
    status: str  # "ok", "warning", "error", "unknown"
    message: str
    details: dict[str, object] | None = None


@result_type
@dataclass(frozen=True)
class ActionResult:
    """Result from an action that modifies state.
    
    Use for: jobs.cancel, build.run, storage.migrate, etc.
    """
    action: str
    success: bool
    affected_count: int = 0
    message: str | None = None


@result_type
@dataclass(frozen=True)
class ExportResult:
    """Result from an export/generation operation.
    
    Use for: docs.generate, storage.export, datasets.snapshot, etc.
    """
    output_path: str
    item_count: int
    duration_seconds: float | None = None
```

---

## 5. Commands (`commands/`)

### 5.1 Shared Flags (`commands/_common.py`)

```python
"""Shared CLI flag definitions and mixins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from codeintel.cli.rendering.types import OutputFormat


# Annotated type aliases for CLI parameters
Verbose = Annotated[
    int,
    Parameter(name=["--verbose", "-v"], count=True,
              help="Increase verbosity (0=warnings, 1=info, 2=debug)."),
]

ProjectRoot = Annotated[
    Path | None,
    Parameter(name=["--root", "-r"],
              help="Explicit project root directory."),
]

OutputFmt = Annotated[
    OutputFormat,
    Parameter(name="--output-format", show_choices=True,
              help="Output format."),
]

JsonFlag = Annotated[
    bool,
    Parameter(name="--json", negative=(),
              help="Alias for --output-format json."),
]


@dataclass(frozen=True)
class SharedFlags:
    """Unified infrastructure flags for all commands.
    
    Include this as a field in command dataclasses to get consistent
    project root, output format, JSON flag, and verbosity parameters.
    
    Example
    -------
    @cli_command("my.command")
    @dataclass(frozen=True)
    class MyCommand(Command[MyResult]):
        flags: SharedFlags = field(default_factory=SharedFlags,
                                   metadata=SHARED_FLAGS_METADATA)
        name: str = "default"
    """
    project_root: ProjectRoot = None
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False
    verbose: Verbose = 0
    
    def resolve_output_format(self) -> OutputFormat:
        """Resolve output format with JSON flag taking precedence."""
        if self.json:
            return OutputFormat.JSON
        return self.output_format


SHARED_FLAGS_METADATA: dict[str, Parameter] = {"parameter": Parameter(name="*")}
"""Metadata for SharedFlags to enable Cyclopts nested parameter flattening."""


def shared_flags_field() -> SharedFlags:
    """Create a SharedFlags field with Cyclopts metadata."""
    return field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)
```

### 5.2 Example Command Group (`commands/jobs.py`)

```python
"""Job management commands.

Commands for listing, inspecting, and managing background jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.commands._common import SharedFlags, shared_flags_field
from codeintel.cli.core.command import Command, cli_command
from codeintel.cli.core.results import CliResult, result_type
from codeintel.cli.core.result_types import ListResult, ActionResult
from codeintel.cli.errors.factory import (
    fail_job_not_found,
    fail_job_not_completed,
)

if TYPE_CHECKING:
    from codeintel.cli.deps import Deps
    from codeintel.cli.jobs import JobStatus


# ---------------------------------------------------------------------------
# Result Types (domain-specific, auto-serialized)
# ---------------------------------------------------------------------------

@result_type
@dataclass(frozen=True)
class JobInfo:
    """Information about a single job."""
    job_id: str
    operation_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@result_type
@dataclass(frozen=True)
class JobOutput:
    """Output from a completed job."""
    job_id: str
    has_output: bool
    output: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@cli_command("jobs.list", require_storage=False)
@dataclass(frozen=True)
class ListJobs(Command[ListResult[JobInfo]]):
    """List background jobs.
    
    Filter by status and limit results.
    """
    
    status: JobStatus | None = None
    limit: int = 20
    flags: SharedFlags = shared_flags_field()
    
    def execute(self, deps: Deps) -> CliResult[ListResult[JobInfo]]:
        """Execute job listing."""
        jobs = deps.jobs.list(status=self.status, limit=self.limit)
        
        items = [
            JobInfo(
                job_id=j.job_id,
                operation_id=j.operation_id,
                status=j.status.value,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                error=j.error,
            )
            for j in jobs
        ]
        
        return CliResult.ok(ListResult.from_items(items))


@cli_command("jobs.status", require_storage=False)
@dataclass(frozen=True)
class GetJobStatus(Command[JobInfo]):
    """Get detailed status of a specific job."""
    
    job_id: str
    flags: SharedFlags = shared_flags_field()
    
    def execute(self, deps: Deps) -> CliResult[JobInfo]:
        """Execute status lookup."""
        job = deps.jobs.get(self.job_id)
        
        if job is None:
            return fail_job_not_found(self.job_id)
        
        return CliResult.ok(JobInfo(
            job_id=job.job_id,
            operation_id=job.operation_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        ))


@cli_command("jobs.output", require_storage=False)
@dataclass(frozen=True)
class GetJobOutput(Command[JobOutput]):
    """Get output from a completed job."""
    
    job_id: str
    flags: SharedFlags = shared_flags_field()
    
    def execute(self, deps: Deps) -> CliResult[JobOutput]:
        """Execute output retrieval."""
        job = deps.jobs.get(self.job_id)
        
        if job is None:
            return fail_job_not_found(self.job_id)
        
        if job.status != JobStatus.COMPLETED:
            return fail_job_not_completed(self.job_id, job.status.value)
        
        output = deps.jobs.get_output(self.job_id)
        
        return CliResult.ok(JobOutput(
            job_id=self.job_id,
            has_output=output is not None,
            output=output,
        ))


@cli_command("jobs.cancel", require_storage=False)
@dataclass(frozen=True)
class CancelJob(Command[ActionResult]):
    """Cancel a running job."""
    
    job_id: str
    flags: SharedFlags = shared_flags_field()
    
    def execute(self, deps: Deps) -> CliResult[ActionResult]:
        """Execute job cancellation."""
        cancelled = deps.jobs.cancel(self.job_id)
        
        if not cancelled:
            return CliResult.ok(ActionResult(
                action="cancel",
                success=False,
                message=f"Could not cancel job {self.job_id}",
            ))
        
        return CliResult.ok(ActionResult(
            action="cancel",
            success=True,
            affected_count=1,
            message=f"Cancelled job {self.job_id}",
        ))
```

### 5.3 Command with Storage/Runtime (`commands/datasets.py`)

```python
"""Dataset commands.

Commands that require storage access demonstrate dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.commands._common import SharedFlags, shared_flags_field
from codeintel.cli.core.command import Command, cli_command
from codeintel.cli.core.results import CliResult, result_type
from codeintel.cli.core.result_types import ListResult
from codeintel.cli.errors.factory import fail_dataset_not_found

if TYPE_CHECKING:
    from codeintel.cli.deps import Deps


@result_type
@dataclass(frozen=True)
class DatasetInfo:
    """Dataset metadata."""
    table_key: str
    name: str
    row_count: int | None = None
    description: str | None = None


@result_type
@dataclass(frozen=True)
class DatasetSchema:
    """Detailed dataset schema."""
    table_key: str
    name: str
    columns: list[dict[str, str | bool]]
    row_count: int | None = None
    upstream_dependencies: list[str] | None = None


@cli_command("datasets.list", require_storage=True)
@dataclass(frozen=True)
class ListDatasets(Command[ListResult[DatasetInfo]]):
    """List available datasets in the storage layer."""
    
    pattern: str | None = None
    flags: SharedFlags = shared_flags_field()
    
    def execute(self, deps: Deps) -> CliResult[ListResult[DatasetInfo]]:
        """Execute dataset listing."""
        # deps.storage is guaranteed available because require_storage=True
        catalog = deps.storage.gateway.get_dataset_catalog()
        
        datasets = catalog.list_datasets(pattern=self.pattern)
        
        items = [
            DatasetInfo(
                table_key=ds.table_key,
                name=ds.name,
                row_count=ds.row_count,
                description=ds.description,
            )
            for ds in datasets
        ]
        
        return CliResult.ok(ListResult.from_items(items))


@cli_command("datasets.describe", require_storage=True)
@dataclass(frozen=True)
class DescribeDataset(Command[DatasetSchema]):
    """Describe a specific dataset's schema."""
    
    table_key: str
    include_dependencies: bool = False
    flags: SharedFlags = shared_flags_field()
    
    def execute(self, deps: Deps) -> CliResult[DatasetSchema]:
        """Execute schema description."""
        catalog = deps.storage.gateway.get_dataset_catalog()
        
        ds = catalog.get_dataset(self.table_key)
        if ds is None:
            return fail_dataset_not_found(self.table_key)
        
        upstream = None
        if self.include_dependencies:
            upstream = catalog.get_upstream_dependencies(self.table_key)
        
        return CliResult.ok(DatasetSchema(
            table_key=ds.table_key,
            name=ds.name,
            columns=ds.columns,
            row_count=ds.row_count,
            upstream_dependencies=upstream,
        ))
```

---

## 6. Dependencies (`deps/`)

### 6.1 Protocols (`deps/protocols.py`)

```python
"""Protocols for injectable dependencies.

Each protocol defines a focused interface. Commands declare which they need.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.cli.jobs import Job, JobStatus
    from codeintel.storage.gateway import StorageGateway


@runtime_checkable
class StorageAccess(Protocol):
    """Protocol for storage layer access."""
    
    @property
    def gateway(self) -> StorageGateway:
        """Get the storage gateway (lazy-loaded)."""
        ...
    
    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway."""
        ...


@runtime_checkable
class JobManager(Protocol):
    """Protocol for background job management."""
    
    def list(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 20,
    ) -> list[Job]:
        """List jobs with optional filters."""
        ...
    
    def get(self, job_id: str) -> Job | None:
        """Get a specific job."""
        ...
    
    def get_output(self, job_id: str) -> dict[str, object] | None:
        """Get output from completed job."""
        ...
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        ...


@runtime_checkable
class ServingAccess(Protocol):
    """Protocol for serving layer operations."""
    
    def invoke(
        self,
        operation_id: str,
        params: dict[str, object],
        *,
        skip_prereqs: bool = False,
    ) -> dict[str, object]:
        """Invoke a serving operation."""
        ...
```

### 6.2 Deps Container (`deps/container.py`)

```python
"""Dependency container for command execution."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from codeintel.cli.deps.protocols import JobManager, ServingAccess, StorageAccess
from codeintel.cli.deps.providers import LazyStorageProvider, LazyServingProvider

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig


@dataclass
class Deps:
    """Container for command dependencies.
    
    Commands receive this as their execution context. Dependencies are
    lazy-loaded based on what the command actually accesses.
    
    Example
    -------
    def execute(self, deps: Deps) -> CliResult[T]:
        # Storage only loaded if accessed
        rows = deps.storage.gateway.query("SELECT * FROM t")
        
        # Jobs always available (lightweight)
        jobs = deps.jobs.list(limit=10)
    """
    
    config: CliConfig
    logger: logging.Logger
    jobs: JobManager
    
    # Optional/lazy-loaded
    _storage: StorageAccess | None = field(default=None, repr=False)
    _serving: ServingAccess | None = field(default=None, repr=False)
    
    @property
    def storage(self) -> StorageAccess:
        """Get storage access (raises if not configured)."""
        if self._storage is None:
            raise RuntimeError(
                "Storage not available. Command must declare require_storage=True"
            )
        return self._storage
    
    @property
    def serving(self) -> ServingAccess:
        """Get serving access (raises if not configured)."""
        if self._serving is None:
            raise RuntimeError(
                "Serving not available. Command must declare require_serving=True"
            )
        return self._serving


class DepsBuilder:
    """Builder for constructing Deps with appropriate providers.
    
    Used by @cli_command to build dependencies based on command requirements.
    
    Example
    -------
    with DepsBuilder().with_storage().with_jobs().build() as deps:
        result = command.execute(deps)
    """
    
    def __init__(self) -> None:
        self._require_storage = False
        self._require_serving = False
        self._project_root: Path | None = None
        self._db_path: Path | None = None
    
    def with_storage(self, *, db_path: Path | None = None) -> DepsBuilder:
        """Enable storage access."""
        self._require_storage = True
        self._db_path = db_path
        return self
    
    def with_serving(self) -> DepsBuilder:
        """Enable serving access."""
        self._require_serving = True
        return self
    
    def with_project(self, root: Path | None) -> DepsBuilder:
        """Set project root for resolution."""
        self._project_root = root
        return self
    
    @contextmanager
    def build(self) -> Iterator[Deps]:
        """Build Deps and manage resource lifecycle."""
        from codeintel.cli.config import load_config
        from codeintel.cli.jobs import get_job_manager
        
        config = load_config(validate=False)
        logger = logging.getLogger("codeintel.cli")
        jobs = get_job_manager()
        
        storage: StorageAccess | None = None
        serving: ServingAccess | None = None
        
        try:
            if self._require_storage:
                storage = LazyStorageProvider(
                    project_root=self._project_root,
                    db_path=self._db_path,
                )
            
            if self._require_serving:
                serving = LazyServingProvider(
                    storage=storage,
                    config=config,
                )
            
            yield Deps(
                config=config,
                logger=logger,
                jobs=jobs,
                _storage=storage,
                _serving=serving,
            )
        finally:
            # Cleanup resources
            if storage is not None and hasattr(storage, "close"):
                storage.close()
```

### 6.3 Lazy Providers (`deps/providers.py`)

```python
"""Lazy resource providers.

Resources are only initialized when first accessed, reducing startup overhead.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


class LazyStorageProvider:
    """Lazy-loading storage provider.
    
    Gateway is only opened when .gateway is first accessed.
    """
    
    def __init__(
        self,
        *,
        project_root: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._project_root = project_root
        self._db_path = db_path
        self._gateway: StorageGateway | None = None
    
    @property
    def gateway(self) -> StorageGateway:
        """Get or create storage gateway."""
        if self._gateway is None:
            self._gateway = self._open_gateway()
        return self._gateway
    
    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway."""
        from codeintel.storage.gateway import StorageConfig, open_gateway
        
        db_path = self._resolve_db_path()
        config = StorageConfig(db_path=db_path, read_only=False)
        gw = open_gateway(config)
        try:
            yield gw
        finally:
            gw.close()
    
    def close(self) -> None:
        """Close gateway if open."""
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None
    
    def _open_gateway(self) -> StorageGateway:
        """Open read-only gateway."""
        from codeintel.storage.gateway import StorageConfig, open_gateway
        
        db_path = self._resolve_db_path()
        config = StorageConfig(db_path=db_path, read_only=True)
        return open_gateway(config)
    
    def _resolve_db_path(self) -> Path:
        """Resolve database path from config or explicit value."""
        if self._db_path is not None:
            return self._db_path
        
        from codeintel.cli.resolution.runtime import resolve_from_params
        
        params: dict[str, object] = {}
        if self._project_root is not None:
            params["project_root"] = self._project_root
        
        runtime = resolve_from_params(params)
        return runtime.db_path
```

---

## 7. Execution (`execution/`)

### 7.1 Operation Registry (`execution/registry.py`)

```python
"""Operation registry without global singleton.

The registry is created at application startup and passed through the call chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    
    from codeintel.cli.core.command import Command
    from codeintel.cli.core.results import CliResult


@dataclass(frozen=True)
class OperationSpec:
    """Specification for a registered CLI operation.
    
    Derived automatically from Command classes by @cli_command.
    """
    
    operation_id: str
    name: str
    description: str
    command_class: type[Command[Any]]
    group: str
    
    # Resource requirements
    require_storage: bool = False
    require_runtime: bool = False
    require_serving: bool = False
    
    # Metadata
    tags: tuple[str, ...] = ()
    hidden: bool = False
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for introspection/help."""
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "require_storage": self.require_storage,
            "require_runtime": self.require_runtime,
            "tags": list(self.tags),
            "hidden": self.hidden,
        }


@dataclass
class OperationRegistry:
    """Registry for CLI operations.
    
    Not a global singleton - created at app startup and passed explicitly.
    """
    
    _operations: dict[str, OperationSpec] = field(default_factory=dict)
    
    def register(self, spec: OperationSpec) -> OperationSpec:
        """Register an operation."""
        if spec.operation_id in self._operations:
            raise ValueError(f"Operation already registered: {spec.operation_id}")
        self._operations[spec.operation_id] = spec
        return spec
    
    def get(self, operation_id: str) -> OperationSpec | None:
        """Get an operation by ID."""
        return self._operations.get(operation_id)
    
    def list_operations(
        self,
        *,
        group: str | None = None,
        include_hidden: bool = False,
    ) -> list[OperationSpec]:
        """List operations with optional filters."""
        ops = list(self._operations.values())
        
        if group is not None:
            ops = [op for op in ops if op.group == group]
        if not include_hidden:
            ops = [op for op in ops if not op.hidden]
        
        return sorted(ops, key=lambda op: op.operation_id)
    
    def list_groups(self) -> list[str]:
        """List all operation groups."""
        return sorted({op.group for op in self._operations.values()})


# Module-level registry for decorator access during import
# This is the ONE acceptable global - it's populated at import time
# and then frozen. Tests can create isolated registries.
_DEFAULT_REGISTRY = OperationRegistry()


def get_default_registry() -> OperationRegistry:
    """Get the default registry (populated at import time)."""
    return _DEFAULT_REGISTRY


def create_fresh_registry() -> OperationRegistry:
    """Create a fresh registry (for testing)."""
    return OperationRegistry()
```

### 7.2 Middleware (`execution/middleware.py`)

```python
"""Execution middleware for cross-cutting concerns.

Middleware wraps command execution to add logging, metrics, error handling.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from codeintel.cli.core.command import Command
    from codeintel.cli.core.results import CliResult
    from codeintel.cli.deps import Deps

T = TypeVar("T")


class ExecutionMiddleware(ABC):
    """Base class for execution middleware."""
    
    @abstractmethod
    def before_execute(self, command: Command[T], deps: Deps) -> None:
        """Called before command execution."""
        ...
    
    @abstractmethod
    def after_execute(
        self,
        command: Command[T],
        result: CliResult[T],
        duration_seconds: float,
    ) -> CliResult[T]:
        """Called after command execution. May modify result."""
        ...
    
    @abstractmethod
    def on_error(
        self,
        command: Command[T],
        error: Exception,
        duration_seconds: float,
    ) -> CliResult[T] | None:
        """Called on exception. Return result to suppress, None to propagate."""
        ...


class LoggingMiddleware(ExecutionMiddleware):
    """Log command execution."""
    
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger("codeintel.cli.execution")
    
    def before_execute(self, command: Command[T], deps: Deps) -> None:
        self._log.info("Executing %s", command.__operation_id__)
    
    def after_execute(
        self,
        command: Command[T],
        result: CliResult[T],
        duration_seconds: float,
    ) -> CliResult[T]:
        status = "success" if result.success else "failure"
        self._log.info(
            "Completed %s: %s (%.3fs)",
            command.__operation_id__,
            status,
            duration_seconds,
        )
        return result
    
    def on_error(
        self,
        command: Command[T],
        error: Exception,
        duration_seconds: float,
    ) -> CliResult[T] | None:
        self._log.exception(
            "Error in %s after %.3fs: %s",
            command.__operation_id__,
            duration_seconds,
            error,
        )
        return None  # Propagate exception


@dataclass
class ExecutionPipeline:
    """Pipeline that applies middleware around command execution."""
    
    middleware: list[ExecutionMiddleware]
    
    def execute(self, command: Command[T], deps: Deps) -> CliResult[T]:
        """Execute command with middleware."""
        # Before hooks
        for mw in self.middleware:
            mw.before_execute(command, deps)
        
        start = time.perf_counter()
        
        try:
            result = command.execute(deps)
            duration = time.perf_counter() - start
            
            # After hooks (in reverse order)
            for mw in reversed(self.middleware):
                result = mw.after_execute(command, result, duration)
            
            return result
            
        except Exception as e:
            duration = time.perf_counter() - start
            
            # Error hooks
            for mw in reversed(self.middleware):
                handled = mw.on_error(command, e, duration)
                if handled is not None:
                    return handled
            
            raise  # No middleware handled the error
```

---

## 8. Rendering (`rendering/`)

The rendering layer is already well-designed. Key points:

```python
# rendering/service.py - UnifiedRenderer is kept as-is

# Key integration point: CliResult.to_dict() now auto-serializes
def render_result(self, result: CliResult[T]) -> int:
    """Render result using auto-serialization."""
    # Warnings to stderr
    for warning in result.warnings:
        self._emit_warning(warning)
    
    # Failure case
    if not result.success:
        if result.error:
            self.render_error(result.error)
        return self._exit_code_for_error(result.error)
    
    # Success case - to_dict() handles all serialization automatically
    if result.data is not None:
        self._render_data(result.data)
    
    return 0
```

---

## 9. Errors (`errors/`)

The error handling layer is already well-designed. Key patterns:

```python
# errors/factory.py - Factory functions return CliResult directly

def fail_job_not_found[T](job_id: str) -> CliResult[T]:
    """Create failed result for job not found."""
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:not-found",
            title="Job Not Found",
            detail=f"Job not found: {job_id}",
            status=404,
        )
    )

# Usage in commands:
def execute(self, deps: Deps) -> CliResult[JobInfo]:
    job = deps.jobs.get(self.job_id)
    if job is None:
        return fail_job_not_found(self.job_id)  # Type-safe, no casting needed
    return CliResult.ok(JobInfo(...))
```

---

## 10. Testing Patterns

### 10.1 Testing Commands Directly

```python
"""Test commands without CLI infrastructure."""

import pytest

from codeintel.cli.commands.jobs import ListJobs, JobInfo
from codeintel.cli.core.results import CliResult
from codeintel.cli.core.result_types import ListResult


class FakeJobManager:
    """Fake job manager for testing."""
    
    def __init__(self, jobs: list[Job] | None = None) -> None:
        self._jobs = jobs or []
    
    def list(self, *, status=None, limit=20) -> list[Job]:
        result = self._jobs
        if status is not None:
            result = [j for j in result if j.status == status]
        return result[:limit]


@pytest.fixture
def fake_deps() -> Deps:
    """Create Deps with fakes for testing."""
    return Deps(
        config=FakeConfig(),
        logger=logging.getLogger("test"),
        jobs=FakeJobManager([
            Job(job_id="j1", operation_id="test.op", status=JobStatus.RUNNING),
            Job(job_id="j2", operation_id="test.op", status=JobStatus.COMPLETED),
        ]),
    )


def test_list_jobs_returns_all(fake_deps: Deps) -> None:
    """Test listing all jobs."""
    # Arrange
    cmd = ListJobs(limit=10)
    
    # Act
    result = cmd.execute(fake_deps)
    
    # Assert
    assert result.success
    assert result.data is not None
    assert result.data.count == 2


def test_list_jobs_filters_by_status(fake_deps: Deps) -> None:
    """Test filtering by status."""
    cmd = ListJobs(status=JobStatus.RUNNING, limit=10)
    
    result = cmd.execute(fake_deps)
    
    assert result.success
    assert result.data.count == 1
    assert result.data.items[0].status == "running"
```

### 10.2 Testing with Real Dependencies

```python
"""Integration tests with real storage."""

import pytest
from pathlib import Path

from codeintel.cli.commands.datasets import ListDatasets
from codeintel.cli.deps import DepsBuilder


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create test database with sample data."""
    db_path = tmp_path / "test.duckdb"
    # ... populate test data ...
    return db_path


def test_list_datasets_integration(test_db: Path) -> None:
    """Test dataset listing with real storage."""
    with DepsBuilder().with_storage(db_path=test_db).build() as deps:
        cmd = ListDatasets(pattern="analytics.*")
        result = cmd.execute(deps)
        
        assert result.success
        assert all("analytics." in ds.table_key for ds in result.data.items)
```

### 10.3 Testing Result Serialization

```python
"""Test auto-serialization."""

from codeintel.cli.core.results import result_type


@result_type
@dataclass(frozen=True)
class TestResult:
    required: str
    optional: str | None = None
    nested: NestedResult | None = None


def test_to_dict_omits_none_fields() -> None:
    """Verify None fields are omitted."""
    result = TestResult(required="value")
    
    d = result.to_dict()
    
    assert d == {"required": "value"}
    assert "optional" not in d
    assert "nested" not in d


def test_to_dict_includes_non_none_fields() -> None:
    """Verify non-None fields are included."""
    result = TestResult(required="value", optional="opt")
    
    d = result.to_dict()
    
    assert d == {"required": "value", "optional": "opt"}
```

---

## 11. Migration Strategy

### Phase 1: Foundation (Non-Breaking)

**Goal**: Add new infrastructure alongside existing code.

1. Add `core/command.py` with `Command[T]` base class
2. Add `@result_type` decorator to `core/results.py`
3. Add `deps/` package with protocols and container
4. Add generic result types to `core/result_types.py`

**Validation**: Existing code continues to work unchanged.

### Phase 2: Pilot Migration

**Goal**: Migrate one command group to validate the pattern.

1. Migrate `commands/jobs.py` to new pattern
2. Update `handlers/jobs.py` result types to use `@result_type`
3. Keep `HandlerContext` bridge via `Deps.from_legacy_context()`

**Validation**: Jobs commands work with new and old patterns.

### Phase 3: Full Migration

**Goal**: Migrate remaining command groups.

Order by complexity:
1. Simple read-only: `health`, `plugins`, `help_commands`
2. Storage readers: `datasets`, `storage`, `graphs`
3. State modifiers: `build`, `docs`, `history`
4. Complex: `ops`, `serve`

### Phase 4: Cleanup

**Goal**: Remove legacy code.

1. Remove `HandlerContext.param_*` methods
2. Remove separate `handlers/` files (logic now in commands)
3. Remove manual `to_dict()` implementations
4. Update all imports and documentation

---

## 12. Extension Points for Plugins/Build

This architecture provides clean extension points for future plugin and build system integration:

### 12.1 Plugin Commands

Plugins can define commands using the same patterns:

```python
# In a plugin's main.py
from codeintel.cli.core.command import Command, cli_command
from codeintel.cli.core.results import CliResult, result_type


@result_type
@dataclass(frozen=True)
class PluginResult:
    message: str


@cli_command("myplugin.greet")
@dataclass(frozen=True)
class Greet(Command[PluginResult]):
    """Greet the user."""
    
    name: str = "World"
    
    def execute(self, deps: Deps) -> CliResult[PluginResult]:
        return CliResult.ok(PluginResult(message=f"Hello, {self.name}!"))


def register(registry: OperationRegistry) -> None:
    """Register plugin operations."""
    # Commands self-register via @cli_command, but plugins can also
    # programmatically register if needed
    pass
```

### 12.2 Build System Integration

Build commands can use the same `Deps` pattern with build-specific dependencies:

```python
@dataclass
class BuildDeps(Deps):
    """Extended deps for build commands."""
    
    executor: BuildExecutor
    target_graph: TargetGraph


@cli_command("build.run", require_storage=True)
@dataclass(frozen=True)
class RunBuild(Command[BuildExecutionResult]):
    """Execute build targets."""
    
    targets: tuple[str, ...] = ()
    force: bool = False
    
    def execute(self, deps: BuildDeps) -> CliResult[BuildExecutionResult]:
        # Build-specific logic with typed deps
        plan = deps.executor.plan(
            targets=self.targets,
            graph=deps.target_graph,
            force=self.force,
        )
        return deps.executor.execute(plan)
```

### 12.3 Custom Middleware

Plugins can add middleware for cross-cutting concerns:

```python
class MetricsMiddleware(ExecutionMiddleware):
    """Emit metrics for command execution."""
    
    def after_execute(self, command, result, duration):
        metrics.emit(
            name="cli.command.duration",
            value=duration,
            tags={"operation": command.__operation_id__, "success": result.success},
        )
        return result
```

---

## Appendix: Key Type Definitions

```python
# Type variables used throughout
T = TypeVar("T")  # Result data type
E = TypeVar("E", bound=Enum)  # Enum type for param extraction

# Protocol for anything that can serialize to dict
@runtime_checkable
class SerializableResult(Protocol):
    def to_dict(self) -> dict[str, object]: ...

# Command base with generic result type
class Command(ABC, Generic[T]):
    @abstractmethod
    def execute(self, deps: Deps) -> CliResult[T]: ...

# Result wrapper (unchanged from current)
@dataclass
class CliResult(Generic[T]):
    success: bool
    data: T | None = None
    error: ProblemDetail | None = None
    
    @classmethod
    def ok(cls, data: T) -> CliResult[T]: ...
    
    @classmethod
    def fail(cls, error: ProblemDetail) -> CliResult[T]: ...
```

---

## Summary

This architecture provides:

1. **End-to-end type safety**: No string-based parameter extraction
2. **80% less boilerplate**: Auto-serialization, generic result types
3. **Explicit dependencies**: Easy to test, easy to understand
4. **Clean extension points**: Plugins and build use the same patterns
5. **Evolutionary migration**: Can be implemented incrementally

The design prioritizes maintainability and extensibility while keeping the excellent parts of the current architecture (CliResult, UnifiedRenderer, error factories).
