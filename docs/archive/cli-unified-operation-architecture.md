# CLI Unified Operation Architecture

> **Status**: Proposed  
> **Author**: Architecture Review  
> **Date**: 2025-12-10  
> **Scope**: `src/codeintel/cli/`, `src/codeintel/serving/operations/`

---

## Executive Summary

This document specifies a **unified operation architecture** that transcends the CLI to provide a single execution model for CLI commands, plugins, serving endpoints, and background jobs. Rather than building a "best-in-class CLI" that plugins and serving adapt to, we build **best-in-class operations** that all entry points consume uniformly.

**Core Insight**: An "operation" is the atomic unit of business logic. CLI, HTTP, plugins, and jobs are just *adapters* that extract parameters from different sources and present results in different formats. The operation itself should be identical regardless of how it's invoked.

**Key Benefits**:
- **Single implementation**: Write once, expose via CLI, HTTP, plugin, and job queue
- **Unified observability**: One middleware stack for logging/metrics/tracing across all entry points
- **Consistent behavior**: Same validation, error handling, and capabilities everywhere
- **Natural plugin model**: Plugins register operations, not CLI-specific handlers
- **Clean testing**: Test operations directly without any adapter infrastructure

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Package Structure](#3-package-structure)
4. [Operations (`operations/`)](#4-operations-operations)
5. [Context (`context/`)](#5-context-context)
6. [Adapters (`adapters/`)](#6-adapters-adapters)
7. [Middleware (`middleware/`)](#7-middleware-middleware)
8. [Result Types (`results/`)](#8-result-types-results)
9. [Errors (`errors/`)](#9-errors-errors)
10. [Registry (`registry/`)](#10-registry-registry)
11. [CLI Integration (`cli/`)](#11-cli-integration-cli)
12. [Plugin Integration](#12-plugin-integration)
13. [Testing Patterns](#13-testing-patterns)
14. [Migration Strategy](#14-migration-strategy)

---

## 1. Design Principles

### 1.1 Operations are Entry-Point Agnostic

An operation should not know or care whether it was invoked from CLI, HTTP, or a plugin:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Entry Points (Adapters)                          │
├────────────┬─────────────┬────────────────┬────────────┬────────────────┤
│   CLI      │    HTTP     │    Plugin      │   MCP      │   Background   │
│  (cyclopts)│  (FastAPI)  │  (sandbox)     │ (stdio)    │   Job Queue    │
├────────────┴─────────────┴────────────────┴────────────┴────────────────┤
│                                                                          │
│                        Parameter Extraction                               │
│            (each adapter extracts params from its source)                │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                      Operation[TParams, TResult]                         │
│              (single implementation, full type safety)                   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                            OpContext                                     │
│        (storage, serving, jobs, telemetry — injected)                   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                        Result[TResult]                                   │
│            (success/failure, adapters format for output)                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Anti-pattern** (current):
```python
# CLI-specific handler
def list_jobs_handler(ctx: HandlerContext) -> CliResult[JobsListResult]:
    limit = ctx.param_int("limit", 20)
    jobs = get_job_manager().list(limit=limit)
    return CliResult.ok(JobsListResult.from_models(jobs))

# HTTP-specific handler (separate implementation!)
@app.get("/api/jobs")
def list_jobs_http(limit: int = 20) -> JobsListResponse:
    jobs = get_job_manager().list(limit=limit)
    return JobsListResponse.from_models(jobs)
```

**Pattern** (proposed):
```python
# Single operation implementation
@operation("jobs.list")
class ListJobs(Operation[ListJobsParams, ListResult[JobInfo]]):
    """List background jobs with optional filters."""
    
    def execute(self, params: ListJobsParams, ctx: OpContext) -> Result[ListResult[JobInfo]]:
        jobs = ctx.jobs.list(status=params.status, limit=params.limit)
        return Result.ok(ListResult.from_items([JobInfo.from_model(j) for j in jobs]))

# CLI adapter (auto-generated from params type)
# HTTP adapter (auto-generated from params type)
# Plugin adapter (validates capabilities, delegates)
# Job adapter (deserializes params, runs async)
```

### 1.2 Parameters are Explicit Types

Each operation declares its parameters as a frozen dataclass. This single declaration drives CLI argument parsing, HTTP schema generation, and validation:

```python
@dataclass(frozen=True)
class ListJobsParams:
    """Parameters for jobs.list operation."""
    status: JobStatus | None = None
    limit: int = 20
    
    # CLI-specific hints (optional, adapter-specific)
    class CliMeta:
        status = Parameter(help="Filter by status")
        limit = Parameter(help="Maximum jobs to return")
```

### 1.3 Context Provides All Resources

Operations receive an `OpContext` that provides access to all resources they might need. The context is adapter-agnostic:

```python
def execute(self, params: P, ctx: OpContext) -> Result[R]:
    # Storage access (lazy-loaded)
    rows = ctx.storage.gateway.query("SELECT * FROM t")
    
    # Job management
    jobs = ctx.jobs.list(limit=10)
    
    # Configuration
    project = ctx.config.project_root
    
    # Telemetry (automatic, but accessible for custom spans)
    with ctx.telemetry.span("custom_operation"):
        ...
```

### 1.4 Middleware is Universal

A single middleware stack applies to all operation invocations, regardless of entry point:

```python
# Same logging/metrics/tracing for CLI, HTTP, plugin, and job invocations
pipeline = OperationPipeline([
    TelemetryMiddleware(),      # Spans, metrics
    LoggingMiddleware(),         # Structured logs
    ValidationMiddleware(),      # Param validation
    CapabilityMiddleware(),      # Plugin sandboxing
    ErrorHandlingMiddleware(),   # Consistent error translation
])
```

### 1.5 Results are Typed and Serializable

Operations return `Result[T]` where `T` is auto-serializable. Adapters handle format conversion:

```python
# Operation returns typed Result
def execute(...) -> Result[ListResult[JobInfo]]:
    return Result.ok(ListResult.from_items(items))

# CLI adapter: renders as table or JSON
# HTTP adapter: serializes to JSON response
# Plugin adapter: returns structured dict
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Entry Points                                │
│  ┌───────┐  ┌────────┐  ┌─────────┐  ┌────────┐  ┌──────────────────┐  │
│  │  CLI  │  │  HTTP  │  │ Plugin  │  │  MCP   │  │ Background Job   │  │
│  │ App   │  │ Router │  │ Sandbox │  │ Server │  │    Worker        │  │
│  └───┬───┘  └───┬────┘  └────┬────┘  └───┬────┘  └────────┬─────────┘  │
│      │          │            │           │                 │            │
│      └──────────┴────────────┴───────────┴─────────────────┘            │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Adapter Layer                                 │  │
│  │  • Extract params from source (args, body, env)                  │  │
│  │  • Map to Operation.params_type                                  │  │
│  │  • Format Result for output                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    OperationPipeline                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │  │
│  │  │Telemetry │→│ Logging  │→│Validation│→│Capability│→ execute() │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               Operation[TParams, TResult]                         │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │  │
│  │  │   Params Type   │  │   Result Type    │  │   execute()     │  │  │
│  │  │  (frozen DC)    │  │  (@result_type)  │  │  Pure Logic     │  │  │
│  │  └─────────────────┘  └──────────────────┘  └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        OpContext                                  │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────────┐   │  │
│  │  │storage │ │ config │ │  jobs  │ │telemetry│ │capabilities │   │  │
│  │  └────────┘ └────────┘ └────────┘ └─────────┘ └─────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Result[TResult]                               │  │
│  │              (success/failure + data + metadata)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Package Structure

```
src/codeintel/
├── operations/                    # Core operation infrastructure
│   ├── __init__.py
│   ├── base.py                    # Operation[P, R] protocol, @operation
│   ├── params.py                  # Parameter base classes, validation
│   ├── context.py                 # OpContext, resource providers
│   ├── result.py                  # Result[T], @result_type
│   ├── pipeline.py                # OperationPipeline, execution
│   ├── registry.py                # OperationRegistry, OperationSpec
│   │
│   ├── middleware/                # Middleware implementations
│   │   ├── __init__.py
│   │   ├── base.py                # OperationMiddleware protocol
│   │   ├── logging.py             # LoggingMiddleware
│   │   ├── telemetry.py           # TelemetryMiddleware (spans, metrics)
│   │   ├── validation.py          # ValidationMiddleware
│   │   ├── capabilities.py        # CapabilityMiddleware (sandbox)
│   │   └── errors.py              # ErrorHandlingMiddleware
│   │
│   ├── adapters/                  # Entry point adapters
│   │   ├── __init__.py
│   │   ├── cli.py                 # CLI adapter (cyclopts integration)
│   │   ├── http.py                # HTTP adapter (FastAPI integration)
│   │   ├── plugin.py              # Plugin adapter (sandbox, capabilities)
│   │   └── job.py                 # Background job adapter
│   │
│   ├── catalog/                   # Operation implementations by domain
│   │   ├── __init__.py
│   │   ├── jobs.py                # jobs.* operations
│   │   ├── datasets.py            # datasets.* operations
│   │   ├── storage.py             # storage.* operations
│   │   ├── build.py               # build.* operations
│   │   ├── health.py              # health.* operations
│   │   ├── plugins.py             # plugins.* operations
│   │   ├── graphs.py              # graphs.* operations
│   │   ├── docs.py                # docs.* operations
│   │   └── ...
│   │
│   └── errors/                    # Error handling
│       ├── __init__.py
│       ├── problem_detail.py      # RFC 9457 ProblemDetail
│       ├── taxonomy.py            # Error type enums
│       └── factory.py             # fail_* helpers
│
├── cli/                           # CLI-specific infrastructure
│   ├── __init__.py
│   ├── app.py                     # Root cyclopts App
│   ├── commands/                  # CLI command wrappers (thin)
│   │   ├── __init__.py
│   │   └── _registry.py           # Auto-register operations as commands
│   ├── rendering/                 # CLI output rendering
│   │   ├── __init__.py
│   │   ├── service.py             # UnifiedRenderer
│   │   ├── table.py               # Table rendering
│   │   └── types.py               # OutputFormat
│   ├── completions/               # Shell completions (unchanged)
│   └── config/                    # CLI config loading (unchanged)
│
├── serving/                       # HTTP-specific infrastructure
│   ├── operations/                # HTTP operation registration
│   └── ...
│
└── plugins/                       # Plugin infrastructure
    ├── registry.py                # Plugin operation registration
    ├── sandbox.py                 # Capability sandboxing
    └── ...
```

---

## 4. Operations (`operations/`)

### 4.1 The Operation Protocol (`operations/base.py`)

```python
"""Core operation protocol and @operation decorator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result


@runtime_checkable
class Operation[TParams, TResult](Protocol):
    """Protocol for operations.
    
    Operations are the atomic units of business logic in CodeIntel.
    They receive typed parameters and return typed results.
    
    Type Parameters
    ---------------
    TParams
        Parameter dataclass type (frozen, validated).
    TResult
        Result data type (auto-serializable via @result_type).
    
    Example
    -------
    @operation("jobs.list")
    class ListJobs(Operation[ListJobsParams, ListResult[JobInfo]]):
        '''List background jobs.'''
        
        def execute(
            self, params: ListJobsParams, ctx: OpContext
        ) -> Result[ListResult[JobInfo]]:
            jobs = ctx.jobs.list(status=params.status, limit=params.limit)
            return Result.ok(ListResult.from_items(
                [JobInfo.from_model(j) for j in jobs]
            ))
    """
    
    # Class-level metadata (set by @operation decorator)
    __operation_id__: ClassVar[str]
    __params_type__: ClassVar[type[TParams]]
    __result_type__: ClassVar[type[TResult]]
    __capabilities__: ClassVar[frozenset[str]]
    
    @abstractmethod
    def execute(self, params: TParams, ctx: OpContext) -> Result[TResult]:
        """Execute the operation.
        
        Parameters
        ----------
        params
            Validated parameter instance.
        ctx
            Operation context with resources.
            
        Returns
        -------
        Result[TResult]
            Success with result data or failure with error.
        """
        ...


@dataclass(frozen=True)
class OperationSpec:
    """Specification for a registered operation.
    
    Created by @operation decorator and stored in OperationRegistry.
    """
    
    operation_id: str
    name: str
    description: str
    params_type: type[object]
    result_type: type[object]
    operation_class: type[Operation[object, object]]
    group: str
    
    # Capability requirements
    capabilities: frozenset[str] = frozenset()
    
    # Resource requirements (derived from capabilities)
    require_storage: bool = False
    require_runtime: bool = False
    require_serving: bool = False
    
    # Visibility
    hidden: bool = False
    tags: tuple[str, ...] = ()
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for introspection."""
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "capabilities": sorted(self.capabilities),
            "require_storage": self.require_storage,
            "require_runtime": self.require_runtime,
            "tags": list(self.tags),
            "hidden": self.hidden,
        }
```

### 4.2 The `@operation` Decorator (`operations/base.py` continued)

```python
from codeintel.operations.registry import get_default_registry


# Standard capability constants
class Capability:
    """Standard operation capabilities."""
    
    STORAGE_READ = "storage:read"
    STORAGE_WRITE = "storage:write"
    RUNTIME = "runtime"
    SERVING = "serving"
    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    NETWORK = "network"
    FILESYSTEM_READ = "filesystem:read"
    FILESYSTEM_WRITE = "filesystem:write"


def operation[P, R](
    operation_id: str,
    *,
    capabilities: frozenset[str] = frozenset(),
    hidden: bool = False,
    tags: tuple[str, ...] = (),
) -> Callable[[type[Operation[P, R]]], type[Operation[P, R]]]:
    """Register an operation class.
    
    The decorator:
    1. Extracts params and result types from class signature
    2. Sets class-level metadata
    3. Registers with OperationRegistry
    4. Validates the class structure
    
    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "jobs.list", "datasets.describe").
    capabilities
        Required capabilities (e.g., {"storage:read"}).
    hidden
        If True, hide from help/discovery.
    tags
        Optional tags for filtering.
        
    Example
    -------
    @operation("jobs.list", capabilities={Capability.JOBS_READ})
    class ListJobs(Operation[ListJobsParams, ListResult[JobInfo]]):
        def execute(self, params, ctx):
            ...
    """
    def decorator(cls: type[Operation[P, R]]) -> type[Operation[P, R]]:
        # Extract type parameters from class
        params_type, result_type = _extract_type_params(cls)
        
        # Validate class structure
        if not hasattr(cls, "execute"):
            raise TypeError(f"{cls.__name__} must define execute()")
        
        # Set class-level metadata
        cls.__operation_id__ = operation_id
        cls.__params_type__ = params_type
        cls.__result_type__ = result_type
        cls.__capabilities__ = capabilities
        
        # Derive resource requirements from capabilities
        require_storage = (
            Capability.STORAGE_READ in capabilities or
            Capability.STORAGE_WRITE in capabilities
        )
        require_runtime = Capability.RUNTIME in capabilities
        require_serving = Capability.SERVING in capabilities
        
        # Extract description from docstring
        description = cls.__doc__ or f"Execute {operation_id}"
        description = description.strip().split("\n", maxsplit=1)[0].strip()
        
        # Extract group from operation_id
        group = operation_id.split(".", maxsplit=1)[0]
        
        # Create and register spec
        spec = OperationSpec(
            operation_id=operation_id,
            name=cls.__name__,
            description=description,
            params_type=params_type,
            result_type=result_type,
            operation_class=cls,
            group=group,
            capabilities=capabilities,
            require_storage=require_storage,
            require_runtime=require_runtime,
            require_serving=require_serving,
            hidden=hidden,
            tags=tags,
        )
        
        get_default_registry().register(spec)
        
        return cls
    
    return decorator


def _extract_type_params(cls: type[object]) -> tuple[type[object], type[object]]:
    """Extract TParams and TResult from Operation[TParams, TResult]."""
    import typing
    
    for base in getattr(cls, "__orig_bases__", ()):
        if hasattr(base, "__origin__") and base.__origin__ is Operation:
            args = typing.get_args(base)
            if len(args) == 2:
                return args[0], args[1]
    
    raise TypeError(
        f"{cls.__name__} must explicitly specify type parameters: "
        f"Operation[ParamsType, ResultType]"
    )
```

### 4.3 Parameter Types (`operations/params.py`)

```python
"""Parameter base classes and validation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Protocol

if TYPE_CHECKING:
    from cyclopts import Parameter


class ParamsProtocol(Protocol):
    """Protocol for operation parameter types."""
    
    def validate(self) -> list[str]:
        """Validate parameters, return list of error messages."""
        ...


@dataclass(frozen=True)
class BaseParams:
    """Base class for operation parameters.
    
    Subclass and add fields. Use annotations for CLI hints.
    
    Example
    -------
    @dataclass(frozen=True)
    class ListJobsParams(BaseParams):
        status: JobStatus | None = None
        limit: int = 20
        
        class CliMeta:
            '''CLI-specific hints.'''
            status: ClassVar[Parameter] = Parameter(help="Filter by job status")
            limit: ClassVar[Parameter] = Parameter(help="Max jobs to return")
        
        def validate(self) -> list[str]:
            errors = []
            if self.limit < 1:
                errors.append("limit must be >= 1")
            if self.limit > 1000:
                errors.append("limit must be <= 1000")
            return errors
    """
    
    def validate(self) -> list[str]:
        """Validate parameters. Override to add custom validation."""
        return []
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for logging/debugging."""
        return {
            f.name: _serialize_param(getattr(self, f.name))
            for f in dataclasses.fields(self)
        }


def _serialize_param(value: object) -> object:
    """Serialize a parameter value for logging."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _serialize_param(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    return value


# Common parameter components (composable)

@dataclass(frozen=True)
class ProjectParams:
    """Mixin for operations that work with a project."""
    
    project_root: Path | None = None


@dataclass(frozen=True)  
class PaginationParams:
    """Mixin for operations that support pagination."""
    
    limit: int = 20
    offset: int = 0
    
    def validate(self) -> list[str]:
        errors = []
        if self.limit < 1:
            errors.append("limit must be >= 1")
        if self.limit > 1000:
            errors.append("limit must be <= 1000")
        if self.offset < 0:
            errors.append("offset must be >= 0")
        return errors


@dataclass(frozen=True)
class OutputParams:
    """Mixin for operations with configurable output."""
    
    output_format: str = "text"
    verbose: int = 0
```

---

## 5. Context (`context/`)

### 5.1 OpContext (`operations/context.py`)

```python
"""Operation context providing access to resources."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.operations.middleware.telemetry import TelemetryContext
    from codeintel.storage.gateway import StorageGateway


@runtime_checkable
class StorageAccess(Protocol):
    """Protocol for storage layer access."""
    
    @property
    def gateway(self) -> StorageGateway:
        """Get read-only storage gateway."""
        ...
    
    @contextmanager
    def write_scope(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway."""
        ...


@runtime_checkable
class JobsAccess(Protocol):
    """Protocol for job management."""
    
    def list(
        self,
        *,
        status: str | None = None,
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
    
    def cleanup(self, *, max_age_days: int = 7) -> int:
        """Clean up old completed/failed jobs."""
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


@dataclass
class OpContext:
    """Context for operation execution.
    
    Provides access to all resources an operation might need.
    Resources are lazy-loaded based on what's actually accessed.
    
    Example
    -------
    def execute(self, params: P, ctx: OpContext) -> Result[R]:
        # Access storage (loaded on first access)
        rows = ctx.storage.gateway.query("SELECT * FROM t")
        
        # Access jobs (always available, lightweight)
        jobs = ctx.jobs.list(limit=10)
        
        # Access config
        if ctx.config.project_root:
            ...
        
        # Create custom telemetry span
        with ctx.telemetry.span("my_operation"):
            ...
    """
    
    # Always available (lightweight)
    config: OpConfig
    logger: logging.Logger
    jobs: JobsAccess
    telemetry: TelemetryContext
    
    # Capability-gated resources (raise if not available)
    _storage: StorageAccess | None = field(default=None, repr=False)
    _serving: ServingAccess | None = field(default=None, repr=False)
    
    # Capability tracking for sandbox
    _granted_capabilities: frozenset[str] = field(
        default_factory=frozenset, repr=False
    )
    
    @property
    def storage(self) -> StorageAccess:
        """Get storage access.
        
        Raises
        ------
        CapabilityError
            If storage:read capability not granted.
        """
        if self._storage is None:
            raise CapabilityError(
                "storage:read",
                "Operation requires storage:read capability",
            )
        return self._storage
    
    @property
    def serving(self) -> ServingAccess:
        """Get serving access.
        
        Raises
        ------
        CapabilityError
            If serving capability not granted.
        """
        if self._serving is None:
            raise CapabilityError(
                "serving",
                "Operation requires serving capability",
            )
        return self._serving
    
    def has_capability(self, capability: str) -> bool:
        """Check if a capability is granted."""
        return capability in self._granted_capabilities


@dataclass(frozen=True)
class OpConfig:
    """Configuration available to operations."""
    
    project_root: Path | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    verbosity: int = 0


class CapabilityError(Exception):
    """Raised when an operation tries to access a resource without capability."""
    
    def __init__(self, capability: str, message: str) -> None:
        self.capability = capability
        super().__init__(message)
```

### 5.2 Context Builder (`operations/context.py` continued)

```python
from codeintel.operations.context import OpContext, OpConfig


class OpContextBuilder:
    """Builder for constructing OpContext with resources.
    
    Used by adapters to build context based on operation requirements.
    
    Example
    -------
    with (
        OpContextBuilder()
        .with_capabilities({"storage:read", "jobs:read"})
        .with_project(project_root)
        .build()
    ) as ctx:
        result = operation.execute(params, ctx)
    """
    
    def __init__(self) -> None:
        self._capabilities: set[str] = set()
        self._project_root: Path | None = None
        self._db_path: Path | None = None
        self._verbosity: int = 0
    
    def with_capabilities(self, caps: frozenset[str]) -> OpContextBuilder:
        """Grant capabilities to the context."""
        self._capabilities.update(caps)
        return self
    
    def with_project(self, root: Path | None) -> OpContextBuilder:
        """Set project root for runtime resolution."""
        self._project_root = root
        return self
    
    def with_storage(self, *, db_path: Path | None = None) -> OpContextBuilder:
        """Enable storage access."""
        self._capabilities.add(Capability.STORAGE_READ)
        self._db_path = db_path
        return self
    
    def with_verbosity(self, level: int) -> OpContextBuilder:
        """Set verbosity level."""
        self._verbosity = level
        return self
    
    @contextmanager
    def build(self) -> Iterator[OpContext]:
        """Build context and manage resource lifecycle."""
        from codeintel.operations.providers import (
            LazyStorageProvider,
            LazyServingProvider,
            LazyJobsProvider,
            TelemetryContextImpl,
        )
        
        logger = logging.getLogger("codeintel.operations")
        
        config = OpConfig(
            project_root=self._project_root,
            db_path=self._db_path,
            verbosity=self._verbosity,
        )
        
        storage: StorageAccess | None = None
        serving: ServingAccess | None = None
        
        try:
            # Initialize storage if capability granted
            if Capability.STORAGE_READ in self._capabilities:
                storage = LazyStorageProvider(
                    project_root=self._project_root,
                    db_path=self._db_path,
                )
            
            # Initialize serving if capability granted
            if Capability.SERVING in self._capabilities:
                serving = LazyServingProvider(storage=storage)
            
            yield OpContext(
                config=config,
                logger=logger,
                jobs=LazyJobsProvider(),
                telemetry=TelemetryContextImpl(),
                _storage=storage,
                _serving=serving,
                _granted_capabilities=frozenset(self._capabilities),
            )
            
        finally:
            # Cleanup resources
            if storage is not None and hasattr(storage, "close"):
                storage.close()
```

---

## 6. Adapters (`adapters/`)

### 6.1 CLI Adapter (`operations/adapters/cli.py`)

```python
"""CLI adapter for operations.

Generates cyclopts commands from operation specifications.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_type_hints

from cyclopts import App, Parameter

from codeintel.operations.context import OpContextBuilder
from codeintel.operations.pipeline import get_default_pipeline
from codeintel.operations.registry import get_default_registry

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec
    from codeintel.operations.result import Result


class CliAdapter:
    """Adapter that generates CLI commands from operations.
    
    For each registered operation, generates:
    1. A dataclass command with fields from params_type
    2. A __call__ that invokes the operation through the pipeline
    3. Cyclopts registration for argument parsing
    
    Example
    -------
    # In cli/app.py
    root_app = App(name="codeintel")
    cli_adapter = CliAdapter(root_app)
    cli_adapter.register_all()  # Registers all operations as commands
    """
    
    def __init__(self, app: App) -> None:
        self._app = app
        self._sub_apps: dict[str, App] = {}
    
    def register_all(self) -> None:
        """Register all operations from the registry."""
        registry = get_default_registry()
        
        for spec in registry.list_operations(include_hidden=False):
            self._register_operation(spec)
    
    def _register_operation(self, spec: OperationSpec) -> None:
        """Register a single operation as a CLI command."""
        # Get or create sub-app for group
        group = spec.group
        if group not in self._sub_apps:
            sub_app = App(name=group, help=f"{group.title()} operations")
            self._app.command(sub_app)
            self._sub_apps[group] = sub_app
        
        sub_app = self._sub_apps[group]
        
        # Generate command class from params type
        command_cls = self._generate_command_class(spec)
        
        # Register with cyclopts
        command_name = spec.operation_id.split(".", maxsplit=1)[-1]
        sub_app.command(name=command_name)(command_cls)
    
    def _generate_command_class(self, spec: OperationSpec) -> type:
        """Generate a dataclass command from operation spec."""
        params_type = spec.params_type
        
        # Get fields from params type
        if dataclasses.is_dataclass(params_type):
            param_fields = dataclasses.fields(params_type)
        else:
            param_fields = []
        
        # Build command fields
        fields_dict: dict[str, tuple[type, Any]] = {}
        
        for fld in param_fields:
            # Get CLI hints from CliMeta if present
            cli_meta = getattr(params_type, "CliMeta", None)
            cli_hint = getattr(cli_meta, fld.name, None) if cli_meta else None
            
            if cli_hint is not None:
                fields_dict[fld.name] = (
                    Annotated[fld.type, cli_hint],
                    fld.default if fld.default is not dataclasses.MISSING else ...,
                )
            else:
                fields_dict[fld.name] = (
                    fld.type,
                    fld.default if fld.default is not dataclasses.MISSING else ...,
                )
        
        # Add shared flags
        fields_dict["_output_format"] = (
            Annotated[str, Parameter(name="--output-format", help="Output format")],
            "text",
        )
        fields_dict["_verbose"] = (
            Annotated[int, Parameter(name=["-v", "--verbose"], count=True)],
            0,
        )
        fields_dict["_json"] = (
            Annotated[bool, Parameter(name="--json", help="JSON output")],
            False,
        )
        
        # Create command class dynamically
        @dataclass(frozen=True)
        class GeneratedCommand:
            __doc__ = spec.description
            
            def __call__(self) -> None:
                self._execute()
            
            def _execute(self) -> None:
                _execute_cli_command(self, spec)
        
        # Add fields to class
        for name, (type_, default) in fields_dict.items():
            setattr(GeneratedCommand, "__annotations__", 
                    {**getattr(GeneratedCommand, "__annotations__", {}), name: type_})
        
        return dataclasses.dataclass(frozen=True)(GeneratedCommand)


def _execute_cli_command(cmd: object, spec: OperationSpec) -> None:
    """Execute a CLI command by invoking the operation."""
    from codeintel.cli.rendering import get_renderer
    
    # Extract parameters from command instance
    params = _extract_params(cmd, spec.params_type)
    
    # Extract output preferences
    output_format = getattr(cmd, "_output_format", "text")
    if getattr(cmd, "_json", False):
        output_format = "json"
    verbose = getattr(cmd, "_verbose", 0)
    
    # Build context
    project_root = getattr(params, "project_root", None)
    
    with (
        OpContextBuilder()
        .with_capabilities(spec.capabilities)
        .with_project(project_root)
        .with_verbosity(verbose)
        .build()
    ) as ctx:
        # Execute through pipeline
        pipeline = get_default_pipeline()
        operation = spec.operation_class()
        result = pipeline.execute(operation, params, ctx)
    
    # Render result
    renderer = get_renderer(output_format)
    exit_code = renderer.render_result(result)
    
    if exit_code != 0:
        sys.exit(exit_code)


def _extract_params(cmd: object, params_type: type) -> object:
    """Extract params from command fields."""
    if not dataclasses.is_dataclass(params_type):
        return params_type()
    
    kwargs: dict[str, object] = {}
    for fld in dataclasses.fields(params_type):
        if hasattr(cmd, fld.name):
            kwargs[fld.name] = getattr(cmd, fld.name)
    
    return params_type(**kwargs)
```

### 6.2 HTTP Adapter (`operations/adapters/http.py`)

```python
"""HTTP adapter for operations.

Generates FastAPI routes from operation specifications.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from codeintel.operations.context import OpContextBuilder
from codeintel.operations.pipeline import get_default_pipeline
from codeintel.operations.registry import get_default_registry

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec


class HttpAdapter:
    """Adapter that generates HTTP routes from operations.
    
    For each operation:
    1. Creates a route (GET for reads, POST for writes)
    2. Generates Pydantic model from params_type
    3. Maps Result to HTTP response
    
    Example
    -------
    router = APIRouter(prefix="/api/v1")
    http_adapter = HttpAdapter(router)
    http_adapter.register_all()
    """
    
    def __init__(self, router: APIRouter) -> None:
        self._router = router
    
    def register_all(self) -> None:
        """Register all operations as HTTP routes."""
        registry = get_default_registry()
        
        for spec in registry.list_operations(include_hidden=False):
            self._register_operation(spec)
    
    def _register_operation(self, spec: OperationSpec) -> None:
        """Register a single operation as an HTTP route."""
        # Determine HTTP method based on capabilities
        is_write = (
            "storage:write" in spec.capabilities or
            "jobs:write" in spec.capabilities or
            "filesystem:write" in spec.capabilities
        )
        method = "post" if is_write else "get"
        
        # Create route path from operation_id
        # e.g., "jobs.list" -> "/jobs/list"
        path = "/" + spec.operation_id.replace(".", "/")
        
        # Create endpoint function
        async def endpoint(
            params: spec.params_type,  # type: ignore[name-defined]
            _spec: OperationSpec = spec,
        ) -> dict[str, object]:
            return await _execute_http_operation(params, _spec)
        
        # Register route
        route_method = getattr(self._router, method)
        route_method(
            path,
            response_model=None,  # We return dicts
            summary=spec.description,
            tags=[spec.group],
        )(endpoint)


async def _execute_http_operation(
    params: object,
    spec: OperationSpec,
) -> dict[str, object]:
    """Execute an operation from HTTP context."""
    # Build context (async-aware)
    project_root = getattr(params, "project_root", None)
    
    with (
        OpContextBuilder()
        .with_capabilities(spec.capabilities)
        .with_project(project_root)
        .build()
    ) as ctx:
        pipeline = get_default_pipeline()
        operation = spec.operation_class()
        result = pipeline.execute(operation, params, ctx)
    
    # Map to HTTP response
    if not result.success:
        error = result.error
        raise HTTPException(
            status_code=error.status if error else 500,
            detail=error.to_dict() if error else {"title": "Unknown error"},
        )
    
    return result.to_dict()
```

### 6.3 Plugin Adapter (`operations/adapters/plugin.py`)

```python
"""Plugin adapter for operations.

Provides sandboxed operation execution for plugins with capability gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.operations.context import CapabilityError, OpContextBuilder
from codeintel.operations.pipeline import get_default_pipeline
from codeintel.operations.registry import get_default_registry

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec
    from codeintel.operations.result import Result
    from codeintel.plugins.manifest import PluginManifest


@dataclass
class PluginAdapter:
    """Adapter for plugin operation execution.
    
    Validates plugin capabilities before allowing operation invocation.
    Provides sandboxed context with only granted capabilities.
    
    Example
    -------
    adapter = PluginAdapter(plugin_manifest)
    
    # This will fail if plugin doesn't have storage:read
    result = adapter.invoke("datasets.list", {"limit": 10})
    """
    
    manifest: PluginManifest
    
    def invoke(
        self,
        operation_id: str,
        params: dict[str, object],
    ) -> Result[object]:
        """Invoke an operation on behalf of a plugin.
        
        Parameters
        ----------
        operation_id
            Operation to invoke.
        params
            Parameters as a dictionary.
            
        Returns
        -------
        Result[object]
            Operation result.
            
        Raises
        ------
        CapabilityError
            If plugin doesn't have required capabilities.
        ValueError
            If operation not found.
        """
        registry = get_default_registry()
        spec = registry.get(operation_id)
        
        if spec is None:
            raise ValueError(f"Unknown operation: {operation_id}")
        
        # Check capabilities
        missing = spec.capabilities - self.manifest.capabilities
        if missing:
            raise CapabilityError(
                next(iter(missing)),
                f"Plugin {self.manifest.name} lacks capabilities: {missing}",
            )
        
        # Build params from dict
        params_obj = self._build_params(spec.params_type, params)
        
        # Build sandboxed context (only plugin's capabilities)
        with (
            OpContextBuilder()
            .with_capabilities(self.manifest.capabilities)
            .build()
        ) as ctx:
            pipeline = get_default_pipeline()
            operation = spec.operation_class()
            return pipeline.execute(operation, params_obj, ctx)
    
    def _build_params(
        self,
        params_type: type[object],
        params_dict: dict[str, object],
    ) -> object:
        """Build params object from dictionary."""
        import dataclasses
        
        if not dataclasses.is_dataclass(params_type):
            return params_type()
        
        kwargs: dict[str, object] = {}
        for fld in dataclasses.fields(params_type):
            if fld.name in params_dict:
                kwargs[fld.name] = params_dict[fld.name]
            elif fld.default is not dataclasses.MISSING:
                kwargs[fld.name] = fld.default
            elif fld.default_factory is not dataclasses.MISSING:
                kwargs[fld.name] = fld.default_factory()
        
        return params_type(**kwargs)
```

---

## 7. Middleware (`middleware/`)

### 7.1 Middleware Protocol (`operations/middleware/base.py`)

```python
"""Base middleware protocol for operation execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.operations.base import Operation
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result


class OperationMiddleware(ABC):
    """Base class for operation middleware.
    
    Middleware wraps operation execution to add cross-cutting concerns.
    Applied to ALL operation invocations regardless of entry point.
    """
    
    @abstractmethod
    def before[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
    ) -> None:
        """Called before operation execution.
        
        May modify context (e.g., start span, set correlation ID).
        """
        ...
    
    @abstractmethod
    def after[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        result: Result[R],
        duration_seconds: float,
    ) -> Result[R]:
        """Called after operation execution.
        
        May modify or wrap the result.
        """
        ...
    
    @abstractmethod
    def on_error[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        error: Exception,
        duration_seconds: float,
    ) -> Result[R] | None:
        """Called when operation raises an exception.
        
        Return a Result to suppress the exception.
        Return None to propagate the exception.
        """
        ...
```

### 7.2 Telemetry Middleware (`operations/middleware/telemetry.py`)

```python
"""Telemetry middleware for spans and metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.operations.middleware.base import OperationMiddleware

if TYPE_CHECKING:
    from codeintel.operations.base import Operation
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result

LOG = logging.getLogger(__name__)


class TelemetryMiddleware(OperationMiddleware):
    """Emit spans and metrics for operation execution.
    
    Creates OpenTelemetry spans (if available) and emits metrics
    for operation duration and success/failure counts.
    """
    
    def __init__(self) -> None:
        self._tracer = _get_tracer()
        self._meter = _get_meter()
    
    def before[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
    ) -> None:
        """Start operation span."""
        if self._tracer is not None:
            span = self._tracer.start_span(
                f"operation.{operation.__operation_id__}",
                attributes={
                    "operation.id": operation.__operation_id__,
                    "operation.capabilities": list(operation.__capabilities__),
                },
            )
            ctx.telemetry.set_span(span)
    
    def after[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        result: Result[R],
        duration_seconds: float,
    ) -> Result[R]:
        """End span and emit metrics."""
        op_id = operation.__operation_id__
        
        # End span
        span = ctx.telemetry.get_span()
        if span is not None:
            span.set_attribute("operation.success", result.success)
            if not result.success and result.error:
                span.set_attribute("operation.error_type", result.error.type)
            span.end()
        
        # Emit metrics
        if self._meter is not None:
            self._meter.record(
                "operation.duration_seconds",
                duration_seconds,
                {"operation": op_id, "success": str(result.success)},
            )
            self._meter.increment(
                "operation.invocations_total",
                {"operation": op_id, "success": str(result.success)},
            )
        
        return result
    
    def on_error[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        error: Exception,
        duration_seconds: float,
    ) -> Result[R] | None:
        """Record error in span."""
        span = ctx.telemetry.get_span()
        if span is not None:
            span.record_exception(error)
            span.set_attribute("operation.success", False)
            span.end()
        
        return None  # Propagate exception


def _get_tracer():
    """Get OpenTelemetry tracer (optional dependency)."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("codeintel.operations")
    except ImportError:
        return None


def _get_meter():
    """Get metrics meter (optional dependency)."""
    try:
        from opentelemetry import metrics
        return metrics.get_meter("codeintel.operations")
    except ImportError:
        return None
```

### 7.3 Logging Middleware (`operations/middleware/logging.py`)

```python
"""Structured logging middleware."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.operations.middleware.base import OperationMiddleware

if TYPE_CHECKING:
    from codeintel.operations.base import Operation
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result


class LoggingMiddleware(OperationMiddleware):
    """Log operation execution with structured fields."""
    
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger("codeintel.operations")
    
    def before[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
    ) -> None:
        """Log operation start."""
        self._log.info(
            "Starting operation",
            extra={
                "operation_id": operation.__operation_id__,
                "capabilities": list(operation.__capabilities__),
            },
        )
    
    def after[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        result: Result[R],
        duration_seconds: float,
    ) -> Result[R]:
        """Log operation completion."""
        if result.success:
            self._log.info(
                "Operation completed",
                extra={
                    "operation_id": operation.__operation_id__,
                    "success": True,
                    "duration_seconds": duration_seconds,
                },
            )
        else:
            error_type = result.error.type if result.error else "unknown"
            self._log.warning(
                "Operation failed",
                extra={
                    "operation_id": operation.__operation_id__,
                    "success": False,
                    "duration_seconds": duration_seconds,
                    "error_type": error_type,
                },
            )
        
        return result
    
    def on_error[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        error: Exception,
        duration_seconds: float,
    ) -> Result[R] | None:
        """Log operation exception."""
        self._log.exception(
            "Operation raised exception",
            extra={
                "operation_id": operation.__operation_id__,
                "duration_seconds": duration_seconds,
                "error_type": type(error).__name__,
            },
        )
        return None  # Propagate
```

### 7.4 Validation Middleware (`operations/middleware/validation.py`)

```python
"""Parameter validation middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.operations.middleware.base import OperationMiddleware
from codeintel.operations.result import Result
from codeintel.operations.errors import ProblemDetail

if TYPE_CHECKING:
    from codeintel.operations.base import Operation
    from codeintel.operations.context import OpContext


class ValidationMiddleware(OperationMiddleware):
    """Validate operation parameters before execution."""
    
    def before[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
    ) -> None:
        """Validate parameters."""
        if hasattr(params, "validate"):
            errors = params.validate()
            if errors:
                raise ValidationError(errors)
    
    def after[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        result: Result[R],
        duration_seconds: float,
    ) -> Result[R]:
        """Pass through result."""
        return result
    
    def on_error[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        error: Exception,
        duration_seconds: float,
    ) -> Result[R] | None:
        """Convert ValidationError to Result."""
        if isinstance(error, ValidationError):
            return Result.fail(
                ProblemDetail(
                    type="urn:codeintel:validation:invalid-params",
                    title="Invalid Parameters",
                    detail="; ".join(error.errors),
                    status=400,
                )
            )
        return None  # Propagate other errors


class ValidationError(Exception):
    """Raised when parameter validation fails."""
    
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")
```

### 7.5 Error Handling Middleware (`operations/middleware/errors.py`)

```python
"""Error handling middleware."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.operations.middleware.base import OperationMiddleware
from codeintel.operations.result import Result
from codeintel.operations.errors import ProblemDetail
from codeintel.operations.context import CapabilityError

if TYPE_CHECKING:
    from codeintel.operations.base import Operation
    from codeintel.operations.context import OpContext

LOG = logging.getLogger(__name__)


class ErrorHandlingMiddleware(OperationMiddleware):
    """Convert exceptions to Result failures.
    
    This should be the outermost middleware to catch all errors.
    """
    
    def before[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
    ) -> None:
        """No-op."""
        pass
    
    def after[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        result: Result[R],
        duration_seconds: float,
    ) -> Result[R]:
        """Pass through result."""
        return result
    
    def on_error[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        error: Exception,
        duration_seconds: float,
    ) -> Result[R] | None:
        """Convert known exceptions to Results."""
        if isinstance(error, CapabilityError):
            return Result.fail(
                ProblemDetail(
                    type="urn:codeintel:security:capability-denied",
                    title="Capability Denied",
                    detail=str(error),
                    status=403,
                )
            )
        
        # Log unexpected errors and convert to generic failure
        LOG.exception("Unexpected error in operation %s", operation.__operation_id__)
        return Result.fail(
            ProblemDetail(
                type="urn:codeintel:internal:unexpected-error",
                title="Internal Error",
                detail=f"An unexpected error occurred: {type(error).__name__}",
                status=500,
            )
        )
```

---

## 8. Result Types (`results/`)

### 8.1 Result Container (`operations/result.py`)

```python
"""Result types for operations."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from codeintel.operations.errors import ProblemDetail


@runtime_checkable
class Serializable(Protocol):
    """Protocol for types that can serialize to dict."""
    
    def to_dict(self) -> dict[str, object]: ...


@dataclass
class Result[T]:
    """Result from an operation.
    
    Wraps success data or failure error with optional metadata.
    
    Example
    -------
    # Success
    return Result.ok(ListResult.from_items(jobs))
    
    # Failure
    return Result.fail(ProblemDetail(
        type="urn:codeintel:jobs:not-found",
        title="Job Not Found",
        detail=f"Job {job_id} not found",
        status=404,
    ))
    """
    
    success: bool
    data: T | None = None
    error: ProblemDetail | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)
    
    @classmethod
    def ok(
        cls,
        data: T,
        *,
        warnings: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Self:
        """Create successful result."""
        return cls(
            success=True,
            data=data,
            warnings=warnings or [],
            metadata=metadata or {},
        )
    
    @classmethod
    def fail(
        cls,
        error: ProblemDetail,
        *,
        warnings: list[str] | None = None,
    ) -> Self:
        """Create failed result."""
        return cls(
            success=False,
            error=error,
            warnings=warnings or [],
        )
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for output."""
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


def result_type[T](cls: type[T]) -> type[T]:
    """Decorator that adds automatic to_dict() to a dataclass.
    
    The generated to_dict():
    - Omits None fields
    - Recursively serializes nested result types
    - Handles lists, dicts, Enums, Paths
    
    Example
    -------
    @result_type
    @dataclass(frozen=True)
    class JobInfo:
        job_id: str
        status: str
        error: str | None = None  # Omitted from dict if None
    """
    if not is_dataclass(cls):
        raise TypeError(f"@result_type requires a dataclass, got {cls}")
    
    def to_dict(self: object) -> dict[str, object]:
        result: dict[str, object] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if value is None:
                continue
            result[f.name] = _serialize_value(value)
        return result
    
    cls.to_dict = to_dict  # type: ignore[attr-defined]
    cls._result_type_generated = True  # type: ignore[attr-defined]
    return cls


def _serialize_value(value: object) -> object:
    """Recursively serialize a value."""
    if isinstance(value, Serializable):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {
            f.name: _serialize_value(getattr(value, f.name))
            for f in fields(value)
        }
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
```

### 8.2 Generic Result Types (`operations/result_types.py`)

```python
"""Reusable generic result types.

These cover the most common operation output patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.operations.result import result_type


@result_type
@dataclass(frozen=True)
class ListResult[T]:
    """Generic list result with count.
    
    Use for: jobs.list, datasets.list, plugins.list, etc.
    """
    
    items: list[T]
    count: int
    
    @classmethod
    def from_items(cls, items: list[T]) -> ListResult[T]:
        """Create from item list, auto-computing count."""
        return cls(items=items, count=len(items))
    
    @classmethod
    def empty(cls) -> ListResult[T]:
        """Create empty list result."""
        return cls(items=[], count=0)


@result_type
@dataclass(frozen=True)
class PagedResult[T]:
    """Paginated list result.
    
    Use when operations support pagination.
    """
    
    items: list[T]
    count: int
    total: int
    offset: int
    limit: int
    has_more: bool
    
    @classmethod
    def from_items(
        cls,
        items: list[T],
        *,
        total: int,
        offset: int,
        limit: int,
    ) -> PagedResult[T]:
        """Create from paginated query results."""
        return cls(
            items=items,
            count=len(items),
            total=total,
            offset=offset,
            limit=limit,
            has_more=offset + len(items) < total,
        )


@result_type
@dataclass(frozen=True)
class StatusResult:
    """Generic status result.
    
    Use for: health checks, connectivity tests, etc.
    """
    
    status: str  # "ok", "warning", "error", "unknown"
    message: str
    details: dict[str, object] | None = None


@result_type
@dataclass(frozen=True)
class ActionResult:
    """Result from a state-modifying action.
    
    Use for: jobs.cancel, storage.migrate, plugins.enable, etc.
    """
    
    action: str
    success: bool
    affected_count: int = 0
    message: str | None = None


@result_type
@dataclass(frozen=True)
class ExportResult:
    """Result from an export/generation operation.
    
    Use for: docs.generate, datasets.snapshot, etc.
    """
    
    output_path: str
    item_count: int
    format: str | None = None
    duration_seconds: float | None = None


@result_type
@dataclass(frozen=True)
class LookupResult[T]:
    """Result from looking up a single item.
    
    Use for: jobs.status, datasets.describe, etc.
    """
    
    item: T
    found: bool = True


@result_type
@dataclass(frozen=True)
class ValidationResult:
    """Result from a validation operation.
    
    Use for: datasets.lint, build.validate, etc.
    """
    
    valid: bool
    errors: list[str]
    warnings: list[str]
    checked_count: int = 0
```

---

## 9. Errors (`errors/`)

### 9.1 ProblemDetail (`operations/errors/problem_detail.py`)

```python
"""RFC 9457 Problem Details for HTTP APIs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemDetail:
    """RFC 9457 Problem Details.
    
    Standard error representation used across all adapters.
    HTTP adapter maps to response; CLI renders as error message.
    
    Example
    -------
    ProblemDetail(
        type="urn:codeintel:jobs:not-found",
        title="Job Not Found",
        detail="Job abc-123 was not found in the job registry",
        status=404,
    )
    """
    
    type: str
    title: str
    detail: str
    status: int
    instance: str | None = None
    extensions: dict[str, object] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, object]:
        """Serialize for output."""
        result: dict[str, object] = {
            "type": self.type,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
        }
        if self.instance is not None:
            result["instance"] = self.instance
        result.update(self.extensions)
        return result
```

### 9.2 Error Factories (`operations/errors/factory.py`)

```python
"""Convenience factory functions for common errors."""

from __future__ import annotations

from codeintel.operations.result import Result
from codeintel.operations.errors.problem_detail import ProblemDetail


def fail_not_found[R](
    resource: str,
    identifier: str,
    *,
    domain: str = "resource",
) -> Result[R]:
    """Create failure for resource not found."""
    return Result.fail(
        ProblemDetail(
            type=f"urn:codeintel:{domain}:not-found",
            title=f"{resource.title()} Not Found",
            detail=f"{resource.title()} not found: {identifier}",
            status=404,
        )
    )


def fail_validation[R](
    message: str,
    *,
    errors: list[str] | None = None,
) -> Result[R]:
    """Create failure for validation error."""
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:validation:invalid-params",
            title="Validation Error",
            detail=message,
            status=400,
            extensions={"errors": errors} if errors else {},
        )
    )


def fail_missing_required[R](param: str) -> Result[R]:
    """Create failure for missing required parameter."""
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:validation:missing-required",
            title="Missing Required Parameter",
            detail=f"Required parameter '{param}' was not provided",
            status=400,
        )
    )


def fail_capability_denied[R](
    capability: str,
    *,
    operation: str | None = None,
) -> Result[R]:
    """Create failure for denied capability."""
    detail = f"Required capability '{capability}' is not granted"
    if operation:
        detail = f"Operation '{operation}' requires capability '{capability}'"
    
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:security:capability-denied",
            title="Capability Denied",
            detail=detail,
            status=403,
        )
    )


def fail_internal[R](
    message: str,
    *,
    error_type: str | None = None,
) -> Result[R]:
    """Create failure for internal error."""
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:internal:error",
            title="Internal Error",
            detail=message,
            status=500,
            extensions={"error_type": error_type} if error_type else {},
        )
    )


# Domain-specific factories

def fail_job_not_found[R](job_id: str) -> Result[R]:
    """Create failure for job not found."""
    return fail_not_found("job", job_id, domain="jobs")


def fail_job_not_completed[R](job_id: str, current_status: str) -> Result[R]:
    """Create failure for job not in completed state."""
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:not-completed",
            title="Job Not Completed",
            detail=f"Job {job_id} is {current_status}, not completed",
            status=409,
        )
    )


def fail_dataset_not_found[R](table_key: str) -> Result[R]:
    """Create failure for dataset not found."""
    return fail_not_found("dataset", table_key, domain="datasets")
```

---

## 10. Registry (`registry/`)

### 10.1 Operation Registry (`operations/registry.py`)

```python
"""Operation registry for discovery and lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec


@dataclass
class OperationRegistry:
    """Registry for all operations.
    
    Populated at import time via @operation decorator.
    Provides lookup and discovery for adapters.
    """
    
    _operations: dict[str, OperationSpec] = field(default_factory=dict)
    
    def register(self, spec: OperationSpec) -> OperationSpec:
        """Register an operation."""
        if spec.operation_id in self._operations:
            raise ValueError(f"Operation already registered: {spec.operation_id}")
        self._operations[spec.operation_id] = spec
        return spec
    
    def get(self, operation_id: str) -> OperationSpec | None:
        """Get operation by ID."""
        return self._operations.get(operation_id)
    
    def list_operations(
        self,
        *,
        group: str | None = None,
        capabilities: frozenset[str] | None = None,
        include_hidden: bool = False,
        tags: tuple[str, ...] | None = None,
    ) -> list[OperationSpec]:
        """List operations with optional filters."""
        ops = list(self._operations.values())
        
        if group is not None:
            ops = [op for op in ops if op.group == group]
        
        if capabilities is not None:
            # Filter to operations whose capabilities are subset
            ops = [op for op in ops if op.capabilities <= capabilities]
        
        if not include_hidden:
            ops = [op for op in ops if not op.hidden]
        
        if tags is not None:
            ops = [op for op in ops if any(t in op.tags for t in tags)]
        
        return sorted(ops, key=lambda op: op.operation_id)
    
    def list_groups(self) -> list[str]:
        """List all operation groups."""
        return sorted({op.group for op in self._operations.values()})
    
    def list_capabilities(self) -> list[str]:
        """List all required capabilities across operations."""
        caps: set[str] = set()
        for op in self._operations.values():
            caps.update(op.capabilities)
        return sorted(caps)


# Module-level default registry
_DEFAULT_REGISTRY = OperationRegistry()


def get_default_registry() -> OperationRegistry:
    """Get the default registry."""
    return _DEFAULT_REGISTRY


def create_isolated_registry() -> OperationRegistry:
    """Create an isolated registry for testing."""
    return OperationRegistry()
```

---

## 11. CLI Integration (`cli/`)

### 11.1 Minimal CLI Shell (`cli/app.py`)

```python
"""CLI application entry point.

The CLI is now a thin shell that delegates to operations.
"""

from __future__ import annotations

from cyclopts import App

from codeintel.operations.adapters.cli import CliAdapter
from codeintel.cli.rendering import configure_renderer


def create_app() -> App:
    """Create the CLI application."""
    app = App(
        name="codeintel",
        help="CodeIntel unified CLI for build, analytics, and serving operations.",
        result_action=["call_if_callable", "return_value"],
        print_error=True,
    )
    
    # Register all operations as CLI commands
    adapter = CliAdapter(app)
    adapter.register_all()
    
    # Add CLI-only commands (completions, help)
    _add_cli_specific_commands(app)
    
    return app


def _add_cli_specific_commands(app: App) -> None:
    """Add commands that are CLI-specific (not operations)."""
    from codeintel.cli.commands.completions import completions_app
    
    app.command(completions_app)


def main() -> None:
    """CLI entry point."""
    configure_renderer()
    app = create_app()
    app()


if __name__ == "__main__":
    main()
```

---

## 12. Plugin Integration

### 12.1 Plugin Operation Registration (`plugins/registry.py`)

```python
"""Plugin operation registration.

Plugins can register operations that go through the same pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.operations.adapters.plugin import PluginAdapter
from codeintel.operations.registry import get_default_registry

if TYPE_CHECKING:
    from codeintel.operations.base import Operation
    from codeintel.plugins.manifest import PluginManifest


@dataclass
class PluginOperationRegistry:
    """Registry for plugin-provided operations.
    
    Plugins register operations that are namespaced and capability-gated.
    """
    
    manifest: PluginManifest
    
    def register[P, R](self, operation_class: type[Operation[P, R]]) -> None:
        """Register a plugin operation.
        
        The operation ID must be prefixed with the plugin name.
        The operation's capabilities must be subset of plugin's capabilities.
        
        Parameters
        ----------
        operation_class
            Operation class to register.
            
        Raises
        ------
        ValueError
            If operation ID not prefixed with plugin name.
            If operation requires capabilities the plugin doesn't have.
        """
        op_id = operation_class.__operation_id__
        
        # Validate prefix
        if not op_id.startswith(f"{self.manifest.name}."):
            raise ValueError(
                f"Plugin operation ID must start with '{self.manifest.name}.', "
                f"got '{op_id}'"
            )
        
        # Validate capabilities
        required = operation_class.__capabilities__
        granted = self.manifest.capabilities
        missing = required - granted
        
        if missing:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register operation "
                f"'{op_id}' because it requires capabilities {missing} "
                f"which are not granted"
            )
        
        # Register with main registry
        from codeintel.operations.base import OperationSpec
        
        spec = OperationSpec(
            operation_id=op_id,
            name=operation_class.__name__,
            description=operation_class.__doc__ or "",
            params_type=operation_class.__params_type__,
            result_type=operation_class.__result_type__,
            operation_class=operation_class,
            group=self.manifest.name,
            capabilities=required,
            tags=("plugin", self.manifest.name),
        )
        
        get_default_registry().register(spec)
```

### 12.2 Example Plugin (`plugins/examples/hello_plugin.py`)

```python
"""Example plugin demonstrating operation registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.operations.base import Operation, operation
from codeintel.operations.result import Result, result_type
from codeintel.plugins.manifest import PluginManifest

if TYPE_CHECKING:
    from codeintel.operations.context import OpContext


# Plugin manifest
MANIFEST = PluginManifest(
    name="hello",
    version="1.0.0",
    description="Example hello world plugin",
    capabilities=frozenset(),  # No special capabilities needed
)


# Result type
@result_type
@dataclass(frozen=True)
class GreetingResult:
    """Greeting result."""
    
    message: str
    greeted: str


# Parameters
@dataclass(frozen=True)
class GreetParams:
    """Parameters for hello.greet operation."""
    
    name: str = "World"
    
    class CliMeta:
        from cyclopts import Parameter
        name = Parameter(help="Name to greet")


# Operation
@operation("hello.greet")
class Greet(Operation[GreetParams, GreetingResult]):
    """Greet someone by name."""
    
    def execute(
        self,
        params: GreetParams,
        ctx: OpContext,
    ) -> Result[GreetingResult]:
        """Execute the greeting."""
        message = f"Hello, {params.name}!"
        return Result.ok(GreetingResult(message=message, greeted=params.name))


# Plugin registration function
def register(registry) -> None:
    """Register plugin operations."""
    # Operations auto-register via @operation decorator
    # This function can do additional setup if needed
    pass
```

---

## 13. Testing Patterns

### 13.1 Testing Operations Directly

```python
"""Test operations without any adapter infrastructure."""

import pytest
from dataclasses import dataclass

from codeintel.operations.context import OpContext, OpConfig
from codeintel.operations.result import Result


# Create fake implementations for testing

class FakeJobsAccess:
    """Fake jobs access for testing."""
    
    def __init__(self, jobs: list | None = None) -> None:
        self._jobs = jobs or []
    
    def list(self, *, status=None, limit=20):
        result = self._jobs
        if status is not None:
            result = [j for j in result if j.status == status]
        return result[:limit]
    
    def get(self, job_id: str):
        return next((j for j in self._jobs if j.job_id == job_id), None)


class FakeTelemetry:
    """Fake telemetry for testing."""
    
    def span(self, name: str):
        return nullcontext()
    
    def get_span(self):
        return None
    
    def set_span(self, span):
        pass


@pytest.fixture
def test_context() -> OpContext:
    """Create test context with fakes."""
    return OpContext(
        config=OpConfig(),
        logger=logging.getLogger("test"),
        jobs=FakeJobsAccess([
            FakeJob("j1", "test.op", "running"),
            FakeJob("j2", "test.op", "completed"),
        ]),
        telemetry=FakeTelemetry(),
    )


def test_list_jobs_returns_all(test_context: OpContext) -> None:
    """Test listing all jobs."""
    from codeintel.operations.catalog.jobs import ListJobs, ListJobsParams
    
    operation = ListJobs()
    params = ListJobsParams(limit=10)
    
    result = operation.execute(params, test_context)
    
    assert result.success
    assert result.data is not None
    assert result.data.count == 2


def test_list_jobs_filters_by_status(test_context: OpContext) -> None:
    """Test filtering by status."""
    from codeintel.operations.catalog.jobs import ListJobs, ListJobsParams
    
    operation = ListJobs()
    params = ListJobsParams(status="running", limit=10)
    
    result = operation.execute(params, test_context)
    
    assert result.success
    assert result.data.count == 1
    assert result.data.items[0].status == "running"
```

### 13.2 Testing with Pipeline

```python
"""Test operations through the middleware pipeline."""

from codeintel.operations.pipeline import OperationPipeline
from codeintel.operations.middleware.validation import ValidationMiddleware
from codeintel.operations.middleware.logging import LoggingMiddleware


def test_validation_middleware_rejects_invalid_params(test_context):
    """Test that validation middleware catches invalid params."""
    from codeintel.operations.catalog.jobs import ListJobs, ListJobsParams
    
    pipeline = OperationPipeline([
        ValidationMiddleware(),
        LoggingMiddleware(),
    ])
    
    operation = ListJobs()
    params = ListJobsParams(limit=-1)  # Invalid!
    
    result = pipeline.execute(operation, params, test_context)
    
    assert not result.success
    assert "limit must be >= 1" in result.error.detail
```

### 13.3 Testing Adapters

```python
"""Test the CLI adapter."""

from codeintel.operations.adapters.cli import CliAdapter, _extract_params
from codeintel.operations.registry import create_isolated_registry


def test_cli_adapter_extracts_params():
    """Test parameter extraction from CLI command."""
    @dataclass(frozen=True)
    class TestParams:
        name: str = "default"
        limit: int = 20
    
    @dataclass(frozen=True)
    class FakeCommand:
        name: str = "custom"
        limit: int = 50
        _verbose: int = 0
        _json: bool = False
    
    cmd = FakeCommand()
    params = _extract_params(cmd, TestParams)
    
    assert params.name == "custom"
    assert params.limit == 50
```

---

## 14. Migration Strategy

### Phase 1: Foundation (2-3 days)

**Goal**: Create operations infrastructure alongside existing CLI.

1. Create `src/codeintel/operations/` package structure
2. Implement core types: `Operation`, `OpContext`, `Result`
3. Implement middleware base and pipeline
4. Implement basic middleware (logging, validation, errors)
5. Create CLI adapter skeleton

**Validation**: New package imports cleanly, no impact on existing code.

### Phase 2: Pilot Operations (2-3 days)

**Goal**: Migrate one operation group to validate the pattern.

1. Migrate `jobs` operations to new pattern
2. Wire CLI adapter to generate commands
3. Verify both old and new CLI work
4. Write comprehensive tests

**Validation**: `codeintel jobs list` works via new pipeline.

### Phase 3: Full Migration (5-7 days)

**Goal**: Migrate all operations.

Order by complexity:
1. **Simple**: `health`, `plugins`, `completions`
2. **Read-only storage**: `datasets.list`, `storage.info`, `graphs.stats`
3. **Write operations**: `jobs.cancel`, `storage.migrate`
4. **Complex**: `build`, `docs`, `serve`

### Phase 4: Adapter Expansion (2-3 days)

**Goal**: Wire HTTP and plugin adapters.

1. Implement HTTP adapter for FastAPI
2. Update plugin sandbox to use PluginAdapter
3. Wire MCP server to operation pipeline

### Phase 5: Cleanup (2-3 days)

**Goal**: Remove legacy code.

1. Remove `HandlerContext` and related infrastructure
2. Remove old handler files from `cli/handlers/`
3. Remove compatibility shims from `cli/deps/compat.py`
4. Update all documentation

---

## Summary

This unified operation architecture provides:

1. **Single source of truth**: One operation implementation, multiple entry points
2. **Universal middleware**: Same logging/metrics/tracing everywhere
3. **Capability-based security**: Natural sandboxing for plugins
4. **Type safety end-to-end**: Params → Operation → Result, all typed
5. **Clean testing**: Test operations directly without adapters
6. **Future-proof**: Easy to add new entry points (MCP, webhooks, queues)

The key insight is that **operations are the product**—CLI, HTTP, and plugins are just delivery mechanisms. By centering the architecture on operations rather than CLI commands, we achieve a fundamentally simpler and more maintainable design.
