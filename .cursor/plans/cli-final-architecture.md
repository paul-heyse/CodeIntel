# CLI Architecture: Final State Specification

> **Document Version:** 1.0  
> **Created:** 2024-12-09  
> **Status:** Approved for Implementation  
> **Supersedes:** Phase B/C compatibility-focused approach

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Final Module Structure](#2-final-module-structure)
3. [Component Specifications](#3-component-specifications)
   - [execution/types.py](#31-executiontypespy)
   - [execution/context.py](#32-executioncontextpy)
   - [execution/middleware.py](#33-executionmiddlewarepy)
   - [execution/progress.py](#34-executionprogresspy)
   - [execution/executor.py](#35-executionexecutorpy)
   - [plugins/](#36-plugins-package)
4. [Consumer Updates](#4-consumer-updates)
5. [Files to Delete](#5-files-to-delete)
6. [Migration Implementation Plan](#6-migration-implementation-plan)
7. [Acceptance Criteria](#7-acceptance-criteria)

---

## 1. Design Philosophy

### 1.1 Core Principles

1. **`ExecutionContext` as the Universal Currency** — All middleware, handlers, and executors operate on a rich context object, not scattered parameters
2. **Single Implementation Per Concept** — One middleware protocol, one progress tracker, one executor
3. **Explicit Over Implicit** — No magic globals; dependencies are injected or passed explicitly
4. **Async-First with Sync Compatibility** — All middleware supports async; sync is automatic fallback
5. **Zero Legacy Code** — No shims, no deprecated modules, no backward compatibility layers

### 1.2 Why ExecutionContext is Superior

| Aspect | Legacy `(op_id, params)` | `ExecutionContext` |
|--------|--------------------------|-------------------|
| **Extensibility** | New fields require signature changes | Add fields without breaking existing code |
| **Type Safety** | Dict types lose information | Full dataclass typing |
| **Cancellation** | Separate mechanism | Built-in `check_cancelled()` |
| **Tracing** | Manual correlation ID passing | Embedded in context |
| **Progress** | Separate callback parameter | Embedded progress callback |
| **Metadata** | Nowhere to put it | `metadata: dict[str, Any]` |
| **Async/Sync** | Two interfaces | One interface, async methods with sync fallback |

### 1.3 Key Decisions

- **No deprecation warnings** — We are pre-production; clean cut to final state
- **No compatibility shims** — All legacy code is deleted, not wrapped
- **Move content, don't re-export** — Files in `plugins/` contain actual code, not imports from elsewhere

---

## 2. Final Module Structure

```
src/codeintel/cli/
│
├── execution/                          # CORE: Unified execution infrastructure
│   ├── __init__.py                    # Public API (all exports)
│   ├── types.py                       # Handler types, progress events, enums
│   ├── context.py                     # ExecutionContext, ExecutionResult
│   ├── middleware.py                  # Middleware protocol + implementations
│   ├── progress.py                    # ProgressTracker (sync + async)
│   └── executor.py                    # OperationSpec, UnifiedOperationExecutor
│
├── plugins/                            # CORE: Unified plugin infrastructure
│   ├── __init__.py                    # Public API
│   ├── manifest.py                    # PluginManifest, SemanticVersion, PluginCapability
│   ├── sandbox.py                     # PluginSandbox, SandboxedImporter, SandboxConfig
│   ├── discovery.py                   # discover_plugins, DiscoveredPlugin
│   ├── loader.py                      # PluginLoader, LoadedPlugin, PluginLoadResult
│   ├── registry.py                    # register_plugin_operations, PluginManager
│   └── testing.py                     # PluginTestHarness, create_plugin_scaffold
│
├── resilience.py                       # UPDATED: Uses execution.Middleware
├── observability.py                    # UPDATED: Uses execution.Middleware
├── telemetry.py                        # Unchanged (no middleware dependency)
├── cli_errors.py                       # Unchanged (ProblemDetail)
├── results.py                          # Unchanged (CliResult)
├── cli_render.py                       # Unchanged (OutputRenderer)
│
├── operation_registry.py               # UPDATED: Uses execution.OperationSpec
├── operations/                         # UPDATED: All use execution.OperationSpec
│   ├── build_operations.py
│   ├── dataset_operations.py
│   ├── docs_operations.py
│   ├── graph_operations.py
│   ├── history_operations.py
│   ├── ide_operations.py
│   ├── op_operations.py
│   ├── storage_operations.py
│   └── subsystem_operations.py
│
├── cyclopts_app.py                     # UPDATED: Uses execution.*
├── cyclopts_ops.py                     # UPDATED: Uses execution.get_middleware_stack
├── cyclopts_*.py                       # UPDATED: Various cyclopts modules
│
├── introspection.py                    # UPDATED: Uses execution.OperationSpec
├── pipelines.py                        # UPDATED: Uses execution.get_executor
├── shell.py                            # UPDATED: Uses execution.get_executor
└── job_runner.py                       # UPDATED: Uses execution.get_executor
```

### 2.1 Deleted Files (No Longer Exist)

```
src/codeintel/cli/
├── executor.py                    # DELETED: absorbed into execution/
├── async_executor.py              # DELETED: absorbed into execution/executor.py
├── async_types.py                 # DELETED: absorbed into execution/types.py
├── cli_middleware.py              # DELETED: absorbed into execution/middleware.py
├── async_middleware.py            # DELETED: absorbed into execution/middleware.py
├── cli_progress.py                # DELETED: absorbed into execution/progress.py
├── async_progress.py              # DELETED: absorbed into execution/progress.py
├── plugins.py                     # DELETED: replaced by plugins/
├── plugin_manifest.py             # DELETED: moved to plugins/manifest.py
├── plugin_sandbox.py              # DELETED: moved to plugins/sandbox.py
└── plugin_testing.py              # DELETED: moved to plugins/testing.py
```

---

## 3. Component Specifications

### 3.1 `execution/types.py`

```python
"""Unified type definitions for CLI execution pipeline."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from codeintel.cli.results import CliResult

T = TypeVar("T")


class ProgressState(Enum):
    """Progress state for operations."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProgressEvent:
    """Progress update from an operation."""
    operation_id: str
    state: ProgressState
    progress: float | None = None          # 0.0 to 1.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    items_completed: int | None = None
    items_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operation_id": self.operation_id,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "items_completed": self.items_completed,
            "items_total": self.items_total,
        }


@dataclass
class StreamingResult[T]:
    """Container for streaming results (progress or final)."""
    progress: ProgressEvent | None = None
    result: CliResult[T] | None = None

    @property
    def is_progress(self) -> bool:
        """Check if this is a progress event."""
        return self.progress is not None

    @property
    def is_result(self) -> bool:
        """Check if this is a final result."""
        return self.result is not None


@dataclass
class ProgressConfig:
    """Configuration for progress reporting."""
    enabled: bool = True
    verbose: bool = False
    refresh_rate: float = 10.0
    show_spinner: bool = True
    update_interval: float = 0.1
    format_string: str = "{message} [{current}/{total}]"


# Handler type aliases using Python 3.12+ syntax
type SyncHandler[T] = Callable[..., CliResult[T]]
type AsyncHandler[T] = Callable[..., Awaitable[CliResult[T]]]
type StreamingHandler[T] = Callable[..., AsyncGenerator[StreamingResult[T], None]]
type AnyHandler[T] = SyncHandler[T] | AsyncHandler[T] | StreamingHandler[T]

type ProgressCallback = Callable[[ProgressEvent], None]


def is_async_handler(handler: object) -> bool:
    """Check if handler is async."""
    return asyncio.iscoroutinefunction(handler) or (
        callable(handler) and asyncio.iscoroutinefunction(type(handler).__call__)
    )


def is_streaming_handler(handler: object) -> bool:
    """Check if handler is a streaming (async generator) handler."""
    return inspect.isasyncgenfunction(handler) or (
        callable(handler) and inspect.isasyncgenfunction(type(handler).__call__)
    )


def get_handler_type(handler: object) -> str:
    """Determine handler type: 'sync', 'async', or 'streaming'."""
    if is_streaming_handler(handler):
        return "streaming"
    if is_async_handler(handler):
        return "async"
    return "sync"
```

---

### 3.2 `execution/context.py`

```python
"""Execution context and result types."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from codeintel.cli.execution.types import ProgressCallback, ProgressEvent, ProgressState


@dataclass
class ExecutionContext:
    """Rich context for operation execution.
    
    This is the single parameter passed to all middleware hooks,
    providing all information needed for cross-cutting concerns.
    """
    
    # Core identification
    operation_id: str
    params: dict[str, Any]
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Tracing and correlation
    trace_id: str = ""
    correlation_id: str = ""
    
    # Metadata for middleware to share data
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Cancellation support
    cancellation_event: asyncio.Event | None = None
    
    # Progress reporting
    progress_callback: ProgressCallback | None = None
    
    @property
    def is_async(self) -> bool:
        """Check if running in async context."""
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False
    
    def check_cancelled(self) -> None:
        """Check if cancelled and raise if so.
        
        Raises
        ------
        asyncio.CancelledError
            If cancellation has been requested.
        """
        if self.cancellation_event is not None and self.cancellation_event.is_set():
            msg = f"Operation {self.operation_id} was cancelled"
            raise asyncio.CancelledError(msg)
    
    async def check_cancelled_async(self) -> None:
        """Async version of check_cancelled.
        
        Raises
        ------
        asyncio.CancelledError
            If cancellation has been requested.
        """
        self.check_cancelled()
    
    def report_progress(
        self,
        *,
        progress: float | None = None,
        message: str = "",
        items_completed: int | None = None,
        items_total: int | None = None,
    ) -> None:
        """Report progress through callback if configured."""
        if self.progress_callback is not None:
            event = ProgressEvent(
                operation_id=self.operation_id,
                state=ProgressState.RUNNING,
                progress=progress,
                message=message,
                items_completed=items_completed,
                items_total=items_total,
            )
            self.progress_callback(event)
    
    @classmethod
    def for_operation(
        cls,
        operation_id: str,
        params: dict[str, Any],
        *,
        trace_id: str = "",
        correlation_id: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionContext:
        """Create context for an operation."""
        return cls(
            operation_id=operation_id,
            params=params,
            trace_id=trace_id,
            correlation_id=correlation_id,
            progress_callback=progress_callback,
            cancellation_event=asyncio.Event(),
        )


@dataclass
class ExecutionResult[T]:
    """Result of operation execution with metadata."""
    
    result: T | None = None
    duration_seconds: float = 0.0
    was_cancelled: bool = False
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

### 3.3 `execution/middleware.py`

```python
"""Unified middleware protocol and implementations."""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


class Middleware(ABC):
    """Base class for operation execution middleware.
    
    All middleware receives ExecutionContext, enabling rich cross-cutting
    concerns with access to operation ID, params, tracing, progress, etc.
    
    Async methods have default implementations that call sync versions,
    so most middleware only needs to implement the sync methods.
    """
    
    @abstractmethod
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Execute before operation invocation.
        
        Parameters
        ----------
        ctx
            Execution context with operation_id, params, metadata, etc.
        
        Returns
        -------
        dict[str, Any]
            Middleware-specific context to pass to after_invoke/on_error.
        """
        ...
    
    @abstractmethod
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Execute after successful operation invocation.
        
        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_ctx
            Context returned from before_invoke.
        """
        ...
    
    @abstractmethod
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Execute on operation error.
        
        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_ctx
            Context returned from before_invoke.
        """
        ...
    
    # Async versions with sync fallback
    async def before_invoke_async(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Async before_invoke (defaults to sync)."""
        return self.before_invoke(ctx)
    
    async def after_invoke_async(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Async after_invoke (defaults to sync)."""
        self.after_invoke(ctx, result, mw_ctx)
    
    async def on_error_async(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Async on_error (defaults to sync)."""
        self.on_error(ctx, exc, mw_ctx)


@dataclass
class MiddlewareStack:
    """Stack of middleware to execute around operations."""
    
    middleware: list[Middleware] = field(default_factory=list)
    
    def add(self, mw: Middleware) -> None:
        """Add middleware to the stack."""
        self.middleware.append(mw)
    
    @contextmanager
    def wrap(self, ctx: ExecutionContext) -> Iterator[None]:
        """Wrap sync operation execution with middleware.
        
        Parameters
        ----------
        ctx
            Execution context.
        
        Yields
        ------
        None
            Control to the wrapped operation.
        """
        contexts: list[dict[str, Any]] = []
        
        # Before hooks
        for mw in self.middleware:
            mw_ctx = mw.before_invoke(ctx)
            contexts.append(mw_ctx)
        
        try:
            yield
        except Exception as exc:
            # Error hooks (reverse order)
            for mw, mw_ctx in zip(
                reversed(self.middleware),
                reversed(contexts),
                strict=True,
            ):
                with contextlib.suppress(Exception):
                    mw.on_error(ctx, exc, mw_ctx)
            raise
        else:
            # After hooks (reverse order)
            for mw, mw_ctx in zip(
                reversed(self.middleware),
                reversed(contexts),
                strict=True,
            ):
                mw.after_invoke(ctx, None, mw_ctx)
    
    @asynccontextmanager
    async def wrap_async(self, ctx: ExecutionContext) -> AsyncIterator[None]:
        """Wrap async operation execution with middleware.
        
        Parameters
        ----------
        ctx
            Execution context.
        
        Yields
        ------
        None
            Control to the wrapped operation.
        """
        contexts: list[dict[str, Any]] = []
        
        # Before hooks (async)
        for mw in self.middleware:
            mw_ctx = await mw.before_invoke_async(ctx)
            contexts.append(mw_ctx)
        
        try:
            yield
        except Exception as exc:
            # Error hooks (reverse order, async)
            for mw, mw_ctx in zip(
                reversed(self.middleware),
                reversed(contexts),
                strict=True,
            ):
                with contextlib.suppress(Exception):
                    await mw.on_error_async(ctx, exc, mw_ctx)
            raise
        else:
            # After hooks (reverse order, async)
            for mw, mw_ctx in zip(
                reversed(self.middleware),
                reversed(contexts),
                strict=True,
            ):
                await mw.after_invoke_async(ctx, None, mw_ctx)


# =============================================================================
# Concrete Middleware Implementations
# =============================================================================


class LoggingMiddleware(Middleware):
    """Log operation execution details."""
    
    def __init__(self, *, log_params: bool = False) -> None:
        """Initialize with optional param logging."""
        self._log_params = log_params
    
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Log operation start."""
        extra: dict[str, Any] = {
            "operation_id": ctx.operation_id,
            "trace_id": ctx.trace_id,
        }
        if self._log_params:
            extra["params"] = ctx.params
        LOG.info("Starting operation %s", ctx.operation_id, extra=extra)
        return {"start_time": time.monotonic()}
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Log operation completion."""
        duration = time.monotonic() - mw_ctx["start_time"]
        LOG.info(
            "Operation %s completed in %.3fs",
            ctx.operation_id,
            duration,
            extra={
                "operation_id": ctx.operation_id,
                "duration_seconds": duration,
                "trace_id": ctx.trace_id,
            },
        )
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Log operation error."""
        duration = time.monotonic() - mw_ctx.get("start_time", 0)
        LOG.error(
            "Operation %s failed after %.3fs: %s",
            ctx.operation_id,
            duration,
            exc,
            extra={
                "operation_id": ctx.operation_id,
                "duration_seconds": duration,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "trace_id": ctx.trace_id,
            },
        )


class TimingMiddleware(Middleware):
    """Track operation timing in context metadata."""
    
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Record start time."""
        return {"start_time": time.monotonic()}
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Record duration in metadata."""
        duration = time.monotonic() - mw_ctx["start_time"]
        ctx.metadata["duration_seconds"] = duration
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Record duration even on error."""
        duration = time.monotonic() - mw_ctx.get("start_time", 0)
        ctx.metadata["duration_seconds"] = duration


class MetricsMiddleware(Middleware):
    """Collect operation metrics."""
    
    def __init__(self) -> None:
        """Initialize metrics storage."""
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._durations: dict[str, list[float]] = {}
    
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Record operation start."""
        return {"start_time": time.monotonic()}
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Record success metrics."""
        op_id = ctx.operation_id
        duration = time.monotonic() - mw_ctx["start_time"]
        
        self._counts[op_id] = self._counts.get(op_id, 0) + 1
        if op_id not in self._durations:
            self._durations[op_id] = []
        self._durations[op_id].append(duration)
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Record error metrics."""
        op_id = ctx.operation_id
        self._errors[op_id] = self._errors.get(op_id, 0) + 1
    
    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics."""
        return {
            "operation_counts": dict(self._counts),
            "operation_errors": dict(self._errors),
            "operation_durations": {
                op_id: {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations) if durations else 0,
                }
                for op_id, durations in self._durations.items()
            },
        }


class TracingMiddleware(Middleware):
    """OpenTelemetry tracing integration."""
    
    def __init__(self) -> None:
        """Initialize tracing."""
        self._has_otel = False
        self._trace: Any = None
        try:
            from opentelemetry import trace
            self._trace = trace
            self._has_otel = True
        except ImportError:
            pass
    
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Start trace span."""
        if not self._has_otel:
            return {}
        
        tracer = self._trace.get_tracer("codeintel.cli")
        span = tracer.start_span(f"cli.{ctx.operation_id}")
        span.set_attribute("operation.id", ctx.operation_id)
        span.set_attribute("operation.trace_id", ctx.trace_id)
        
        return {"span": span}
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """End span with success."""
        span = mw_ctx.get("span")
        if span is not None:
            span.set_attribute("success", True)
            span.end()
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """End span with error."""
        span = mw_ctx.get("span")
        if span is not None:
            span.record_exception(exc)
            span.set_attribute("error", True)
            span.end()


class ProgressMiddleware(Middleware):
    """Emit progress events at operation boundaries."""
    
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        """Emit start event."""
        ctx.report_progress(progress=0.0, message=f"Starting {ctx.operation_id}")
        return {}
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Emit completion event."""
        ctx.report_progress(progress=1.0, message="Completed")
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        """Emit failure event."""
        if ctx.progress_callback is not None:
            from codeintel.cli.execution.types import ProgressEvent, ProgressState
            event = ProgressEvent(
                operation_id=ctx.operation_id,
                state=ProgressState.FAILED,
                message=f"Failed: {exc}",
            )
            ctx.progress_callback(event)


# =============================================================================
# Global Middleware Stack
# =============================================================================

_MIDDLEWARE_STACK: MiddlewareStack | None = None


def get_middleware_stack() -> MiddlewareStack:
    """Get the global middleware stack."""
    global _MIDDLEWARE_STACK  # noqa: PLW0603
    if _MIDDLEWARE_STACK is None:
        _MIDDLEWARE_STACK = MiddlewareStack()
    return _MIDDLEWARE_STACK


def configure_middleware_stack(stack: MiddlewareStack) -> None:
    """Configure the global middleware stack."""
    global _MIDDLEWARE_STACK  # noqa: PLW0603
    _MIDDLEWARE_STACK = stack


def configure_default_middleware() -> None:
    """Configure default middleware (logging + timing)."""
    stack = get_middleware_stack()
    if not any(isinstance(mw, LoggingMiddleware) for mw in stack.middleware):
        stack.add(LoggingMiddleware())
    if not any(isinstance(mw, TimingMiddleware) for mw in stack.middleware):
        stack.add(TimingMiddleware())
```

---

### 3.4 `execution/progress.py`

Contains unified progress tracking with sync and async support:

- `ProgressTracker` — High-level progress tracking with Rich backend
- `ProgressRenderer` — Renders progress events to console
- `stream_progress()` — Async function to consume streaming results
- `progress_generator()` — Create progress-reporting async generators
- Global `get_progress_tracker()` and `configure_progress()` functions

---

### 3.5 `execution/executor.py`

Contains:

- `OperationCategory` — Enum for operation types (READ, WRITE, etc.)
- `OperationSpec[T]` — Dataclass defining an operation
- `UnifiedOperationExecutor` — Single executor handling sync/async/streaming
- Global `get_executor()` and `configure_executor()` functions
- `run_async_operation()` and `run_sync()` convenience functions

---

### 3.6 Plugins Package

#### `plugins/manifest.py`
Full implementation (moved from `plugin_manifest.py`):
- `PluginCapability` enum
- `SemanticVersion` dataclass
- `PluginDependency` dataclass
- `PluginManifest` dataclass with validation

#### `plugins/sandbox.py`
Full implementation (moved from `plugin_sandbox.py`):
- `ALLOWED_MODULES` and `CAPABILITY_MODULES` constants
- `SandboxConfig` dataclass
- `SandboxedImporter` class
- `PluginSandbox` context manager

#### `plugins/discovery.py`
- `DEFAULT_PLUGIN_PATHS` constant
- `DiscoveredPlugin` dataclass
- `discover_plugins()` function
- Legacy plugin detection helpers

#### `plugins/loader.py`
- `LoadedPlugin` dataclass
- `PluginLoadResult` dataclass
- `PluginLoader` class with sandbox support

#### `plugins/registry.py`
- `PluginProtocol` for legacy plugins
- `PluginInfo` dataclass
- `PluginManager` class
- `register_plugin_operations()` function
- `initialize_plugins()` function
- Global `get_plugin_manager()` function

#### `plugins/testing.py`
Full implementation (moved from `plugin_testing.py`):
- `PluginTestResult` dataclass
- `PluginTestHarness` class
- `create_plugin_scaffold()` function

---

## 4. Consumer Updates

### 4.1 `resilience.py`

```python
# BEFORE
from codeintel.cli.cli_middleware import OperationMiddleware

class ResilienceMiddleware(OperationMiddleware):
    def before_invoke(self, op_id: str, params: dict[str, Any]) -> dict[str, Any]:
        category = _get_category(op_id)
        ...

# AFTER
from codeintel.cli.execution import Middleware, ExecutionContext

class ResilienceMiddleware(Middleware):
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        category = _get_category(ctx.operation_id)
        if self._config.circuit_breaker_enabled and category:
            breaker = self._breakers.get_breaker(category)
            breaker.allow_request()
        
        return {
            "start_time": time.monotonic(),
            "category": category,
        }
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: object,
        mw_ctx: dict[str, Any],
    ) -> None:
        category = mw_ctx.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_success()
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_ctx: dict[str, Any],
    ) -> None:
        category = mw_ctx.get("category")
        if category and self._config.circuit_breaker_enabled:
            breaker = self._breakers.get_breaker(category)
            breaker.record_failure()
```

### 4.2 `observability.py`

```python
# BEFORE
from codeintel.cli.cli_middleware import OperationMiddleware

class ObservabilityMiddleware(OperationMiddleware):
    def before_invoke(self, op_id: str, params: dict[str, Any]) -> dict[str, Any]:
        ...

# AFTER
from codeintel.cli.execution import Middleware, ExecutionContext

class ObservabilityMiddleware(Middleware):
    def before_invoke(self, ctx: ExecutionContext) -> dict[str, Any]:
        context: dict[str, Any] = {
            "start_time": time.monotonic(),
            "operation_id": ctx.operation_id,
        }
        
        if self._config.tracing_enabled:
            span = self._start_span(ctx.operation_id, ctx.params)
            context["span"] = span
        
        if self._config.structured_logging:
            extra: dict[str, Any] = {
                "operation_id": ctx.operation_id,
                "trace_id": ctx.trace_id,
            }
            if self._config.log_params:
                extra["params"] = _sanitize_params(ctx.params)
            LOG.info("Operation started", extra=extra)
        
        return context
```

### 4.3 `cyclopts_ops.py`

```python
# BEFORE
from codeintel.cli.cli_middleware import get_middleware_stack

def _invoke_with_middleware(...):
    middleware = get_middleware_stack()
    with middleware.wrap(op_id, params):
        ...

# AFTER
from codeintel.cli.execution import get_middleware_stack, ExecutionContext

def _invoke_with_middleware(...):
    middleware = get_middleware_stack()
    ctx = ExecutionContext.for_operation(op_id, params)
    with middleware.wrap(ctx):
        ...
```

### 4.4 All `operations/*.py` Files

```python
# BEFORE
from codeintel.cli.executor import OperationCategory, OperationSpec

# AFTER
from codeintel.cli.execution import OperationCategory, OperationSpec
```

### 4.5 Other Consumers

| File | Change |
|------|--------|
| `operation_registry.py` | Import from `execution` |
| `introspection.py` | Import from `execution` |
| `pipelines.py` | Import `get_executor` from `execution` |
| `shell.py` | Import `get_executor` from `execution` |
| `job_runner.py` | Import `get_executor` from `execution` |
| `cyclopts_plugins.py` | Import from `plugins` |

---

## 5. Files to Delete

After migration completes, these files are deleted entirely:

| File | Replacement |
|------|-------------|
| `executor.py` | `execution/executor.py` |
| `async_executor.py` | `execution/executor.py` |
| `async_types.py` | `execution/types.py` |
| `cli_middleware.py` | `execution/middleware.py` |
| `async_middleware.py` | `execution/middleware.py` |
| `cli_progress.py` | `execution/progress.py` |
| `async_progress.py` | `execution/progress.py` |
| `plugins.py` | `plugins/` package |
| `plugin_manifest.py` | `plugins/manifest.py` |
| `plugin_sandbox.py` | `plugins/sandbox.py` |
| `plugin_testing.py` | `plugins/testing.py` |

---

## 6. Migration Implementation Plan

### Phase 1: Update `execution/middleware.py`
- Change `UnifiedMiddleware` to `Middleware`
- Update method signatures to use `ExecutionContext`
- Update `MiddlewareStack.wrap()` to take `ExecutionContext`
- Add `MiddlewareStack.wrap_async()`
- Ensure all existing middleware implementations work

### Phase 2: Update Core Consumers
1. **`resilience.py`** — Update `ResilienceMiddleware` signature
2. **`observability.py`** — Update `ObservabilityMiddleware` signature
3. **`cyclopts_ops.py`** — Update middleware invocation

### Phase 3: Update Operation Imports
- Update all `operations/*.py` files
- Update `operation_registry.py`
- Update other executor consumers

### Phase 4: Consolidate Plugins
- Move content (not re-export) from `plugin_*.py` to `plugins/`
- Update `cyclopts_plugins.py` imports
- Ensure `plugins/` package is self-contained

### Phase 5: Delete Legacy Files
- Delete all 11 files listed in Section 5
- Run full test suite
- Verify zero import errors

### Phase 6: Final Verification
- Run `uv run python -m tools.quality_report`
- Run `uv run pytest tests/cli/ -q`
- Verify all acceptance criteria

---

## 7. Acceptance Criteria

- [ ] All tests pass (289+ CLI tests)
- [ ] Zero pyright errors
- [ ] Zero pyrefly errors
- [ ] Zero ruff errors
- [ ] No files remain from "Files to Delete" list
- [ ] All imports use `codeintel.cli.execution` or `codeintel.cli.plugins`
- [ ] No deprecation warnings in code
- [ ] `Middleware` base class uses `ExecutionContext` parameter
- [ ] `MiddlewareStack.wrap()` takes `ExecutionContext`
- [ ] Both sync and async middleware invocation work correctly

