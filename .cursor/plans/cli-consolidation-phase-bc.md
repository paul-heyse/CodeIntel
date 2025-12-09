# CLI Consolidation: Phase B & C Implementation Plan

> **Purpose**: Consolidate parallel architectures in the CLI codebase to create a unified, maintainable architecture while preserving all existing functionality.

---

## Background: CLI Architecture & Functional Context

### What the CLI Does

The CodeIntel CLI (`codeintel`) is the primary interface for developers and automation systems to interact with the CodeIntel code intelligence platform. It provides:

1. **Build Operations** (`codeintel build`) - Index codebases, run analytics pipelines, generate documentation
2. **Query Operations** (`codeintel query`, `codeintel search`) - Search symbols, find references, explore call graphs
3. **Storage Operations** (`codeintel storage`) - Manage indexes, catalogs, and datasets
4. **IDE Integration** (`codeintel ide`) - Support for editor plugins and LSP servers
5. **Plugin Management** (`codeintel plugins`) - Extend CLI with custom operations

The CLI is built on [Cyclopts](https://cyclopts.readthedocs.io/) for command parsing and follows a modular architecture where each command maps to an **operation** that flows through an **execution pipeline**.

### Core Architectural Concepts

#### Operations and the Registry

Every CLI command is backed by an `OperationSpec` - a dataclass that defines:

```
OperationSpec
├── operation_id: str        # e.g., "build.run", "query.symbols"
├── handler: Callable        # The actual implementation
├── category: OperationCategory  # READ, WRITE, COMPUTE, BUILD, etc.
├── param_schema: ValidationSchema  # Input validation rules
├── retryable: bool          # Whether to retry on transient failures
├── retry_policy: RetryPolicy    # Exponential backoff configuration
└── description: str         # For help text generation
```

All operations are registered in a global `OperationRegistry`, enabling:
- Runtime discovery and introspection
- Consistent validation across all commands
- Unified middleware application
- Plugin-based extensibility

#### The Execution Pipeline

When a user runs a command, it flows through the execution pipeline:

```
User Input
    │
    ▼
┌──────────────────┐
│  Cyclopts Parse  │  ← Command-line parsing
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Config Loader   │  ← Merge env vars, config files, flags
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Validation     │  ← Schema-based parameter validation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Middleware     │  ← Logging, tracing, timing, resilience
│   (before)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Handler       │  ← The actual operation logic
│   Execution      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Middleware     │  ← Record metrics, close spans
│   (after)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Rendering      │  ← Format output (JSON, text, table)
└────────┬─────────┘
         │
         ▼
   User Output
```

#### Handler Types

The CLI supports three types of handlers:

1. **Sync Handlers** - Traditional blocking functions
   ```python
   def build_status(**params) -> CliResult[BuildStatus]:
       status = check_build_status()
       return CliResult.success(status)
   ```

2. **Async Handlers** - Non-blocking for I/O-bound operations
   ```python
   async def fetch_remote_index(**params) -> CliResult[Index]:
       async with httpx.AsyncClient() as client:
           data = await client.get(url)
       return CliResult.success(Index.from_bytes(data))
   ```

3. **Streaming Handlers** - Yield progress updates during long operations
   ```python
   async def build_full(**params) -> AsyncGenerator[StreamingResult, None]:
       for i, step in enumerate(steps):
           yield StreamingResult(progress=ProgressEvent(i, len(steps), step.name))
           await step.execute()
       yield StreamingResult(result=CliResult.success(summary))
   ```

#### The Result Protocol

All handlers return `CliResult[T]`, a structured result that carries:
- Success/failure status
- The typed payload (`T`)
- Error details as RFC 9457 Problem Details
- Metadata for rendering

This enables consistent error handling and output formatting across all commands.

### The Plugin System

Plugins allow third parties to extend the CLI with custom operations. A plugin:
1. Is discovered in standard paths (`~/.codeintel/plugins/`, etc.)
2. Declares capabilities and dependencies in a manifest
3. Registers `OperationSpec` instances with the global registry
4. Executes within a sandbox that restricts dangerous imports

**Example plugin structure:**
```
my-plugin/
├── plugin.json           # Manifest with name, version, capabilities
├── my_plugin/
│   ├── __init__.py
│   └── main.py          # Contains register(registry) function
└── tests/
```

### Why Parallel Architectures Emerged

The CLI evolved through several phases:

1. **Phase 1-4**: Built core sync execution with middleware, validation, and rendering
2. **Phase 5**: Added config loading, health checks, background jobs, and plugin basics
3. **Phase 6**: Added structured errors, resilience (retry/circuit breaker), observability
4. **Phase 7**: Added async handlers, streaming progress, shell completions, plugin hardening

Each phase added capabilities, but async support (Phase 7) was implemented as a parallel stack rather than extending the existing sync infrastructure. Similarly, plugin hardening created a new manifest-based system alongside the original `create_plugin()` API.

This resulted in:
- **Two executor implementations** that share 70% of their logic
- **Two middleware hierarchies** with near-identical interfaces
- **Two plugin loading paths** that can't share test infrastructure
- **Duplicated types** like `ProgressEvent` defined in multiple places

### Key Considerations for Consolidation

#### 1. Backward Compatibility is Critical

The CLI is used in CI/CD pipelines, IDE plugins, and developer workflows. Any breaking change would disrupt users. Therefore:

- All existing command signatures must continue to work
- All existing operation IDs must remain valid
- Deprecation warnings must guide migration (not silent breaks)
- Old import paths must work via re-export shims

#### 2. Handler Type Detection Must Be Automatic

Operations shouldn't need to declare `is_async=True`. The executor should detect handler type automatically using:
- `asyncio.iscoroutinefunction()` for async handlers
- `inspect.isasyncgenfunction()` for streaming handlers
- Fall back to sync execution otherwise

#### 3. Middleware Must Work for All Handler Types

A single `LoggingMiddleware` should work whether the handler is sync, async, or streaming. The middleware protocol should:
- Default async methods to call their sync counterparts
- Allow middleware to override only the methods they need
- Preserve the order of middleware execution

#### 4. Progress Must Bridge Sync and Async Worlds

Long-running sync operations need progress too. The unified progress tracker should:
- Accept sync callbacks for sync handlers
- Produce async generators for streaming handlers
- Share the same `ProgressEvent` type

#### 5. Plugin Security Cannot Regress

The sandbox prevents plugins from importing dangerous modules like `subprocess` or `os`. Consolidation must:
- Keep sandbox enforcement on the hot path
- Not accidentally expose new bypass vectors
- Maintain capability-based access control

#### 6. Performance Budgets Exist

The CLI has performance expectations:
- Startup time < 200ms for simple commands
- `codeintel version` < 100ms
- Read operations < 500ms p95 (excluding I/O)

Consolidation should not regress these. Avoid:
- Eager imports of heavy modules
- Unnecessary async overhead for sync operations
- Complex type introspection on hot paths

#### 7. Testing Infrastructure Must Exercise Real Paths

Per the Testing Charter in AGENTS.md:
- No monkeypatching - use real components with isolated instances
- Test through public entry points (CLI commands, not internal helpers)
- Use the same tech stack as production (DuckDB, FAISS, etc.)

Consolidation tests should invoke the unified executor the same way the CLI does.

### Module Dependency Map

Understanding how CLI modules depend on each other helps identify integration points:

```
┌─────────────────────────────────────────────────────────────────┐
│                        cyclopts_app.py                          │
│                    (CLI entry point)                            │
└─────────────┬───────────────────────────────────────────────────┘
              │ imports
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  cyclopts_*.py (command modules)                                │
│  - cyclopts_build.py, cyclopts_query.py, etc.                  │
└─────────────┬───────────────────────────────────────────────────┘
              │ use
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    operation_registry.py                        │
│            (global registry of all operations)                  │
└─────────────┬───────────────────────────────────────────────────┘
              │ contains
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  executor.py / async_executor.py  ◄── CONSOLIDATE (Phase B)    │
│  (operation execution pipeline)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Uses:                                                          │
│  ├── cli_middleware.py / async_middleware.py                   │
│  ├── cli_progress.py / async_progress.py                       │
│  ├── cli_validation.py (param validation)                      │
│  ├── cli_render.py (output formatting)                         │
│  ├── resilience.py (retry, circuit breaker)                    │
│  └── observability.py (tracing, metrics)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  plugins.py + plugin_*.py  ◄── CONSOLIDATE (Phase C)           │
│  (plugin discovery, loading, sandboxing)                        │
├─────────────────────────────────────────────────────────────────┤
│  Integrates with:                                               │
│  ├── operation_registry.py (registers plugin operations)       │
│  ├── executor.py (executes plugin handlers)                    │
│  └── cyclopts_plugins.py (CLI commands for plugin mgmt)        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Shared Infrastructure                        │
├─────────────────────────────────────────────────────────────────┤
│  results.py          - CliResult[T] type                       │
│  cli_errors.py       - ProblemDetail, error types              │
│  error_taxonomy.py   - ErrorCode, StructuredCliError           │
│  cli_types.py        - OutputFormat, common enums              │
│  config_loader.py    - Configuration merging                   │
│  resilience.py       - RetryPolicy, CircuitBreaker (Phase A ✓) │
└─────────────────────────────────────────────────────────────────┘
```

### Files Affected by Each Phase

#### Phase B (Execution Pipeline) Touches:

| File | Action | Notes |
|------|--------|-------|
| `executor.py` | Deprecation shim | Re-export from `execution/` |
| `async_executor.py` | Deprecation shim | Re-export from `execution/` |
| `cli_middleware.py` | Deprecation shim | Re-export from `execution/` |
| `async_middleware.py` | Deprecation shim | Keep async-specific helpers |
| `cli_progress.py` | Deprecation shim | Re-export from `execution/` |
| `async_progress.py` | Deprecation shim | Re-export from `execution/` |
| `async_types.py` | Deprecation shim | Re-export from `execution/` |
| `operation_registry.py` | Update imports | Use `execution.OperationSpec` |
| `operations/*.py` | Update imports | Use `execution.OperationSpec` |
| `cyclopts_app.py` | Update imports | Use unified executor |

#### Phase C (Plugin Architecture) Touches:

| File | Action | Notes |
|------|--------|-------|
| `plugins.py` | Deprecation shim | Re-export from `plugins/` |
| `plugin_manifest.py` | Move to package | Becomes `plugins/manifest.py` |
| `plugin_sandbox.py` | Move to package | Becomes `plugins/sandbox.py` |
| `plugin_testing.py` | Move to package | Becomes `plugins/testing.py` |
| `cyclopts_plugins.py` | Update imports | Use `plugins.initialize_plugins` |
| `cyclopts_app.py` | Update imports | Use new plugin init |

---

## Executive Summary

This plan addresses two major consolidation efforts identified during code review:

1. **Phase B: Unified Execution Pipeline** - Merge sync and async execution paths into a cohesive `execution/` package
2. **Phase C: Unified Plugin Architecture** - Consolidate the two incompatible plugin systems into a single manifest-driven loader

### Success Criteria

- Zero `pyright`, `pyrefly`, or `ruff` errors without suppressions
- All 289+ existing CLI tests pass
- Backward compatibility via deprecation shims
- Reduced code duplication by ~40%
- Single source of truth for each concern

---

## Phase B: Unified Execution Pipeline

### B.1 Problem Statement

Currently, there are parallel execution paths that duplicate significant logic:

| Component | Sync Version | Async Version |
|-----------|--------------|---------------|
| Executor | `executor.py` | `async_executor.py` |
| Middleware | `cli_middleware.py` | `async_middleware.py` |
| Progress | `cli_progress.py` | `async_progress.py` |
| Types | Scattered | `async_types.py` |

**Issues:**
- Retry/tracing/logging logic exists twice
- Progress types (`ProgressEvent`, `ProgressTracker`) duplicated
- Middleware protocols have parallel hierarchies
- Bug fixes must be applied to multiple files

### B.2 Target Architecture

```
src/codeintel/cli/execution/
├── __init__.py           # Public API re-exports
├── types.py              # Unified types (ProgressEvent, StreamingResult, etc.)
├── context.py            # ExecutionContext, AsyncExecutionContext
├── executor.py           # UnifiedOperationExecutor (handles sync + async)
├── middleware.py         # Unified middleware protocol + implementations
├── progress.py           # Unified progress tracking (sync + async)
└── rendering.py          # Progress/result rendering
```

### B.3 Implementation Steps

#### B.3.1 Create `execution/types.py`

**Source files to merge:**
- `async_types.py` → `ProgressState`, `ProgressEvent`, `StreamingResult`, handler type aliases
- `cli_progress.py` → `ProgressConfig` (partial)

**New unified types:**

```python
from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from codeintel.cli.results import CliResult

T = TypeVar("T")


class ProgressState(Enum):
    """Progress states for operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    """Progress update during operation execution.
    
    Works for both sync (via callbacks) and async (via generators).
    """
    current: int
    total: int
    message: str = ""
    state: ProgressState = ProgressState.RUNNING
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, object] = field(default_factory=dict)
    
    @property
    def percentage(self) -> float:
        """Get progress percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0
    
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "current": self.current,
            "total": self.total,
            "message": self.message,
            "state": self.state.value,
            "percentage": self.percentage,
            "timestamp": self.timestamp.isoformat(),
            **self.metadata,
        }


@dataclass
class StreamingResult[T]:
    """Result that can be either a progress event or final result."""
    progress: ProgressEvent | None = None
    result: CliResult[T] | None = None
    
    @property
    def is_progress(self) -> bool:
        return self.progress is not None
    
    @property
    def is_result(self) -> bool:
        return self.result is not None


# Handler type aliases
SyncHandler = Callable[..., CliResult[T]]
AsyncHandler = Callable[..., Awaitable[CliResult[T]]]
StreamingHandler = Callable[..., AsyncGenerator[StreamingResult[T], None]]
AnyHandler = SyncHandler[T] | AsyncHandler[T] | StreamingHandler[T]


@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""
    show_progress: bool = True
    show_spinner: bool = True
    update_interval: float = 0.1
    format_string: str = "{message} [{current}/{total}]"
```

**Acceptance criteria:**
- [ ] All existing handler type aliases work
- [ ] `ProgressEvent` serializes correctly
- [ ] ruff/pyright/pyrefly clean

---

#### B.3.2 Create `execution/context.py`

**Source files to merge:**
- `executor.py` → `ExecutionContext`
- `async_executor.py` → `AsyncExecutionContext`

**Unified approach:**

```python
@dataclass
class ExecutionContext:
    """Unified context for operation execution.
    
    Works for both sync and async operations.
    """
    operation_id: str
    params: dict[str, Any]
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Async-specific (None for sync operations)
    cancellation_event: asyncio.Event | None = None
    progress_callback: Callable[[ProgressEvent], None] | None = None
    
    @property
    def is_async(self) -> bool:
        """Check if this is an async execution context."""
        return self.cancellation_event is not None
    
    def check_cancelled(self) -> None:
        """Check if operation was cancelled.
        
        Raises
        ------
        asyncio.CancelledError
            If cancellation was requested.
        """
        if self.cancellation_event and self.cancellation_event.is_set():
            raise asyncio.CancelledError
    
    async def check_cancelled_async(self) -> None:
        """Async version of cancellation check."""
        self.check_cancelled()
        await asyncio.sleep(0)  # Yield to event loop


@dataclass
class ExecutionResult[T]:
    """Result of operation execution."""
    result: CliResult[T]
    duration: float
    was_cancelled: bool = False
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Acceptance criteria:**
- [ ] Context works for sync operations
- [ ] Context supports async cancellation
- [ ] Progress callbacks work in both modes
- [ ] ruff/pyright/pyrefly clean

---

#### B.3.3 Create `execution/middleware.py`

**Source files to merge:**
- `cli_middleware.py` → `OperationMiddleware`, `MiddlewareStack`, `LoggingMiddleware`, `TimingMiddleware`
- `async_middleware.py` → `AsyncOperationMiddleware`, `AsyncTracingMiddleware`, `AsyncResilienceMiddleware`, `AsyncProgressMiddleware`

**Unified approach:**

```python
from abc import ABC, abstractmethod
from typing import Any

from codeintel.cli.execution.context import ExecutionContext, ExecutionResult
from codeintel.cli.results import CliResult


class UnifiedMiddleware(ABC):
    """Base class for middleware supporting both sync and async.
    
    Middleware can implement sync-only, async-only, or both interfaces.
    The executor will call the appropriate methods based on handler type.
    """
    
    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> ExecutionContext:
        """Execute before operation (sync or async).
        
        Default implementation returns context unchanged.
        """
        return ctx
    
    async def before_invoke_async(
        self,
        ctx: ExecutionContext,
    ) -> ExecutionContext:
        """Async version of before_invoke.
        
        Default delegates to sync version.
        """
        return self.before_invoke(ctx)
    
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
    ) -> CliResult[Any]:
        """Execute after successful operation.
        
        Default implementation returns result unchanged.
        """
        return result
    
    async def after_invoke_async(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
    ) -> CliResult[Any]:
        """Async version of after_invoke."""
        return self.after_invoke(ctx, result)
    
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
    ) -> Exception | None:
        """Handle operation error.
        
        Returns
        -------
        Exception | None
            Exception to raise, or None to suppress.
        """
        return exc
    
    async def on_error_async(
        self,
        ctx: ExecutionContext,
        exc: Exception,
    ) -> Exception | None:
        """Async version of on_error."""
        return self.on_error(ctx, exc)


class UnifiedMiddlewareStack:
    """Stack of middleware for unified execution."""
    
    def __init__(self, middleware: list[UnifiedMiddleware] | None = None) -> None:
        self._middleware = middleware or []
    
    def add(self, mw: UnifiedMiddleware) -> None:
        """Add middleware to stack."""
        self._middleware.append(mw)
    
    def execute_before(self, ctx: ExecutionContext) -> ExecutionContext:
        """Execute all before_invoke hooks (sync)."""
        for mw in self._middleware:
            ctx = mw.before_invoke(ctx)
        return ctx
    
    async def execute_before_async(self, ctx: ExecutionContext) -> ExecutionContext:
        """Execute all before_invoke hooks (async)."""
        for mw in self._middleware:
            ctx = await mw.before_invoke_async(ctx)
        return ctx
    
    # ... similar for after_invoke and on_error
```

**Concrete middleware to implement:**
- `LoggingMiddleware` - Logs operation lifecycle
- `TimingMiddleware` - Tracks duration, sets timeout
- `TracingMiddleware` - OpenTelemetry spans (uses unified resilience)
- `ProgressMiddleware` - Progress tracking/rendering
- `ResilienceMiddleware` - Delegate to `resilience.ResilienceMiddleware`

**Acceptance criteria:**
- [ ] Middleware works for sync operations
- [ ] Middleware works for async operations
- [ ] All existing middleware functionality preserved
- [ ] ruff/pyright/pyrefly clean

---

#### B.3.4 Create `execution/progress.py`

**Source files to merge:**
- `cli_progress.py` → `ProgressTracker`, progress rendering
- `async_progress.py` → `AsyncProgressRenderer`, `stream_progress`

**Unified approach:**

```python
class UnifiedProgressTracker:
    """Progress tracker supporting sync callbacks and async generators.
    
    Usage (sync):
        tracker = UnifiedProgressTracker(total=100)
        tracker.update(50, "Half done")
        
    Usage (async):
        async for event in tracker.stream():
            render(event)
    """
    
    def __init__(
        self,
        total: int,
        config: ProgressConfig | None = None,
    ) -> None:
        self._total = total
        self._current = 0
        self._config = config or ProgressConfig()
        self._queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self._callbacks: list[Callable[[ProgressEvent], None]] = []
    
    def update(self, current: int, message: str = "") -> None:
        """Update progress (sync)."""
        self._current = current
        event = ProgressEvent(
            current=current,
            total=self._total,
            message=message,
        )
        for callback in self._callbacks:
            callback(event)
        # Also queue for async consumers
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if queue full
    
    async def stream(self) -> AsyncGenerator[ProgressEvent, None]:
        """Stream progress events (async)."""
        while True:
            event = await self._queue.get()
            yield event
            if event.state in (ProgressState.COMPLETED, ProgressState.FAILED):
                break
    
    def add_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add sync callback for progress updates."""
        self._callbacks.append(callback)


class ProgressRenderer:
    """Render progress to console (works sync and async)."""
    
    def render(self, event: ProgressEvent) -> None:
        """Render progress event to console."""
        # Spinner, progress bar, etc.
```

**Acceptance criteria:**
- [ ] Sync progress callbacks work
- [ ] Async progress streaming works
- [ ] Console rendering works in both modes
- [ ] ruff/pyright/pyrefly clean

---

#### B.3.5 Create `execution/executor.py`

**Source files to merge:**
- `executor.py` → `OperationExecutor`, `OperationSpec`
- `async_executor.py` → `AsyncOperationExecutor`, async execution logic

**Unified approach:**

```python
@dataclass(frozen=True)
class OperationSpec[T]:
    """Specification for an operation.
    
    Supports sync handlers, async handlers, and streaming handlers.
    """
    operation_id: str
    handler: AnyHandler[T]
    description: str = ""
    category: OperationCategory = OperationCategory.READ
    
    # Validation
    param_schema: ValidationSchema | None = None
    
    # Execution behavior
    retryable: bool = False
    retry_policy: RetryPolicy | None = None
    timeout: float | None = None
    
    # Auto-detected
    is_async: bool | None = None  # None = auto-detect
    is_streaming: bool | None = None  # None = auto-detect
    
    def __post_init__(self) -> None:
        """Auto-detect handler type if not specified."""
        if self.is_async is None:
            object.__setattr__(self, "is_async", is_async_handler(self.handler))
        if self.is_streaming is None:
            object.__setattr__(self, "is_streaming", is_streaming_handler(self.handler))


class UnifiedOperationExecutor:
    """Execute operations with middleware, resilience, and progress.
    
    Handles sync, async, and streaming handlers transparently.
    """
    
    def __init__(
        self,
        middleware: UnifiedMiddlewareStack | None = None,
        resilience_config: ResilienceConfig | None = None,
        progress_config: ProgressConfig | None = None,
    ) -> None:
        self._middleware = middleware or UnifiedMiddlewareStack()
        self._resilience = resilience_config
        self._progress_config = progress_config
    
    def execute[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
    ) -> ExecutionResult[T]:
        """Execute operation (sync entry point).
        
        If handler is async, runs it in an event loop.
        """
        if spec.is_async or spec.is_streaming:
            return asyncio.run(self.execute_async(spec, params))
        return self._execute_sync(spec, params)
    
    async def execute_async[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
    ) -> ExecutionResult[T]:
        """Execute operation (async entry point)."""
        ctx = ExecutionContext(
            operation_id=spec.operation_id,
            params=params,
            cancellation_event=asyncio.Event(),
        )
        
        # Run middleware before
        ctx = await self._middleware.execute_before_async(ctx)
        
        try:
            if spec.is_streaming:
                result = await self._execute_streaming(spec, ctx)
            elif spec.is_async:
                result = await self._execute_async(spec, ctx)
            else:
                result = await asyncio.to_thread(
                    self._execute_handler_sync, spec, ctx
                )
            
            # Run middleware after
            result = await self._middleware.execute_after_async(ctx, result)
            
            return ExecutionResult(
                result=result,
                duration=time.monotonic() - ctx.started_at.timestamp(),
            )
            
        except Exception as exc:
            exc = await self._middleware.execute_on_error_async(ctx, exc)
            if exc:
                raise
    
    async def stream[T](
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
    ) -> AsyncGenerator[StreamingResult[T], None]:
        """Stream execution with progress (async generator)."""
        # For streaming handlers
```

**Acceptance criteria:**
- [ ] Sync handlers execute correctly
- [ ] Async handlers execute correctly
- [ ] Streaming handlers execute correctly
- [ ] Middleware chain works for all handler types
- [ ] Resilience (retry/circuit breaker) integrates correctly
- [ ] Progress tracking works
- [ ] ruff/pyright/pyrefly clean

---

#### B.3.6 Create `execution/__init__.py`

```python
"""Unified execution pipeline for CLI operations.

This package provides a single execution infrastructure that supports
sync, async, and streaming handlers with consistent middleware,
resilience, and progress tracking.
"""

from codeintel.cli.execution.context import (
    ExecutionContext,
    ExecutionResult,
)
from codeintel.cli.execution.executor import (
    OperationSpec,
    UnifiedOperationExecutor,
)
from codeintel.cli.execution.middleware import (
    LoggingMiddleware,
    ProgressMiddleware,
    TimingMiddleware,
    TracingMiddleware,
    UnifiedMiddleware,
    UnifiedMiddlewareStack,
)
from codeintel.cli.execution.progress import (
    ProgressRenderer,
    UnifiedProgressTracker,
)
from codeintel.cli.execution.types import (
    AnyHandler,
    AsyncHandler,
    ProgressConfig,
    ProgressEvent,
    ProgressState,
    StreamingHandler,
    StreamingResult,
    SyncHandler,
)

__all__ = [
    # Types
    "AnyHandler",
    "AsyncHandler",
    "ProgressConfig",
    "ProgressEvent",
    "ProgressState",
    "StreamingHandler",
    "StreamingResult",
    "SyncHandler",
    # Context
    "ExecutionContext",
    "ExecutionResult",
    # Executor
    "OperationSpec",
    "UnifiedOperationExecutor",
    # Middleware
    "LoggingMiddleware",
    "ProgressMiddleware",
    "TimingMiddleware",
    "TracingMiddleware",
    "UnifiedMiddleware",
    "UnifiedMiddlewareStack",
    # Progress
    "ProgressRenderer",
    "UnifiedProgressTracker",
]
```

---

#### B.3.7 Update Existing Modules

1. **Update `executor.py`** to be a deprecation shim:
   ```python
   """Operation executor (deprecated).
   
   .. deprecated:: 1.0
       Use :mod:`codeintel.cli.execution` instead.
   """
   import warnings
   from codeintel.cli.execution import (
       OperationSpec,
       UnifiedOperationExecutor as OperationExecutor,
       # ... other re-exports
   )
   warnings.warn(...)
   ```

2. **Update `async_executor.py`** to be a deprecation shim

3. **Update `cli_middleware.py`** to be a deprecation shim

4. **Update `async_middleware.py`** to be a deprecation shim

5. **Update `cli_progress.py`** to be a deprecation shim

6. **Update `async_progress.py`** to be a deprecation shim

7. **Update `async_types.py`** to be a deprecation shim

---

#### B.3.8 Update Imports Across Codebase

Files to update:
- `operation_registry.py`
- `cyclopts_app.py`
- All operation modules under `operations/`
- Test files under `tests/cli/`

---

### B.4 Testing Strategy

1. **Unit tests** for each new module
2. **Integration tests** for executor with different handler types
3. **Regression tests** - all existing 289+ tests must pass
4. **Performance tests** - verify no significant regression

---

## Phase C: Unified Plugin Architecture

### C.1 Problem Statement

Two incompatible plugin systems exist:

| Aspect | Legacy (`plugins.py`) | Modern (`plugin_manifest.py` + friends) |
|--------|----------------------|----------------------------------------|
| Discovery | Bare `.py` files | `plugin.json` manifest |
| Hook | `create_plugin()` → `PluginProtocol` | `register(registry)` |
| Operations | `plugin.get_operations()` list | Direct `registry.register(OperationSpec)` |
| Sandboxing | None | `PluginSandbox` with `SandboxedImporter` |
| Capabilities | None | `PluginCapability` enum |
| Testing | None | `PluginTestHarness` |
| Versioning | None | `SemanticVersion` |

**Issues:**
- No manifest validation on live path
- No sandboxing in production
- Two registration flows
- Two plugin directory shapes
- `PluginTestHarness` can't test legacy plugins

### C.2 Target Architecture

```
src/codeintel/cli/plugins/
├── __init__.py           # Public API
├── discovery.py          # Plugin discovery (manifest-based)
├── loader.py             # Unified plugin loading
├── manifest.py           # PluginManifest, SemanticVersion, PluginCapability
├── sandbox.py            # PluginSandbox, SandboxedImporter
├── registry.py           # Plugin registry integration
└── testing.py            # PluginTestHarness, scaffolding
```

### C.3 Implementation Steps

#### C.3.1 Create `plugins/manifest.py`

**Source:** `plugin_manifest.py` (move with minimal changes)

```python
"""Plugin manifest schema and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

CLI_API_VERSION = "1.0.0"


class PluginCapability(Enum):
    """Capabilities a plugin can request."""
    REGISTER_OPERATIONS = "register_operations"
    ACCESS_CONFIG = "access_config"
    ACCESS_STORAGE = "access_storage"
    ACCESS_NETWORK = "access_network"
    EXECUTE_COMMANDS = "execute_commands"


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version with comparison support."""
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""
    
    @classmethod
    def parse(cls, version: str) -> SemanticVersion:
        """Parse version string."""
        match = SEMVER_PATTERN.match(version)
        if not match:
            msg = f"Invalid semantic version: {version}"
            raise ValueError(msg)
        return cls(
            major=int(match["major"]),
            minor=int(match["minor"]),
            patch=int(match["patch"]),
            prerelease=match["prerelease"] or "",
            build=match["build"] or "",
        )
    
    def is_compatible_with(self, required: SemanticVersion) -> bool:
        """Check if this version satisfies requirement (major must match)."""
        return self.major == required.major and (
            self.minor > required.minor or
            (self.minor == required.minor and self.patch >= required.patch)
        )


@dataclass
class PluginManifest:
    """Plugin manifest from plugin.json."""
    name: str
    version: str
    api_version: str
    description: str = ""
    author: str = ""
    capabilities: list[PluginCapability] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    entry_point: str = "main"
    
    @classmethod
    def load(cls, path: Path) -> PluginManifest:
        """Load manifest from plugin.json file."""
        with path.open() as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> PluginManifest:
        """Create from dictionary."""
        capabilities = [
            PluginCapability(c) for c in data.get("capabilities", [])
        ]
        return cls(
            name=data["name"],
            version=data["version"],
            api_version=data["api_version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            capabilities=capabilities,
            dependencies=data.get("dependencies", []),
            entry_point=data.get("entry_point", "main"),
        )
    
    def validate(self) -> list[str]:
        """Validate manifest, return list of errors."""
        errors = []
        
        # Validate version formats
        try:
            SemanticVersion.parse(self.version)
        except ValueError as e:
            errors.append(f"Invalid version: {e}")
        
        try:
            api_ver = SemanticVersion.parse(self.api_version)
            cli_ver = SemanticVersion.parse(CLI_API_VERSION)
            if not cli_ver.is_compatible_with(api_ver):
                errors.append(
                    f"API version {self.api_version} not compatible "
                    f"with CLI version {CLI_API_VERSION}"
                )
        except ValueError as e:
            errors.append(f"Invalid api_version: {e}")
        
        return errors
```

**Acceptance criteria:**
- [ ] Manifest loads from JSON
- [ ] Version parsing works
- [ ] Validation catches errors
- [ ] ruff/pyright/pyrefly clean

---

#### C.3.2 Create `plugins/sandbox.py`

**Source:** `plugin_sandbox.py` (move with minimal changes)

```python
"""Plugin sandboxing for restricted execution."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from types import ModuleType

from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest

# Modules always allowed
ALLOWED_MODULES = frozenset({
    "builtins", "collections", "dataclasses", "datetime", "enum",
    "functools", "itertools", "json", "logging", "pathlib",
    "re", "typing", "typing_extensions",
})

# Modules requiring specific capabilities
CAPABILITY_MODULES: dict[PluginCapability, frozenset[str]] = {
    PluginCapability.ACCESS_STORAGE: frozenset({"duckdb", "sqlite3"}),
    PluginCapability.ACCESS_NETWORK: frozenset({"httpx", "requests", "urllib"}),
    PluginCapability.EXECUTE_COMMANDS: frozenset({"subprocess", "os"}),
}


@dataclass
class SandboxConfig:
    """Configuration for plugin sandbox."""
    allowed_capabilities: set[PluginCapability] = field(default_factory=set)
    timeout: float = 30.0
    memory_limit: int = 100 * 1024 * 1024  # 100MB


class SandboxedImporter:
    """Custom import hook that restricts module access."""
    
    def __init__(
        self,
        manifest: PluginManifest,
        config: SandboxConfig,
    ) -> None:
        self._manifest = manifest
        self._config = config
        self._allowed = self._compute_allowed_modules()
    
    def _compute_allowed_modules(self) -> frozenset[str]:
        """Compute set of allowed modules based on capabilities."""
        allowed = set(ALLOWED_MODULES)
        
        # Add modules for each capability
        for cap in self._config.allowed_capabilities:
            if cap in CAPABILITY_MODULES:
                allowed.update(CAPABILITY_MODULES[cap])
        
        # Always allow codeintel.cli.results for return types
        allowed.add("codeintel")
        
        return frozenset(allowed)
    
    def find_module(self, name: str, path: object = None) -> SandboxedImporter | None:
        """Check if module import is allowed."""
        # Allow plugin's own modules
        entry_parts = self._manifest.entry_point.split(".", maxsplit=1)
        if entry_parts and name.startswith(entry_parts[0]):
            return None
        
        # Check against allowed list
        root_module = name.split(".", maxsplit=1)[0]
        if root_module in self._allowed or name in self._allowed:
            return None
        
        # Block by returning self (load_module will raise)
        return self
    
    def load_module(self, name: str) -> ModuleType:
        """Raise ImportError for blocked modules."""
        msg = (
            f"Plugin '{self._manifest.name}' attempted to import "
            f"restricted module '{name}'"
        )
        raise ImportError(msg)


class PluginSandbox:
    """Sandbox for safe plugin execution."""
    
    def __init__(
        self,
        manifest: PluginManifest,
        config: SandboxConfig | None = None,
    ) -> None:
        self._manifest = manifest
        self._config = config or SandboxConfig(
            allowed_capabilities=set(manifest.capabilities)
        )
        self._importer = SandboxedImporter(manifest, self._config)
        self._active = False
    
    def __enter__(self) -> PluginSandbox:
        """Enter sandbox context."""
        if self._active:
            msg = "Sandbox already active"
            raise RuntimeError(msg)
        sys.meta_path.insert(0, self._importer)  # type: ignore[arg-type]
        self._active = True
        return self
    
    def __exit__(self, *args: object) -> None:
        """Exit sandbox context."""
        import contextlib
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self._importer)  # type: ignore[arg-type]
        self._active = False
    
    def load_plugin(self) -> ModuleType:
        """Load plugin module within sandbox."""
        if not self._active:
            msg = "Sandbox not active"
            raise RuntimeError(msg)
        return importlib.import_module(self._manifest.entry_point)
```

**Acceptance criteria:**
- [ ] Sandbox blocks restricted imports
- [ ] Capabilities unlock modules
- [ ] Context manager works correctly
- [ ] ruff/pyright/pyrefly clean

---

#### C.3.3 Create `plugins/discovery.py`

**New file for plugin discovery:**

```python
"""Plugin discovery and enumeration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.plugins.manifest import PluginManifest

LOG = logging.getLogger(__name__)

DEFAULT_PLUGIN_PATHS = [
    Path.home() / ".codeintel" / "plugins",
    Path("/etc/codeintel/plugins"),
]


@dataclass
class DiscoveredPlugin:
    """Information about a discovered plugin."""
    path: Path
    manifest: PluginManifest
    valid: bool
    errors: list[str]


def discover_plugins(
    search_paths: list[Path] | None = None,
) -> list[DiscoveredPlugin]:
    """Discover plugins in search paths.
    
    Parameters
    ----------
    search_paths
        Paths to search for plugins. Uses defaults if None.
    
    Returns
    -------
    list[DiscoveredPlugin]
        List of discovered plugins.
    """
    paths = search_paths or DEFAULT_PLUGIN_PATHS
    discovered: list[DiscoveredPlugin] = []
    
    for search_path in paths:
        if not search_path.exists():
            continue
        
        for item in search_path.iterdir():
            if not item.is_dir():
                continue
            
            manifest_path = item / "plugin.json"
            if not manifest_path.exists():
                # Check for legacy plugin (bare .py file)
                legacy = _check_legacy_plugin(item)
                if legacy:
                    discovered.append(legacy)
                continue
            
            try:
                manifest = PluginManifest.load(manifest_path)
                errors = manifest.validate()
                discovered.append(DiscoveredPlugin(
                    path=item,
                    manifest=manifest,
                    valid=len(errors) == 0,
                    errors=errors,
                ))
            except Exception as e:
                LOG.warning("Failed to load manifest %s: %s", manifest_path, e)
                discovered.append(DiscoveredPlugin(
                    path=item,
                    manifest=PluginManifest(
                        name=item.name,
                        version="0.0.0",
                        api_version="0.0.0",
                    ),
                    valid=False,
                    errors=[str(e)],
                ))
    
    return discovered


def _check_legacy_plugin(path: Path) -> DiscoveredPlugin | None:
    """Check for legacy plugin format (bare .py file with create_plugin).
    
    Returns a DiscoveredPlugin with warnings about migration.
    """
    # Look for .py files with create_plugin function
    for py_file in path.glob("*.py"):
        content = py_file.read_text()
        if "def create_plugin" in content:
            return DiscoveredPlugin(
                path=path,
                manifest=PluginManifest(
                    name=path.name,
                    version="0.0.0",
                    api_version="0.0.0",
                    description="Legacy plugin (needs migration)",
                    entry_point=py_file.stem,
                ),
                valid=False,
                errors=[
                    "Legacy plugin format detected. "
                    "Please add plugin.json and migrate to register() API. "
                    "See: https://docs.codeintel.dev/plugins/migration"
                ],
            )
    return None
```

**Acceptance criteria:**
- [ ] Discovers manifest-based plugins
- [ ] Detects legacy plugins with migration warning
- [ ] Validates manifests
- [ ] ruff/pyright/pyrefly clean

---

#### C.3.4 Create `plugins/loader.py`

**Unified plugin loading:**

```python
"""Unified plugin loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from codeintel.cli.plugins.discovery import DiscoveredPlugin, discover_plugins
from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest
from codeintel.cli.plugins.sandbox import PluginSandbox, SandboxConfig

LOG = logging.getLogger(__name__)


@dataclass
class LoadedPlugin:
    """A loaded plugin ready for registration."""
    manifest: PluginManifest
    module: ModuleType
    path: Path
    
    def has_capability(self, cap: PluginCapability) -> bool:
        """Check if plugin has capability."""
        return cap in self.manifest.capabilities


@dataclass
class PluginLoadResult:
    """Result of loading plugins."""
    loaded: list[LoadedPlugin] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    legacy_warnings: list[str] = field(default_factory=list)


class PluginLoader:
    """Load and initialize plugins.
    
    Parameters
    ----------
    sandbox_enabled
        Enable sandboxing for plugins.
    allowed_capabilities
        Capabilities to allow if not in manifest.
    """
    
    def __init__(
        self,
        *,
        sandbox_enabled: bool = True,
        allowed_capabilities: set[PluginCapability] | None = None,
    ) -> None:
        self._sandbox_enabled = sandbox_enabled
        self._allowed_caps = allowed_capabilities or set()
    
    def load_all(
        self,
        search_paths: list[Path] | None = None,
    ) -> PluginLoadResult:
        """Discover and load all plugins.
        
        Parameters
        ----------
        search_paths
            Paths to search for plugins.
        
        Returns
        -------
        PluginLoadResult
            Results of loading.
        """
        result = PluginLoadResult()
        
        discovered = discover_plugins(search_paths)
        
        for plugin in discovered:
            if not plugin.valid:
                if "Legacy plugin" in str(plugin.errors):
                    result.legacy_warnings.append(
                        f"Legacy plugin at {plugin.path}: {plugin.errors[0]}"
                    )
                    # Try loading legacy plugin
                    try:
                        loaded = self._load_legacy(plugin)
                        if loaded:
                            result.loaded.append(loaded)
                            continue
                    except Exception as e:
                        result.failed.append((plugin.path, str(e)))
                        continue
                else:
                    result.failed.append((plugin.path, "; ".join(plugin.errors)))
                continue
            
            try:
                loaded = self._load_plugin(plugin)
                result.loaded.append(loaded)
            except Exception as e:
                LOG.exception("Failed to load plugin %s", plugin.path)
                result.failed.append((plugin.path, str(e)))
        
        return result
    
    def _load_plugin(self, discovered: DiscoveredPlugin) -> LoadedPlugin:
        """Load a manifest-based plugin."""
        manifest = discovered.manifest
        
        if self._sandbox_enabled:
            config = SandboxConfig(
                allowed_capabilities=set(manifest.capabilities) | self._allowed_caps
            )
            with PluginSandbox(manifest, config) as sandbox:
                module = sandbox.load_plugin()
        else:
            import importlib
            module = importlib.import_module(manifest.entry_point)
        
        return LoadedPlugin(
            manifest=manifest,
            module=module,
            path=discovered.path,
        )
    
    def _load_legacy(self, discovered: DiscoveredPlugin) -> LoadedPlugin | None:
        """Load a legacy plugin (create_plugin API)."""
        import importlib
        import sys
        
        # Add plugin path to sys.path temporarily
        sys.path.insert(0, str(discovered.path))
        try:
            module = importlib.import_module(discovered.manifest.entry_point)
            
            # Verify it has create_plugin
            if not hasattr(module, "create_plugin"):
                return None
            
            return LoadedPlugin(
                manifest=discovered.manifest,
                module=module,
                path=discovered.path,
            )
        finally:
            sys.path.remove(str(discovered.path))
```

**Acceptance criteria:**
- [ ] Loads manifest-based plugins with sandbox
- [ ] Loads legacy plugins with warning
- [ ] Reports failures clearly
- [ ] ruff/pyright/pyrefly clean

---

#### C.3.5 Create `plugins/registry.py`

**Integration with operation registry:**

```python
"""Plugin registration with operation registry."""

from __future__ import annotations

import logging
from typing import Any

from codeintel.cli.operation_registry import OperationRegistry, get_operation_registry
from codeintel.cli.plugins.loader import LoadedPlugin, PluginLoadResult, PluginLoader
from codeintel.cli.plugins.manifest import PluginCapability

LOG = logging.getLogger(__name__)


def register_plugin_operations(
    plugin: LoadedPlugin,
    registry: OperationRegistry | None = None,
) -> int:
    """Register operations from a plugin.
    
    Parameters
    ----------
    plugin
        Loaded plugin.
    registry
        Operation registry (uses global if None).
    
    Returns
    -------
    int
        Number of operations registered.
    """
    reg = registry or get_operation_registry()
    
    if not plugin.has_capability(PluginCapability.REGISTER_OPERATIONS):
        LOG.warning(
            "Plugin %s does not have REGISTER_OPERATIONS capability",
            plugin.manifest.name,
        )
        return 0
    
    # Check for register() function (new API)
    if hasattr(plugin.module, "register"):
        return _register_new_api(plugin, reg)
    
    # Check for create_plugin() function (legacy API)
    if hasattr(plugin.module, "create_plugin"):
        return _register_legacy_api(plugin, reg)
    
    LOG.warning(
        "Plugin %s has no register() or create_plugin() function",
        plugin.manifest.name,
    )
    return 0


def _register_new_api(plugin: LoadedPlugin, registry: OperationRegistry) -> int:
    """Register using new register(registry) API."""
    count = 0
    
    class RegistrationProxy:
        """Proxy that counts registrations."""
        
        def register(self, spec: Any) -> None:
            nonlocal count
            # Ensure operation ID is prefixed with plugin name
            if hasattr(spec, "operation_id"):
                if not spec.operation_id.startswith(f"{plugin.manifest.name}."):
                    LOG.warning(
                        "Operation %s should be prefixed with plugin name %s",
                        spec.operation_id,
                        plugin.manifest.name,
                    )
            registry.register(spec)
            count += 1
    
    plugin.module.register(RegistrationProxy())
    return count


def _register_legacy_api(plugin: LoadedPlugin, registry: OperationRegistry) -> int:
    """Register using legacy create_plugin() API."""
    count = 0
    
    instance = plugin.module.create_plugin()
    
    # Legacy API: plugin.get_operations() returns list of specs
    if hasattr(instance, "get_operations"):
        for spec in instance.get_operations():
            registry.register(spec)
            count += 1
    
    return count


def initialize_plugins(
    loader: PluginLoader | None = None,
    registry: OperationRegistry | None = None,
) -> PluginLoadResult:
    """Initialize all plugins and register their operations.
    
    Parameters
    ----------
    loader
        Plugin loader (creates default if None).
    registry
        Operation registry (uses global if None).
    
    Returns
    -------
    PluginLoadResult
        Results of plugin loading.
    """
    ldr = loader or PluginLoader()
    reg = registry or get_operation_registry()
    
    result = ldr.load_all()
    
    for plugin in result.loaded:
        try:
            count = register_plugin_operations(plugin, reg)
            LOG.info(
                "Registered %d operations from plugin %s",
                count,
                plugin.manifest.name,
            )
        except Exception as e:
            LOG.exception("Failed to register operations from %s", plugin.manifest.name)
            result.failed.append((plugin.path, str(e)))
    
    # Log legacy warnings
    for warning in result.legacy_warnings:
        LOG.warning(warning)
    
    return result
```

**Acceptance criteria:**
- [ ] Registers new API plugins
- [ ] Registers legacy API plugins
- [ ] Enforces capability checks
- [ ] ruff/pyright/pyrefly clean

---

#### C.3.6 Create `plugins/testing.py`

**Source:** `plugin_testing.py` (move with minimal changes)

```python
"""Plugin testing utilities."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest
from codeintel.cli.plugins.sandbox import PluginSandbox, SandboxConfig


@dataclass
class PluginTestResult:
    """Result of a plugin test."""
    success: bool
    message: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PluginTestHarness:
    """Test harness for plugins."""
    
    def __init__(
        self,
        manifest: PluginManifest,
        *,
        capabilities: list[PluginCapability] | None = None,
    ) -> None:
        self._manifest = manifest
        self._capabilities = set(capabilities or manifest.capabilities)
        self._registered_operations: list[str] = []
    
    def validate_manifest(self) -> PluginTestResult:
        """Validate plugin manifest."""
        errors = self._manifest.validate()
        return PluginTestResult(
            success=len(errors) == 0,
            message="Manifest validation" + (" passed" if not errors else " failed"),
            errors=errors,
        )
    
    def test_load(self) -> PluginTestResult:
        """Test plugin loading in sandbox."""
        config = SandboxConfig(allowed_capabilities=self._capabilities)
        
        try:
            with PluginSandbox(self._manifest, config) as sandbox:
                module = sandbox.load_plugin()
                
                if not hasattr(module, "register"):
                    return PluginTestResult(
                        success=True,
                        message="Plugin loaded (no register function)",
                        warnings=["Plugin has no 'register' function"],
                    )
                
                return PluginTestResult(
                    success=True,
                    message="Plugin loaded successfully",
                )
        except ImportError as e:
            return PluginTestResult(
                success=False,
                message="Plugin failed to load",
                errors=[f"ImportError: {e}"],
            )
        except Exception as e:
            return PluginTestResult(
                success=False,
                message="Plugin failed to load",
                errors=[f"{type(e).__name__}: {e}"],
            )
    
    def test_operations(self) -> PluginTestResult:
        """Test plugin operation registration."""
        # Implementation similar to existing plugin_testing.py
        ...
    
    def run_all_tests(self) -> list[PluginTestResult]:
        """Run all tests."""
        return [
            self.validate_manifest(),
            self.test_load(),
            self.test_operations(),
        ]
    
    def get_summary(self) -> dict[str, object]:
        """Get test summary."""
        results = self.run_all_tests()
        passed = sum(1 for r in results if r.success)
        return {
            "tests_run": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "registered_operations": self._registered_operations,
        }


def create_plugin_scaffold(
    name: str,
    output_dir: Path,
    *,
    capabilities: list[PluginCapability] | None = None,
) -> Path:
    """Create a new plugin scaffold.
    
    Parameters
    ----------
    name
        Plugin name.
    output_dir
        Directory to create plugin in.
    capabilities
        Capabilities to request.
    
    Returns
    -------
    Path
        Path to created plugin directory.
    """
    caps = capabilities or [PluginCapability.REGISTER_OPERATIONS]
    
    plugin_dir = output_dir / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plugin.json
    manifest = {
        "name": name,
        "version": "0.1.0",
        "api_version": "1.0.0",
        "description": f"{name} plugin",
        "author": "",
        "capabilities": [c.value for c in caps],
        "entry_point": name,
    }
    (plugin_dir / "plugin.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    
    # Create main.py
    main_content = f'''"""Plugin: {name}."""

from codeintel.cli.execution import OperationSpec
from codeintel.cli.results import CliResult


def register(registry):
    """Register plugin operations."""
    registry.register(
        OperationSpec(
            operation_id="{name}.hello",
            handler=_hello_handler,
            description="Say hello",
        )
    )


def _hello_handler(**params) -> CliResult[str]:
    """Example handler."""
    return CliResult.success("Hello from {name}!")
'''
    (plugin_dir / f"{name}.py").write_text(main_content, encoding="utf-8")
    
    # Create __init__.py
    (plugin_dir / "__init__.py").write_text(
        f'"""Plugin: {name}."""\n',
        encoding="utf-8",
    )
    
    # Create tests directory
    tests_dir = plugin_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    
    test_content = f'''"""Tests for {name} plugin."""

from pathlib import Path

from codeintel.cli.plugins.manifest import PluginManifest
from codeintel.cli.plugins.testing import PluginTestHarness


def test_plugin_loads():
    """Test that plugin loads successfully."""
    manifest_path = Path(__file__).parent.parent / "plugin.json"
    manifest = PluginManifest.load(manifest_path)
    harness = PluginTestHarness(manifest)
    
    result = harness.test_load()
    assert result.success, result.errors
'''
    (tests_dir / f"test_{name}.py").write_text(test_content, encoding="utf-8")
    
    return plugin_dir
```

**Acceptance criteria:**
- [ ] Harness tests manifest validation
- [ ] Harness tests sandbox loading
- [ ] Harness tests operation registration
- [ ] Scaffold creates working plugin
- [ ] ruff/pyright/pyrefly clean

---

#### C.3.7 Create `plugins/__init__.py`

```python
"""Unified plugin architecture for CLI.

This package provides:
- Manifest-based plugin discovery
- Sandboxed plugin loading
- Operation registration
- Plugin testing utilities
"""

from codeintel.cli.plugins.discovery import (
    DEFAULT_PLUGIN_PATHS,
    DiscoveredPlugin,
    discover_plugins,
)
from codeintel.cli.plugins.loader import (
    LoadedPlugin,
    PluginLoadResult,
    PluginLoader,
)
from codeintel.cli.plugins.manifest import (
    CLI_API_VERSION,
    PluginCapability,
    PluginManifest,
    SemanticVersion,
)
from codeintel.cli.plugins.registry import (
    initialize_plugins,
    register_plugin_operations,
)
from codeintel.cli.plugins.sandbox import (
    PluginSandbox,
    SandboxConfig,
    SandboxedImporter,
)
from codeintel.cli.plugins.testing import (
    PluginTestHarness,
    PluginTestResult,
    create_plugin_scaffold,
)

__all__ = [
    # Discovery
    "DEFAULT_PLUGIN_PATHS",
    "DiscoveredPlugin",
    "discover_plugins",
    # Loader
    "LoadedPlugin",
    "PluginLoadResult",
    "PluginLoader",
    # Manifest
    "CLI_API_VERSION",
    "PluginCapability",
    "PluginManifest",
    "SemanticVersion",
    # Registry
    "initialize_plugins",
    "register_plugin_operations",
    # Sandbox
    "PluginSandbox",
    "SandboxConfig",
    "SandboxedImporter",
    # Testing
    "PluginTestHarness",
    "PluginTestResult",
    "create_plugin_scaffold",
]
```

---

#### C.3.8 Update Existing Modules

1. **Update `plugins.py`** to be a deprecation shim
2. **Update `plugin_manifest.py`** to be a deprecation shim
3. **Update `plugin_sandbox.py`** to be a deprecation shim
4. **Update `plugin_testing.py`** to be a deprecation shim
5. **Update `cyclopts_plugins.py`** to use new package
6. **Update `cyclopts_app.py`** to use new `initialize_plugins`

---

### C.4 Testing Strategy

1. **Unit tests** for each new module
2. **Integration tests** for plugin loading
3. **Manifest-based plugin tests**
4. **Legacy plugin compatibility tests**
5. **Sandbox security tests**
6. **Regression tests** - all existing tests must pass

---

## Implementation Order

### Week 1: Phase B
1. B.3.1: Create `execution/types.py` (Day 1)
2. B.3.2: Create `execution/context.py` (Day 1)
3. B.3.3: Create `execution/middleware.py` (Day 2)
4. B.3.4: Create `execution/progress.py` (Day 2)
5. B.3.5: Create `execution/executor.py` (Day 3)
6. B.3.6: Create `execution/__init__.py` (Day 3)
7. B.3.7-8: Deprecation shims + import updates (Day 4-5)

### Week 2: Phase C
1. C.3.1: Create `plugins/manifest.py` (Day 1)
2. C.3.2: Create `plugins/sandbox.py` (Day 1)
3. C.3.3: Create `plugins/discovery.py` (Day 2)
4. C.3.4: Create `plugins/loader.py` (Day 2)
5. C.3.5: Create `plugins/registry.py` (Day 3)
6. C.3.6: Create `plugins/testing.py` (Day 3)
7. C.3.7-8: `__init__.py` + deprecation shims (Day 4)
8. Testing and cleanup (Day 5)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing tests | Run full test suite after each step |
| Import cycles | Careful dependency ordering, TYPE_CHECKING guards |
| Performance regression | Benchmark critical paths |
| Legacy plugin breakage | Test with real legacy plugins |
| Sandbox bypass | Security review of import hooks |

---

## Definition of Done

- [ ] All code passes ruff, pyright, pyrefly with zero errors
- [ ] No `# type: ignore` or `# noqa` suppressions added
- [ ] All 289+ existing CLI tests pass
- [ ] New modules have comprehensive docstrings
- [ ] Deprecation warnings emit correctly
- [ ] Legacy plugins still work (with warnings)
- [ ] Code coverage maintained or improved

