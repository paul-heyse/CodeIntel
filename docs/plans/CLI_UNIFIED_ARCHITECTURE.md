# CLI Unified Architecture Specification

> **Status:** Draft  
> **Author:** Architecture Team  
> **Date:** December 2024  
> **Scope:** `src/codeintel/cli/` consolidation

## Executive Summary

This document specifies the target architecture for the CodeIntel CLI subsystem consolidation. The goal is to eliminate architectural duplication across context types, rendering stacks, command wiring, parameter extraction, operation registries, and bootstrap sequences—while achieving a best-in-class design that improves functionality, extensibility, hardness, and maintainability.

The architecture follows a **Handler-Centric Unification** approach where all CLI operations flow through a single execution pipeline with consistent context, rendering, and lifecycle management.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Core Components](#2-core-components)
3. [Unified HandlerContext](#3-unified-handlercontext)
4. [Command Layer](#4-command-layer)
5. [Handler Layer](#5-handler-layer)
6. [Rendering Layer](#6-rendering-layer)
7. [Execution Pipeline](#7-execution-pipeline)
8. [Operation Registry](#8-operation-registry)
9. [Bootstrap and Lifecycle](#9-bootstrap-and-lifecycle)
10. [Type Contracts](#10-type-contracts)
11. [Module Layout](#11-module-layout)
12. [Migration Strategy](#12-migration-strategy)
13. [Decision Log](#13-decision-log)

---

## 1. Architectural Overview

### 1.1 Design Principles

| Principle | Application |
|-----------|-------------|
| **Single Responsibility** | Each module handles one concern (context, rendering, execution) |
| **Dependency Injection** | Resources (gateway, runtime) injected via context, not globals |
| **Lazy Resolution** | Expensive resources resolved on first access, not at construction |
| **Declarative Binding** | Commands declare metadata; framework handles wiring |
| **Consistent Contracts** | All handlers follow same signature and return type |
| **Single Source of Truth** | One implementation per concern, no parallel stacks |

### 1.2 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI Entry Point                               │
│                    commands/app.py (Cyclopts)                        │
├─────────────────────────────────────────────────────────────────────┤
│                      Command Definitions                             │
│            @cli_command decorator + dataclass fields                 │
│                    (commands/*.py)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                    Command Executor                                  │
│        Bootstrap → Context Creation → Handler Dispatch → Render      │
│                  (execution/executor.py)                             │
├─────────────────────────────────────────────────────────────────────┤
│                   HandlerContext (Single Type)                       │
│     config | runtime (lazy) | gateway (lazy) | params | logger       │
│                  (handlers/context.py)                               │
├─────────────────────────────────────────────────────────────────────┤
│                    Handler Functions                                 │
│          def handler(ctx: HandlerContext) -> CliResult[T]            │
│                   (handlers/*.py)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                   UnifiedRenderer                                    │
│        Text/JSON/JSONL output, tables, errors, progress              │
│                 (rendering/service.py)                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow

```
User Input (CLI args)
       │
       ▼
┌──────────────┐
│   Cyclopts   │ ─── Parses args into dataclass fields
└──────────────┘
       │
       ▼
┌──────────────┐
│  @cli_command│ ─── Decorator extracts params, creates context
└──────────────┘
       │
       ▼
┌──────────────┐
│  Executor    │ ─── Bootstrap, middleware, dispatch
└──────────────┘
       │
       ▼
┌──────────────┐
│  Handler     │ ─── Business logic, returns CliResult[T]
└──────────────┘
       │
       ▼
┌──────────────┐
│  Renderer    │ ─── Format output, compute exit code
└──────────────┘
       │
       ▼
   Exit (code)
```

---

## 2. Core Components

### 2.1 Component Inventory

| Component | Module | Responsibility |
|-----------|--------|----------------|
| `HandlerContext` | `handlers/context.py` | Unified context for all handlers |
| `CliResult[T]` | `core/results.py` | Structured handler return type |
| `UnifiedRenderer` | `rendering/service.py` | All output formatting |
| `CommandExecutor` | `execution/executor.py` | Orchestrates execution pipeline |
| `OperationRegistry` | `execution/registry.py` | Maps operation IDs to handlers |
| `@cli_command` | `commands/decorators.py` | Declarative command binding |
| `bootstrap_cli()` | `execution/bootstrap.py` | One-time initialization |

### 2.2 Dependencies Between Components

```
@cli_command ──uses──▶ CommandExecutor ──creates──▶ HandlerContext
                              │                          │
                              │                          ▼
                              │                     Handler Function
                              │                          │
                              ▼                          ▼
                      OperationRegistry            CliResult[T]
                              │                          │
                              └────────────▶ UnifiedRenderer
```

---

## 3. Unified HandlerContext

### 3.1 Design Goals

The `HandlerContext` consolidates functionality from:
- `handlers/base.py:HandlerContext`
- `handlers/protocol.py:EnhancedHandlerContext`
- `execution/context.py:ExecutionContext`

Into a **single, canonical context type** that:
1. Provides lazy access to expensive resources (gateway, graph_runtime)
2. Offers typed parameter accessors (replacing scattered `_get_*_param` helpers)
3. Tracks operation metadata (operation_id, start_time, verbosity)
4. Supports both sync and async execution patterns
5. Manages resource lifecycle (close on completion)

### 3.2 Context Structure

```
HandlerContext
├── Immutable Configuration
│   ├── config: CliConfig
│   ├── operation_id: str
│   ├── output_format: OutputFormat
│   └── verbosity: int
│
├── Lazy Resources (resolved on first access)
│   ├── runtime: ResolvedRuntime
│   ├── gateway: StorageGateway
│   └── graph_runtime: GraphRuntime
│
├── Parameter Access
│   ├── params: Mapping[str, object]  (raw)
│   ├── param_str(key, default) -> str | None
│   ├── param_int(key, default) -> int
│   ├── param_bool(key, default) -> bool
│   ├── param_path(key, default) -> Path | None
│   ├── param_list(key) -> list[str]
│   ├── require_str(key) -> str  (raises if missing)
│   └── require_path(key) -> Path
│
├── Utilities
│   ├── logger: Logger  (named for operation)
│   ├── elapsed_seconds: float
│   └── is_dry_run: bool
│
└── Lifecycle
    └── close() -> None  (cleanup resources)
```

### 3.3 Lazy Resolution Behavior

| Resource | Trigger | Resolution Source |
|----------|---------|-------------------|
| `runtime` | First `.runtime` access | `RuntimeResolver.resolve()` using params |
| `gateway` | First `.gateway` access | `open_gateway()` using `runtime.db_path` |
| `graph_runtime` | First `.graph_runtime` access | `build_graph_runtime()` using gateway + snapshot |

Resolution failures propagate as exceptions with appropriate `ProblemDetail` wrapping.

### 3.4 Parameter Accessor Semantics

| Method | Missing Key | Type Mismatch |
|--------|-------------|---------------|
| `param_str(key, default=None)` | Returns `default` | Coerces via `str()` |
| `param_int(key, default=0)` | Returns `default` | Coerces via `int()` |
| `param_bool(key, default=False)` | Returns `default` | Truthy check or string parse |
| `param_path(key, default=None)` | Returns `default` | Coerces via `Path()` |
| `require_str(key)` | Raises `ValueError` | Coerces via `str()` |
| `require_path(key)` | Raises `ValueError` | Coerces via `Path()` |

Enum parameters: Use `param_str()` then convert, or add `param_enum()` helper.

### 3.5 Context Manager Support

```python
# Automatic cleanup via context manager protocol
with handler_context(...) as ctx:
    result = my_handler(ctx)
# gateway closed automatically

# Or explicit cleanup
ctx = build_handler_context(...)
try:
    result = my_handler(ctx)
finally:
    ctx.close()
```

---

## 4. Command Layer

### 4.1 Design Goals

Eliminate boilerplate in command `__call__` methods by:
1. Providing a declarative `@cli_command` decorator
2. Automatically extracting params from dataclass fields
3. Handling context creation, handler dispatch, and rendering
4. Managing exit codes consistently

### 4.2 Current Boilerplate (To Eliminate)

Every command currently implements:
```python
def __call__(self) -> None:
    runtime_cli = RuntimeCLI()
    output_cli = OutputFormatCLI(output_format=self.output_format)
    params: dict[str, object] = {"field1": self.field1, ...}
    
    with command_context("op.id", runtime_cli, output_cli, params=params) as (ctx, renderer):
        result = handler(ctx)
        exit_code = renderer.render_result(result)
        if exit_code != 0:
            sys.exit(exit_code)
```

### 4.3 Target Pattern: Declarative Binding

```python
@cli_command(
    operation_id="jobs.list",
    handler=jobs_list_handler,
    require_runtime=False,
)
@dataclass
class JobsListCommand:
    """List background jobs."""
    
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True)] = 0
```

The `@cli_command` decorator:
1. Generates `__call__` that extracts all dataclass fields as params
2. Creates `HandlerContext` with appropriate settings
3. Invokes the referenced handler
4. Renders result and exits with appropriate code

### 4.4 Decorator Behavior

```python
def cli_command(
    operation_id: str,
    handler: Callable[[HandlerContext], CliResult[T]],
    *,
    require_runtime: bool = True,
    category: OperationCategory = OperationCategory.READ,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator that transforms a Cyclopts command dataclass into
    a fully-wired command with automatic context and rendering.
    
    Generates __call__ that:
    1. Extracts verbosity from self.verbose (if present)
    2. Extracts output_format from self.output_format (if present)
    3. Collects all other fields into params dict
    4. Calls bootstrap_cli() for logging/config
    5. Creates HandlerContext
    6. Invokes handler
    7. Renders result via UnifiedRenderer
    8. Calls sys.exit() with appropriate code
    """
```

### 4.5 Standard Fields

Commands should include these standard fields for consistency:

| Field | Type | Purpose |
|-------|------|---------|
| `output_format` | `OutputFormat` | Output format selection |
| `verbose` | `int` | Verbosity level (count flag) |
| `project_root` | `Path \| None` | Optional explicit project root |
| `repo` | `str \| None` | Repository slug override |
| `commit` | `str \| None` | Commit SHA override |
| `db_path` | `Path \| None` | Database path override |

The `CommonOptions` dataclass in `options/common.py` can be embedded for commands needing full runtime control.

### 4.6 Manual Override

For commands with complex logic, the decorator can be skipped:

```python
@dataclass
class ComplexCommand:
    """Command with custom __call__ logic."""
    
    def __call__(self) -> None:
        # Custom implementation when decorator doesn't fit
        ...
```

---

## 5. Handler Layer

### 5.1 Handler Contract

All handlers MUST follow this signature:

```python
def handler_name(ctx: HandlerContext) -> CliResult[T]:
    """
    Handler docstring.
    
    Parameters
    ----------
    ctx
        Handler context with params:
        - param1: Description
        - param2: Description
    
    Returns
    -------
    CliResult[T]
        Success result with data or failure with ProblemDetail.
    """
```

### 5.2 Handler Responsibilities

| Responsibility | Handler Owns | Framework Owns |
|----------------|--------------|----------------|
| Business logic | ✓ | |
| Parameter extraction | ✓ (via `ctx.param_*`) | |
| Resource access | ✓ (via `ctx.gateway`) | |
| Error handling | ✓ (return `CliResult.fail()`) | |
| Result construction | ✓ (return `CliResult.ok()`) | |
| Logging setup | | ✓ |
| Context creation | | ✓ |
| Output rendering | | ✓ |
| Exit code | | ✓ |
| Resource cleanup | | ✓ |

### 5.3 Handler Patterns

**Simple Read Handler:**
```python
def list_items_handler(ctx: HandlerContext) -> CliResult[ListItemsResult]:
    category = ctx.param_str("category")
    limit = ctx.param_int("limit", 20)
    
    items = fetch_items(ctx.gateway, category=category, limit=limit)
    
    return CliResult.ok(ListItemsResult(items=items, count=len(items)))
```

**Handler with Validation:**
```python
def process_handler(ctx: HandlerContext) -> CliResult[ProcessResult]:
    target = ctx.require_str("target")  # Raises if missing
    
    if not is_valid_target(target):
        return CliResult.fail(ProblemDetail(
            type="urn:codeintel:cli:invalid-target",
            title="Invalid Target",
            detail=f"Target '{target}' is not valid",
            status=400,
        ))
    
    result = process(target)
    return CliResult.ok(ProcessResult(processed=result))
```

**Handler with Warnings:**
```python
def migrate_handler(ctx: HandlerContext) -> CliResult[MigrateResult]:
    warnings: list[str] = []
    
    if ctx.param_bool("force"):
        warnings.append("Force mode enabled, skipping validation")
    
    result = perform_migration(ctx.gateway)
    
    return CliResult.ok(MigrateResult(...), warnings=warnings)
```

### 5.4 Handler Location

Handlers live in `handlers/*.py` organized by domain:
- `handlers/build.py` - Build operations
- `handlers/graphs.py` - Graph operations
- `handlers/ops.py` - Serving operations
- `handlers/jobs.py` - Job management
- etc.

Each handler file exports its handler functions via `__all__`.

---

## 6. Rendering Layer

### 6.1 Design Goals

Consolidate into a **single rendering implementation** (`UnifiedRenderer`) that:
1. Handles all output formats (TEXT, JSON, JSONL)
2. Renders tables with Rich or plain text
3. Formats errors using RFC 9457 Problem Details
4. Computes exit codes
5. Manages stdout/stderr appropriately

### 6.2 Renderer Interface

```python
class UnifiedRenderer:
    """Single renderer for all CLI output."""
    
    def __init__(self, ctx: RenderContext) -> None: ...
    
    # Primary entry point for CliResult
    def render_result(self, result: CliResult[T]) -> int:
        """Render result, return exit code."""
    
    # Specialized rendering
    def render_table(self, rows: Sequence[dict], spec: TableSpec) -> None: ...
    def render_error(self, error: ProblemDetail) -> None: ...
    def render_message(self, message: str, *, level: str = "info") -> None: ...
    def emit_progress(self, current: int, total: int, message: str | None = None) -> None: ...
```

### 6.3 Exit Code Computation

| Condition | Exit Code |
|-----------|-----------|
| `result.success == True` | 0 |
| `result.error.status < 500` | 1 (user error) |
| `result.error.status >= 500` | 2 (internal error) |
| No error but `success == False` | 1 |

### 6.4 Output Format Behavior

| Format | Data Rendering | Error Rendering | Table Rendering |
|--------|---------------|-----------------|-----------------|
| TEXT | Pretty-print, Rich if TTY | Styled error message | Rich table or plain |
| JSON | `{"data": ..., "metadata": ...}` | JSON Problem Details | JSON array |
| JSONL | One JSON object per line | JSON Problem Details | One object per row |

### 6.5 Canonical Type Definitions

All table/column types come from `rendering/table.py`:

```python
# rendering/table.py - SINGLE SOURCE
@dataclass(frozen=True)
class ColumnSpec:
    key: str
    header: str
    style: str | None = None
    justify: JustifyMethod = "left"
    width: int | None = None

@dataclass(frozen=True)
class TableSpec:
    columns: tuple[ColumnSpec, ...]
    title: str | None = None
    caption: str | None = None
    show_row_numbers: bool = False
    empty_message: str = "No data."
```

`rendering/renderers.py` is **deleted**; its functionality is merged into `service.py`.

---

## 7. Execution Pipeline

### 7.1 Pipeline Stages

```
1. Bootstrap
   └── bootstrap_cli(verbosity) → CliConfig
       ├── Load config (once)
       └── Setup logging (once)

2. Context Creation
   └── HandlerContext(config, operation_id, params, output_format, verbosity)

3. Pre-Execution (Optional)
   └── Middleware: logging, metrics, tracing

4. Handler Dispatch
   └── handler(ctx) → CliResult[T]

5. Post-Execution (Optional)
   └── Middleware: cleanup, metrics finalization

6. Rendering
   └── renderer.render_result(result) → exit_code

7. Exit
   └── sys.exit(exit_code)
```

### 7.2 Executor Responsibilities

The `CommandExecutor` orchestrates the pipeline:

```python
class CommandExecutor:
    """Orchestrates CLI command execution."""
    
    def execute(
        self,
        operation_id: str,
        handler: Callable[[HandlerContext], CliResult[T]],
        params: dict[str, object],
        *,
        output_format: OutputFormat = OutputFormat.TEXT,
        verbosity: int = 0,
        require_runtime: bool = True,
    ) -> int:
        """
        Execute a handler and return exit code.
        
        1. Bootstrap CLI (idempotent)
        2. Create HandlerContext
        3. Run pre-middleware
        4. Call handler
        5. Run post-middleware
        6. Render result
        7. Return exit code
        """
```

### 7.3 Middleware Integration

Existing middleware (`execution/middleware.py`) integrates at stages 3 and 5:

```python
# Before handler
mw_contexts = middleware_stack.execute_before(ctx)

# Handler call
result = handler(ctx)

# After handler
result = middleware_stack.execute_after(ctx, result, mw_contexts)
```

---

## 8. Operation Registry

### 8.1 Design Goals

Provide a **single registry** that:
1. Maps operation IDs to handler functions
2. Stores metadata (category, description, param schema)
3. Supports discovery for help/introspection
4. Enables programmatic execution

### 8.2 Registry Structure

```python
@dataclass(frozen=True)
class OperationSpec[T]:
    """Specification for a registered operation."""
    
    operation_id: str                                    # e.g., "build.run"
    handler: Callable[[HandlerContext], CliResult[T]]   # Handler function
    category: OperationCategory                         # READ, WRITE, BUILD, etc.
    description: str = ""                               # Human-readable
    require_runtime: bool = True                        # Needs project context
    param_schema: ValidationSchema | None = None        # Optional validation

class OperationRegistry:
    """Central registry for all CLI operations."""
    
    def register(self, spec: OperationSpec[T]) -> OperationSpec[T]: ...
    def get(self, operation_id: str) -> OperationSpec | None: ...
    def list_operations(self, *, category: OperationCategory | None = None) -> list[OperationSpec]: ...
    def execute(self, operation_id: str, params: dict) -> CliResult: ...
```

### 8.3 Registration Patterns

**Explicit registration:**
```python
# In handlers/__init__.py or handlers/registration.py
from codeintel.cli.execution.registry import register_operation, OperationSpec

register_operation(OperationSpec(
    operation_id="jobs.list",
    handler=jobs_list_handler,
    category=OperationCategory.READ,
    description="List background jobs",
    require_runtime=False,
))
```

**Via decorator (when using @cli_command):**
```python
# @cli_command automatically registers the operation
@cli_command(operation_id="jobs.list", handler=jobs_list_handler, ...)
class JobsListCommand: ...
```

### 8.4 Elimination of `operations/*_operations.py`

The current `operations/*.py` files contain placeholder specs that wrap handlers. These are **eliminated** in favor of:
1. Direct registration in handler modules, or
2. Automatic registration via `@cli_command`

---

## 9. Bootstrap and Lifecycle

### 9.1 Bootstrap Function

```python
# execution/bootstrap.py

_BOOTSTRAPPED: bool = False
_CONFIG: CliConfig | None = None

def bootstrap_cli(
    verbosity: int = 0,
    config: CliConfig | None = None,
    *,
    force: bool = False,
) -> CliConfig:
    """
    One-time CLI initialization.
    
    - Loads configuration (env → file → defaults)
    - Configures logging based on verbosity
    - Returns the active CliConfig
    
    Subsequent calls return cached config unless force=True.
    """
    global _BOOTSTRAPPED, _CONFIG
    
    if _BOOTSTRAPPED and not force:
        return _CONFIG or load_config()
    
    _CONFIG = config or load_config(validate=False)
    setup_logging(verbosity, config=_CONFIG)
    _BOOTSTRAPPED = True
    
    return _CONFIG
```

### 9.2 Logging Configuration

Logging is configured **once** during bootstrap:

| Verbosity | Level | Use Case |
|-----------|-------|----------|
| 0 | WARNING | Default, errors and warnings only |
| 1 | INFO | Progress and status messages |
| 2+ | DEBUG | Detailed debugging output |

### 9.3 Resource Lifecycle

```
Command Start
     │
     ▼
┌────────────────┐
│  bootstrap_cli │ ─── Config loaded, logging configured
└────────────────┘
     │
     ▼
┌────────────────┐
│ HandlerContext │ ─── Created (resources NOT yet opened)
└────────────────┘
     │
     ▼
┌────────────────┐
│    Handler     │ ─── May access ctx.gateway (opens lazily)
└────────────────┘
     │
     ▼
┌────────────────┐
│  ctx.close()   │ ─── Gateway closed, runtime cleared
└────────────────┘
     │
     ▼
Command End
```

---

## 10. Type Contracts

### 10.1 Core Types

```python
# Handler signature
Handler[T] = Callable[[HandlerContext], CliResult[T]]

# Result type
CliResult[T]:
    success: bool
    data: T | None
    error: ProblemDetail | None
    warnings: list[str]
    metadata: dict[str, object]

# Problem detail (RFC 9457)
ProblemDetail:
    type: str           # URN identifier
    title: str          # Short summary
    status: int         # HTTP-like status code
    detail: str | None  # Detailed explanation
    instance: str | None
    extensions: dict[str, object]

# Output format
OutputFormat: Enum[TEXT, JSON, JSONL]
```

### 10.2 Context Protocol

For testing and extensibility, `HandlerContext` implements:

```python
class HandlerContextProtocol(Protocol):
    @property
    def config(self) -> CliConfig: ...
    @property
    def operation_id(self) -> str: ...
    @property
    def runtime(self) -> ResolvedRuntime: ...
    @property
    def gateway(self) -> StorageGateway: ...
    @property
    def logger(self) -> logging.Logger: ...
    
    def param_str(self, key: str, default: str | None = None) -> str | None: ...
    def param_int(self, key: str, default: int = 0) -> int: ...
    def param_bool(self, key: str, *, default: bool = False) -> bool: ...
    def require_str(self, key: str) -> str: ...
    
    def close(self) -> None: ...
```

### 10.3 Renderer Protocol

```python
class RendererProtocol(Protocol):
    def render_result(self, result: CliResult[T]) -> int: ...
    def render_table(self, rows: Sequence[dict], spec: TableSpec) -> None: ...
    def render_error(self, error: ProblemDetail) -> None: ...
    def render_message(self, message: str, *, level: str = "info") -> None: ...
```

---

## 11. Module Layout

### 11.1 Target Structure

```
src/codeintel/cli/
├── __init__.py
│
├── commands/                    # Cyclopts command definitions
│   ├── __init__.py
│   ├── _common.py              # RuntimeCLI, OutputFormatCLI (kept for compat)
│   ├── decorators.py           # NEW: @cli_command decorator
│   ├── app.py                  # Root Cyclopts app
│   ├── build.py
│   ├── jobs.py
│   └── ...
│
├── handlers/                    # Business logic
│   ├── __init__.py
│   ├── context.py              # NEW: Unified HandlerContext
│   ├── build.py
│   ├── jobs.py
│   └── ...
│
├── execution/                   # Execution infrastructure
│   ├── __init__.py
│   ├── bootstrap.py            # NEW: bootstrap_cli()
│   ├── executor.py             # CommandExecutor (simplified)
│   ├── registry.py             # MOVED: OperationRegistry (from introspection/)
│   ├── middleware.py
│   ├── progress.py
│   └── types.py
│
├── rendering/                   # Output formatting
│   ├── __init__.py
│   ├── service.py              # UnifiedRenderer (canonical)
│   ├── table.py                # ColumnSpec, TableSpec (canonical)
│   ├── types.py                # OutputFormat, RenderContext
│   └── specs.py                # Pre-built table specs
│
├── resolution/                  # Runtime resolution
│   ├── __init__.py
│   ├── runtime.py              # RuntimeResolver
│   ├── types.py                # ResolvedRuntime
│   └── errors.py
│
├── core/                        # Core types
│   ├── __init__.py
│   ├── results.py              # CliResult
│   └── result_types.py         # Domain result dataclasses
│
├── errors/                      # Error types
│   ├── __init__.py
│   ├── taxonomy.py
│   └── _cli_errors.py
│
├── config/                      # Configuration
│   └── ...
│
├── options/                     # Shared option bundles
│   ├── __init__.py
│   └── common.py               # CommonOptions
│
└── project/                     # Project discovery
    └── ...
```

### 11.2 Files to Delete

After consolidation:

| File | Reason |
|------|--------|
| `handlers/base.py` | Merged into `handlers/context.py` |
| `handlers/protocol.py` | Merged into `handlers/context.py` |
| `rendering/renderers.py` | Merged into `rendering/service.py` |
| `execution/context.py` | Merged into `handlers/context.py` |
| `execution/adapter.py` | Replaced by `@cli_command` decorator |
| `commands/context.py` | Replaced by decorator + bootstrap |
| `introspection/registry.py` | Moved to `execution/registry.py` |
| `operations/*.py` | Eliminated (specs move to registration) |

### 11.3 New Files

| File | Purpose |
|------|---------|
| `handlers/context.py` | Unified `HandlerContext` |
| `commands/decorators.py` | `@cli_command` decorator |
| `execution/bootstrap.py` | `bootstrap_cli()` function |
| `execution/registry.py` | Merged `OperationRegistry` |

---

## 12. Migration Strategy

### 12.1 Phase Overview

| Phase | Focus | Duration |
|-------|-------|----------|
| Phase 1 | Foundation (context, bootstrap) | Week 1 |
| Phase 2 | Handlers (param accessors) | Week 2 |
| Phase 3 | Rendering (single stack) | Week 3 |
| Phase 4 | Commands (decorator) | Week 4 |
| Phase 5 | Registry & cleanup | Week 5 |

### 12.2 Phase 1: Foundation

**Goal:** Create unified `HandlerContext` and `bootstrap_cli()`.

1. Create `handlers/context.py` with new `HandlerContext`
2. Implement all param accessor methods
3. Implement lazy resolution for runtime, gateway, graph_runtime
4. Create `execution/bootstrap.py` with `bootstrap_cli()`
5. Update `commands/context.py` to use new `HandlerContext`
6. Add deprecation warnings to old context types

**Validation:** Existing commands still work with new context.

### 12.3 Phase 2: Handlers

**Goal:** Migrate handlers to use `ctx.param_*` methods.

1. Delete local `_get_str_param`, `_get_int_param`, etc. from each handler
2. Replace with `ctx.param_str()`, `ctx.param_int()`, etc.
3. Ensure all handlers return `CliResult[T]`
4. Update handler docstrings to document params

**Validation:** All handler tests pass.

### 12.4 Phase 3: Rendering

**Goal:** Single rendering implementation.

1. Ensure `rendering/table.py` exports canonical `ColumnSpec`, `TableSpec`
2. Remove duplicate definitions from `rendering/renderers.py`
3. Merge `RichRenderer`, `PlainRenderer`, `get_renderer()` into `UnifiedRenderer`
4. Update `execution/executor.py` to import from `rendering/service.py`
5. Delete `rendering/renderers.py`
6. Update all imports

**Validation:** All output formatting works correctly.

### 12.5 Phase 4: Commands

**Goal:** Declarative command binding.

1. Create `commands/decorators.py` with `@cli_command`
2. Migrate one command (e.g., `jobs.py`) as proof of concept
3. Incrementally migrate remaining commands
4. Delete manual `__call__` boilerplate
5. Remove `commands/context.py` (replaced by decorator internals)

**Validation:** All CLI commands work with new decorator.

### 12.6 Phase 5: Registry & Cleanup

**Goal:** Unified registry, remove dead code.

1. Move `introspection/registry.py` to `execution/registry.py`
2. Integrate `OperationSpec` with `@cli_command` decorator
3. Delete `operations/*.py` placeholder files
4. Delete old context types (`handlers/base.py`, `handlers/protocol.py`)
5. Final cleanup pass

**Validation:** Full test suite passes, `codeintel --help` works.

---

## 13. Decision Log

### 13.1 Why Single HandlerContext?

**Problem:** Five overlapping context types with inconsistent capabilities.

**Decision:** Consolidate to single `HandlerContext` in `handlers/context.py`.

**Rationale:**
- Reduces cognitive load for handler authors
- Ensures consistent param access patterns
- Simplifies testing (one fake to implement)
- Lazy resolution prevents unused resource allocation

### 13.2 Why @cli_command Decorator?

**Problem:** Every command has identical boilerplate in `__call__`.

**Decision:** Decorator generates `__call__` from metadata.

**Rationale:**
- DRY: Define params once as dataclass fields
- Consistent: All commands follow same pattern
- Testable: Handler logic separate from wiring
- Extensible: Decorator can add middleware, metrics, etc.

### 13.3 Why Delete renderers.py?

**Problem:** Two parallel rendering stacks with duplicated types.

**Decision:** Keep `service.py` with `UnifiedRenderer`, delete `renderers.py`.

**Rationale:**
- `UnifiedRenderer` is newer and more complete
- Single source for `ColumnSpec`, `TableSpec` (in `table.py`)
- Executor can use `UnifiedRenderer` directly
- Eliminates theme duplication

### 13.4 Why Centralized Bootstrap?

**Problem:** Logging setup called in multiple places, sometimes multiple times.

**Decision:** Single `bootstrap_cli()` with idempotency guard.

**Rationale:**
- Logging configured once, correctly
- Config loaded once, reused
- Clear initialization sequence
- Easier to test (can force re-bootstrap)

### 13.5 Why Eliminate operations/*.py?

**Problem:** Placeholder specs that don't add value beyond handler registration.

**Decision:** Register operations at handler definition or via `@cli_command`.

**Rationale:**
- Specs and handlers in same place = easier maintenance
- Decorator handles registration automatically
- Reduces indirection
- Introspection uses registry, not separate files

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Handler** | Function that implements CLI operation logic |
| **HandlerContext** | Unified context object passed to handlers |
| **CliResult** | Structured return type from handlers |
| **ProblemDetail** | RFC 9457 error format |
| **CommandExecutor** | Orchestrates handler execution pipeline |
| **OperationRegistry** | Maps operation IDs to handlers |
| **UnifiedRenderer** | Single implementation for all output |
| **Bootstrap** | One-time initialization (logging, config) |

---

## Appendix B: Example Migration

### Before (Current State)

```python
# commands/jobs.py
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT

    def __call__(self) -> None:
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)
        params: dict[str, object] = {"status": self.status, "limit": self.limit}
        
        with command_context("jobs.list", runtime_cli, output_cli, params=params, require_runtime=False) as (ctx, renderer):
            result = jobs_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


# handlers/jobs.py
def _get_str_param(ctx, name, default=None):
    value = ctx.params.get(name)
    return str(value) if value is not None else default

def _get_int_param(ctx, name, default=0):
    value = ctx.params.get(name)
    return int(value) if value is not None else default

def jobs_list_handler(ctx: EnhancedHandlerContext) -> CliResult[JobsListResult]:
    status_str = _get_str_param(ctx, "status")
    limit = _get_int_param(ctx, "limit", 20)
    # ... implementation
```

### After (Target State)

```python
# commands/jobs.py
@cli_command(operation_id="jobs.list", handler=jobs_list_handler, require_runtime=False)
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True)] = 0
    
    # No __call__ needed - decorator generates it


# handlers/jobs.py
def jobs_list_handler(ctx: HandlerContext) -> CliResult[JobsListResult]:
    """List background jobs.
    
    Parameters
    ----------
    ctx
        Handler context with params:
        - status: Optional status filter
        - limit: Maximum jobs to return (default 20)
    """
    status_str = ctx.param_str("status")
    limit = ctx.param_int("limit", 20)
    # ... implementation (same business logic)
```

---

## Appendix C: Testing Considerations

### Handler Testing

```python
def test_jobs_list_handler():
    # Create test context with fake/test implementations
    ctx = build_test_context(
        operation_id="jobs.list",
        params={"status": "running", "limit": 10},
    )
    
    result = jobs_list_handler(ctx)
    
    assert result.success
    assert len(result.data.jobs) <= 10
```

### Renderer Testing

```python
def test_unified_renderer_json():
    ctx, stdout, stderr = RenderContext.for_testing()
    ctx = RenderContext(format=OutputFormat.JSON, color=False, writer=stdout, err_writer=stderr)
    renderer = UnifiedRenderer(ctx)
    
    result = CliResult.ok({"key": "value"})
    exit_code = renderer.render_result(result)
    
    assert exit_code == 0
    assert json.loads(stdout.getvalue())["data"] == {"key": "value"}
```

### Command Testing

With `@cli_command`, commands can be tested at the handler level (easier) or integration tested via Cyclopts test utilities.

---

*End of Specification*
