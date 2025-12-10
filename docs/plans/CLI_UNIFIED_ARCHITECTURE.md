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
├── Runtime Resolution Parameters
│   ├── project_root: Path | None
│   ├── index_path: Path | None
│   └── database_path: Path | None
│
├── Lazy Resources (resolved on first access)
│   ├── runtime: ResolvedRuntime
│   ├── gateway: StorageGateway
│   └── graph_runtime: GraphRuntime
│
├── Parameter Access
│   ├── _params: dict[str, Any]  (internal storage)
│   ├── param_str(key, default) -> str | None
│   ├── param_int(key, default) -> int
│   ├── param_bool(key, default) -> bool
│   ├── param_path(key, default) -> Path | None
│   ├── param_enum(key, enum_type, default) -> E | None
│   ├── require_str(key) -> str  (raises if missing)
│   ├── require_int(key) -> int
│   └── require_path(key) -> Path
│
├── Utilities
│   ├── logger: Logger  (named for operation)
│   └── db_path: Path | None  (shortcut to runtime.db_path)
│
└── Lifecycle
    ├── close() -> None  (cleanup resources)
    ├── __enter__() -> HandlerContext
    └── __exit__() -> None
```

**Note:** Properties like `elapsed_seconds` and `is_dry_run` can be added if needed but are not part of the core implementation. Handlers can compute elapsed time from `ctx.param_int("start_time")` if tracked, and dry-run is typically a param (`ctx.param_bool("dry_run")`).

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
| `param_int(key, default=0)` | Returns `default` | Coerces via `int()`, returns default on error |
| `param_bool(key, default=False)` | Returns `default` | Truthy check or string parse ("true"/"1"/"yes") |
| `param_path(key, default=None)` | Returns `default` | Coerces via `Path()` |
| `param_enum(key, enum_type, default=None)` | Returns `default` | Attempts `enum_type(value)`, returns default on error |
| `require_str(key)` | Raises `ValueError` | Coerces via `str()` |
| `require_int(key)` | Raises `ValueError` | Coerces via `int()`, raises on conversion error |
| `require_path(key)` | Raises `ValueError` | Coerces via `Path()` |

### 3.5 Context Manager Support

```python
# HandlerContext implements __enter__/__exit__ directly
with HandlerContext(config=config, operation_id="my.op", ...) as ctx:
    result = my_handler(ctx)
# gateway closed automatically

# Or use the context manager factory function
from codeintel.cli.handlers.context import handler_context_manager

with handler_context_manager(config, "my.op", params={"key": "value"}) as ctx:
    result = my_handler(ctx)
# cleanup handled

# Or explicit cleanup
ctx = HandlerContext(config=config, operation_id="my.op", ...)
try:
    result = my_handler(ctx)
finally:
    ctx.close()
```

**Note:** The `@cli_command` decorator handles context creation and cleanup internally, so handler authors don't need to manage the context lifecycle directly.

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
    *,
    handler: Callable[[HandlerContext], CliResult[Any]],
    require_runtime: bool = True,
    require_gateway: bool = True,
    require_graph_runtime: bool = False,
    description: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator that transforms a Cyclopts command dataclass into
    a fully-wired command with automatic context and rendering.
    
    At decoration time:
    1. Registers operation with OperationRegistry (auto-registration)
    2. Generates __call__ method for the class
    
    Generated __call__ behavior:
    1. Extracts verbosity from self.verbose (if present)
    2. Extracts output_format from self.output_format (if present)
    3. Collects all other fields into params dict
    4. Calls bootstrap_cli() for logging/config
    5. Creates HandlerContext with extracted params
    6. Invokes handler within context manager (ensures cleanup)
    7. Renders result via UnifiedRenderer
    8. Calls sys.exit() with appropriate code
    
    The description defaults to the class docstring if not provided.
    """
```

**Note:** The decorator handles execution directly rather than delegating to `CommandExecutor.execute()`. This simplifies the architecture while maintaining all required functionality. The existing `CommandExecutor` in `execution/executor.py` remains available for programmatic execution and middleware integration but is not in the critical path for CLI commands.

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
class OperationSpec:
    """Specification for a registered operation."""
    
    operation_id: str                                    # e.g., "build.run", "jobs.list"
    name: str                                            # Display name
    description: str                                     # Human-readable help text
    handler: Callable[[HandlerContext], CliResult[Any]] # Handler function
    group: str                                           # Command group (e.g., "jobs", "build")
    
    # Resource requirements
    require_runtime: bool = True                         # Needs ResolvedRuntime
    require_gateway: bool = True                         # Needs StorageGateway
    require_graph_runtime: bool = False                  # Needs GraphRuntime
    
    # Metadata
    tags: tuple[str, ...] = ()                          # Optional tags for filtering
    hidden: bool = False                                 # Hidden from help output

class OperationRegistry:
    """Central registry for all CLI operations."""
    
    def register(self, spec: OperationSpec) -> OperationSpec: ...
    def get(self, operation_id: str) -> OperationSpec | None: ...
    def require(self, operation_id: str) -> OperationSpec: ...  # Raises if not found
    def list_operations(self, *, group: str | None = None, include_hidden: bool = False) -> list[OperationSpec]: ...
    def list_groups(self) -> list[str]: ...
```

**Design Note:** We use `group: str` rather than an enum because command groups are extensible and correspond directly to CLI subcommand names (e.g., `codeintel jobs list` → group="jobs"). The `tags` field provides additional categorization if needed.

### 8.3 Registration Patterns

**Explicit registration (in handler modules):**
```python
# In handlers/jobs.py (at module level, after handler definitions)
from codeintel.cli.execution.registry import register_operation, OperationSpec

register_operation(OperationSpec(
    operation_id="jobs.list",
    name="List Jobs",
    description="List background jobs with optional status filtering",
    handler=jobs_list_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
))
```

**Via decorator (when using @cli_command):**
```python
# @cli_command automatically registers the operation
# The decorator extracts description from class docstring
@cli_command("jobs.list", handler=jobs_list_handler, require_runtime=False)
class JobsListCommand:
    """List background jobs."""  # Becomes operation description
    ...
```

**Note:** With the `@cli_command` decorator pattern, explicit registration in handler modules is optional—the decorator handles registration automatically. Explicit registration is useful for operations that don't have a corresponding command class (e.g., programmatic-only operations).

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

### 10.2 HandlerContext Interface

`HandlerContext` is a concrete dataclass, not a Protocol. For testing, create instances directly with mock/fake dependencies:

```python
# Test fixture pattern
def create_test_context(
    operation_id: str = "test.op",
    params: dict[str, Any] | None = None,
    config: CliConfig | None = None,
) -> HandlerContext:
    """Create HandlerContext for testing."""
    return HandlerContext(
        config=config or create_test_config(),
        operation_id=operation_id,
        output_format=OutputFormat.TEXT,
        verbosity=0,
        _params=params or {},
    )
```

### 10.3 Renderer Interface

`UnifiedRenderer` is a concrete class. The `RenderingService` Protocol in `rendering/service.py` defines the interface if abstraction is needed:

```python
class RenderingService(Protocol):
    def render_result(self, result: CliResult[T]) -> int: ...
    def render_table(self, rows: Sequence[dict], spec: TableSpec) -> None: ...
    def render_error(self, error: ProblemDetail) -> None: ...
    def render_message(self, message: str, *, level: str = "info") -> None: ...
```

**Note:** Formal Protocol definitions are optional. The concrete implementations (`HandlerContext`, `UnifiedRenderer`) are sufficient for the architecture. Protocols can be added later if interface abstraction becomes necessary for testing or extensibility.

---

## 11. Module Layout

### 11.1 Target Structure

```
src/codeintel/cli/
├── __init__.py
│
├── commands/                    # Cyclopts command definitions
│   ├── __init__.py
│   ├── _common.py              # make_root_app(), Annotated type aliases (simplified)
│   ├── _help.py                # Help text utilities
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

After consolidation (Phase 6):

| File | Reason |
|------|--------|
| `handlers/base.py` | Superseded by `handlers/context.py` |
| `handlers/protocol.py` | Superseded by `handlers/context.py` |
| `execution/context.py` | Superseded by `handlers/context.py` |
| `execution/adapter.py` | Superseded by `commands/decorators.py` |
| `commands/context.py` | Superseded by decorator internals |
| `rendering/renderers.py` | Merged into `rendering/service.py` |
| `introspection/registry.py` | Moved to `execution/registry.py` |
| `operations/build_operations.py` | Registrations move to handlers |
| `operations/dataset_operations.py` | Registrations move to handlers |
| `operations/docs_operations.py` | Registrations move to handlers |
| `operations/graph_operations.py` | Registrations move to handlers |
| `operations/history_operations.py` | Registrations move to handlers |
| `operations/ide_operations.py` | Registrations move to handlers |
| `operations/op_operations.py` | Registrations move to handlers |
| `operations/storage_operations.py` | Registrations move to handlers |
| `operations/subsystem_operations.py` | Registrations move to handlers |

**Note:** `_common.py` is retained but simplified—`RuntimeCLI` and `OutputFormatCLI` classes can be removed as they're no longer needed with the decorator pattern. The file keeps `make_root_app()` and `Annotated` type aliases used in command definitions.

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

| Phase | Name | Focus | Duration |
|-------|------|-------|----------|
| Phase 0 | Preparation | Baselines, inventories, scaffolding | 1-2 days |
| Phase 1 | Foundation Layer | `HandlerContext`, `bootstrap_cli()` | 3-4 days |
| Phase 2 | Rendering Consolidation | Single `UnifiedRenderer` stack | 2-3 days |
| Phase 3 | Handler Migration | All handlers use new context | 5-7 days |
| Phase 4 | Registry Unification | Single `OperationRegistry` | 2-3 days |
| Phase 5 | Command Decorator | `@cli_command` + migrate commands | 5-7 days |
| Phase 6 | Legacy Cleanup | Delete all superseded files | 2-3 days |

**Total:** 4-6 weeks (20-29 working days)

### 12.2 Phase Dependencies

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                       └─────── Can overlap with early Phase 3
```

### 12.3 Detailed Phase Plans

Detailed implementation plans for each phase are available in:

- `docs/plans/phases/PHASE_0_PREPARATION.md`
- `docs/plans/phases/PHASE_1_FOUNDATION.md`
- `docs/plans/phases/PHASE_2_RENDERING.md`
- `docs/plans/phases/PHASE_3_HANDLERS.md`
- `docs/plans/phases/PHASE_4_REGISTRY.md`
- `docs/plans/phases/PHASE_5_DECORATOR.md`
- `docs/plans/phases/PHASE_6_CLEANUP.md`

Each plan includes:
- Detailed task breakdown with effort estimates
- Code examples and patterns
- Testing requirements
- Verification checklists
- Exit criteria
- Rollback procedures

### 12.4 Key Milestones

| Milestone | Phase | Validation |
|-----------|-------|------------|
| New context available | 1 | `HandlerContext` unit tests pass |
| Rendering unified | 2 | `renderers.py` deleted, all output works |
| All handlers migrated | 3 | No `_get_*_param` helpers remain |
| Registry unified | 4 | Operations registered in handlers |
| All commands migrated | 5 | No manual `__call__` methods remain |
| Migration complete | 6 | All legacy files deleted, full tests pass |

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
@cli_command(
    "jobs.list",
    handler=jobs_list_handler,
    require_runtime=False,
    require_gateway=False,
)
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

**Note:** The decorator's first positional argument is `operation_id`, and `handler` is keyword-only. The decorator automatically:
1. Registers the operation with `OperationRegistry`
2. Uses the class docstring as the operation description
3. Generates a `__call__` method that handles all CLI infrastructure

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
