# CLI Unified Architecture Specification

> **Status:** Active (Phase 5 Complete)  
> **Author:** Architecture Team  
> **Date:** December 2024  
> **Last Updated:** December 2024 (Post-Phase 5)  
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
HandlerContext (implemented in handlers/context.py)
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
│   ├── _params: dict[str, object]  (internal storage)
│   ├── params: dict[str, object]   (read-only view for backward compat)
│   │
│   ├── Optional Accessors (return default/None if missing)
│   │   ├── param_str(key, default=None) -> str | None
│   │   ├── param_int(key, default=0) -> int
│   │   ├── param_bool(key, default=False) -> bool
│   │   ├── param_path(key, default=None) -> Path | None
│   │   ├── param_enum(key, enum_type, default=None) -> E | None
│   │   ├── param_list(key, default=None) -> list[str]
│   │   └── param_tuple(key, default=None) -> tuple[str, ...]
│   │
│   └── Required Accessors (raise ParameterError if missing)
│       ├── require_str(key) -> str
│       ├── require_int(key) -> int
│       └── require_path(key) -> Path
│
├── Utilities
│   ├── logger: Logger  (named for operation)
│   ├── db_path: Path | None  (shortcut to runtime.db_path)
│   └── color_enabled: bool  (from config)
│
└── Lifecycle
    ├── close() -> None  (cleanup resources)
    ├── __enter__() -> HandlerContext
    └── __exit__() -> None
```

**Key Implementation Notes:**
- `ParameterError` is the specific exception type for missing/invalid required parameters (subclass of `ValueError`)
- The `params` property provides read-only access to `_params` for backward compatibility
- `param_list()` and `param_tuple()` handle conversion from various sequence types

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
| `param_bool(key, default=False)` | Returns `default` | Truthy check or string parse ("true"/"1"/"yes"/"on") |
| `param_path(key, default=None)` | Returns `default` | Coerces via `Path()` |
| `param_enum(key, enum_type, default=None)` | Returns `default` | Attempts `enum_type(value)`, returns default on error |
| `param_list(key, default=None)` | Returns `default` or `[]` | Coerces via `[str(v) for v in value]` |
| `param_tuple(key, default=None)` | Returns `default` or `()` | Coerces via `tuple(str(v) for v in value)` |
| `require_str(key)` | Raises `ParameterError` | Coerces via `str()` |
| `require_int(key)` | Raises `ParameterError` | Coerces via `int()`, raises on conversion error |
| `require_path(key)` | Raises `ParameterError` | Coerces via `Path()` |

**Note:** `ParameterError` is defined in `handlers/context.py` and is a subclass of `ValueError`. It includes a `key` attribute identifying which parameter caused the error.

### 3.5 Context Manager Support

```python
# HandlerContext implements __enter__/__exit__ directly
with HandlerContext(config=config, operation_id="my.op", _params={...}) as ctx:
    result = my_handler(ctx)
# gateway closed automatically

# Or use the context manager factory function
from codeintel.cli.handlers.context import handler_context_manager

with handler_context_manager(config, "my.op", params={"key": "value"}) as ctx:
    result = my_handler(ctx)
# cleanup handled

# Or explicit cleanup
ctx = HandlerContext(config=config, operation_id="my.op", _params={...})
try:
    result = my_handler(ctx)
finally:
    ctx.close()
```

**Migration Adapter (Temporary - Phase 6 Removal):**
```python
# For gradual migration, convert legacy EnhancedHandlerContext
from codeintel.cli.handlers.context import handler_context_from_enhanced

new_ctx = handler_context_from_enhanced(legacy_ctx, "my.op", extra_params)
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
from codeintel.cli.commands.decorators import CommandConfig, cli_command

# Config bundles resource requirements
_JOBS_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

@cli_command("jobs.list", handler=jobs_list_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True)] = 0
    
    # No __call__ needed - decorator generates it
```

The `@cli_command` decorator:
1. Registers the operation with the NEW `OperationRegistry`
2. Generates a `__call__` method that extracts dataclass fields as params
3. Creates `HandlerContext` with appropriate settings
4. Invokes the referenced handler
5. Renders result via `UnifiedRenderer` and exits with appropriate code

### 4.4 CommandConfig Dataclass

Resource requirements are bundled into a `CommandConfig` to keep the decorator signature clean:

```python
@dataclass(frozen=True)
class CommandConfig:
    """Configuration for @cli_command decorator.
    
    Parameters
    ----------
    require_runtime
        Whether handler needs ResolvedRuntime (default True).
    require_gateway
        Whether handler needs StorageGateway (default True).
    require_graph_runtime
        Whether handler needs GraphRuntime (default False).
    description
        Optional description (defaults to class docstring).
    """
    
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    description: str | None = None

# Default configuration (runtime + gateway required)
DEFAULT_CONFIG = CommandConfig()
```

### 4.5 Decorator Signature and Behavior

```python
def cli_command[T, R](
    operation_id: str,
    *,
    handler: Callable[[HandlerContext], CliResult[R]],
    config: CommandConfig | None = None,
) -> Callable[[type[T]], type[T]]:
    """
    Decorate CLI command dataclasses with automatic execution.
    
    Uses PEP 695 type parameters for proper generic handling:
    - T: Command dataclass type
    - R: Handler result type
    
    At decoration time:
    1. Registers operation with NEW OperationRegistry
    2. Uses class docstring as description (or config.description)
    3. Extracts group from operation_id (e.g., "jobs.list" → "jobs")
    4. Generates __call__ method for the class
    
    Generated __call__ behavior:
    1. Extracts verbosity from self.verbose (default 0)
    2. Calls bootstrap_cli() for logging/config
    3. Extracts output_format (checks self.output_format, self.json flag)
    4. Collects non-infrastructure fields into params dict
    5. Creates HandlerContext with extracted params
    6. Invokes handler within context manager (ensures cleanup)
    7. Renders result via get_renderer()
    8. Calls sys.exit() if exit_code != 0
    """
```

**Note:** The decorator handles execution directly rather than delegating to `CommandExecutor.execute()`. This simplifies the architecture while maintaining all required functionality. The existing `CommandExecutor` in `execution/executor.py` remains available for programmatic execution and middleware integration but is not in the critical path for CLI commands.

### 4.6 Infrastructure Fields

The decorator automatically excludes infrastructure fields from the params dict:

```python
_INFRASTRUCTURE_FIELDS = frozenset({
    "output_format",  # Used for rendering selection
    "verbose",        # Used for logging level
    "json",           # Alternative format flag
    "project",        # Alias for project_root
    "project_root",   # Used for runtime resolution
    "db_path",        # Used for gateway resolution
    "database_path",  # Alias for db_path
    "build_dir",      # Build directory
    "index_path",     # Index path
})
```

**Important:** Fields like `repo`, `repo_root`, and `commit` are NOT excluded because some handlers (e.g., `history.timeseries`) use them as actual command parameters, not just infrastructure.

### 4.7 Standard Fields

Commands should include these standard fields for consistency:

| Field | Type | Purpose | Infrastructure? |
|-------|------|---------|-----------------|
| `output_format` | `OutputFormat` | Output format selection | Yes |
| `verbose` | `int` | Verbosity level (count flag) | Yes |
| `project_root` | `Path \| None` | Project root for runtime | Yes |
| `db_path` | `Path \| None` | Database path override | Yes |
| `repo` | `str \| None` | Repository slug | No (passed to handler) |
| `commit` | `str \| None` | Commit SHA | No (passed to handler) |
| `repo_root` | `Path \| None` | Repository root | No (passed to handler) |

**Note:** The "Infrastructure?" column indicates whether the field is excluded from params. Fields like `repo`, `commit`, and `repo_root` are passed through because some handlers (e.g., `history.timeseries`) need them as parameters.

### 4.8 Manual Override (Special Cases)

For commands with complex logic that doesn't fit the decorator pattern, keep a manual `__call__`:

```python
# Example: graphs.py - conditional handler dispatch
@graphs_app.command(name="plugins")
@dataclass
class GraphPluginsCommand:
    """List or plan graph plugins (conditional logic)."""
    
    plan: Annotated[bool, Parameter(help="Show execution plan")] = False
    # ... other fields
    
    def __call__(self) -> None:
        # Custom: dispatch to different handlers based on flags
        if self.plan or self.validate_plan:
            result = graph_plugins_plan_handler(ctx)
        else:
            result = graph_plugins_list_handler(ctx)
        # ...
```

**Commands NOT migrated to `@cli_command`:**
- `graphs.py` - Conditional handler dispatch based on `--plan` flag
- Commands with multiple handlers or complex control flow

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

Provide a **unified handler-based registry** while maintaining **backward compatibility** during migration:
1. Maps operation IDs to handler functions
2. Stores metadata (group, resource requirements)
3. Supports discovery for help/introspection
4. Enables programmatic execution

### 8.2 Dual Registry Architecture (Phase 4 Implementation)

The Phase 4 implementation maintains **two separate registries** for backward compatibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NEW Handler Registry                             │
│               execution/registry.py (Target State)                   │
│  - OperationSpec with: group, require_runtime, require_gateway, etc │
│  - Populated by: handlers/*.py module-level registrations           │
│  - Access via: get_registry(), register_operation()                 │
├─────────────────────────────────────────────────────────────────────┤
│                     LEGACY Registry                                  │
│             introspection/registry.py (Backward Compat)             │
│  - OperationSpec with: category (enum), param_schema, etc           │
│  - Populated by: operations/*.py module-level registrations         │
│  - Access via: get_operation_registry(), register_operation()       │
└─────────────────────────────────────────────────────────────────────┘
```

**Rationale:** The dual registry allows gradual migration. New handler-based code uses the new registry, while existing tests and introspection code continue to work with the legacy registry until Phase 6 cleanup.

### 8.3 NEW OperationSpec (execution/registry.py)

```python
@dataclass(frozen=True)
class OperationSpec:
    """Specification for a registered handler-based operation."""
    
    operation_id: str                                    # e.g., "build.run", "jobs.list"
    name: str                                            # Display name
    description: str                                     # Human-readable help text
    handler: Callable[[HandlerContext], CliResult[Any]] # Handler function
    group: str                                           # Command group (e.g., "jobs", "build")
    
    # Resource requirements (inform lazy resolution)
    require_runtime: bool = True                         # Needs ResolvedRuntime
    require_gateway: bool = True                         # Needs StorageGateway
    require_graph_runtime: bool = False                  # Needs GraphRuntime
    
    # Metadata
    tags: tuple[str, ...] = ()                          # Optional tags for filtering
    hidden: bool = False                                 # Hidden from help output

class OperationRegistry:
    """Central registry for handler-based CLI operations."""
    
    _operations: dict[str, OperationSpec]               # Private storage
    
    def register(self, spec: OperationSpec) -> OperationSpec: ...
    def get(self, operation_id: str) -> OperationSpec | None: ...
    def require(self, operation_id: str) -> OperationSpec: ...  # Raises KeyError if not found
    def list_operations(self, *, group: str | None = None, include_hidden: bool = False) -> list[OperationSpec]: ...
    def list_groups(self) -> list[str]: ...
    def unregister(self, operation_id: str) -> bool: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, operation_id: str) -> bool: ...
```

**Design Note:** We use `group: str` rather than an enum because command groups are extensible and correspond directly to CLI subcommand names (e.g., `codeintel jobs list` → group="jobs"). The `tags` field provides additional categorization if needed.

### 8.4 LEGACY OperationSpec (execution/executor.py)

The legacy registry uses a different `OperationSpec` type:

```python
@dataclass
class OperationSpec(Generic[T]):
    """Legacy operation specification (kept for backward compat)."""
    
    operation_id: str
    handler: Callable[..., CliResult[T]]        # Different signature
    category: OperationCategory                  # Enum, not string
    param_schema: ValidationSchema | None
    requires_progress: bool = False
    estimated_duration: float | None = None
    description: str = ""
```

**Key differences from NEW spec:**
- `category: OperationCategory` (enum) vs `group: str`
- `param_schema` for validation vs resource requirement booleans
- `requires_progress` and `estimated_duration` vs `tags` and `hidden`
- Different handler signature expectations

### 8.5 Import Patterns

**For new handler-based code (recommended):**
```python
# In handlers/*.py
from codeintel.cli.execution.registry import (
    OperationSpec,
    register_operation,
    get_registry,
)

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

**For backward-compatible code:**
```python
# In operations/*.py (legacy)
from codeintel.cli.execution import OperationCategory, OperationSpec  # OLD type
from codeintel.cli.introspection.registry import register_operation    # LEGACY registry

register_operation(OperationSpec(
    operation_id="build.status",
    handler=_build_status_handler,
    category=OperationCategory.BUILD,
    param_schema=None,
    requires_progress=False,
    description="Show build target status",
))
```

**Via introspection exports (supports both):**
```python
from codeintel.cli.introspection import (
    # Legacy exports (backward compatible)
    OperationRegistry,           # Legacy registry class
    get_operation_registry,      # Legacy global getter
    register_operation,          # Legacy register function
    
    # New exports (for gradual migration)
    HandlerOperationRegistry,    # New registry class
    HandlerOperationSpec,        # New spec dataclass
    get_handler_registry,        # New global getter
    register_handler_operation,  # New register function
    reset_handler_registry,      # New reset function
)
```

### 8.6 Registration Patterns

**Via `@cli_command` decorator (Recommended - Phase 5):**
```python
# In commands/jobs.py
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.jobs import jobs_list_handler

_JOBS_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

@cli_command("jobs.list", handler=jobs_list_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""  # Becomes operation description
    # ... fields
```

The decorator automatically:
1. Creates `OperationSpec` from the class metadata
2. Uses class docstring (first line) as description
3. Extracts group from operation_id (`"jobs.list"` → `"jobs"`)
4. Registers with `execution/registry.py`

**Handler modules NO LONGER contain registrations:**
```python
# In handlers/jobs.py (Phase 5 state)
# - NO register_operation() calls
# - NO OperationSpec imports from execution.registry
# - Only handler function definitions and result types
```

**Legacy operations/*.py (for backward compatibility):**
```python
# In operations/jobs_operations.py (LEGACY - to be deleted in Phase 6)
# These populate the introspection/registry.py for old code paths
register_operation(OperationSpec(...))  # OLD spec type
```

### 8.7 Phase 6 Cleanup

In Phase 6, the following will be deleted:
- `introspection/registry.py` (legacy registry)
- All `operations/*.py` files (legacy registrations)
- Legacy exports from `introspection/__init__.py`

The NEW registry in `execution/registry.py` becomes the single source of truth.

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
# Test fixture pattern (actual implementation)
from unittest.mock import MagicMock
from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.context import HandlerContext

def _build_test_context(
    params: dict[str, object],
    operation_id: str = "test.op",
) -> HandlerContext:
    """Create HandlerContext for testing."""
    mock_config = MagicMock(spec=CliConfig)
    return HandlerContext(
        config=mock_config,
        operation_id=operation_id,
        _params=params,
    )

# Usage in tests
def test_my_handler():
    ctx = _build_test_context({"name": "value", "count": 10})
    result = my_handler(ctx)
    assert result.success
```

**For handlers requiring resources (gateway, runtime):**
```python
def _build_test_context_with_resources(params: dict[str, object]) -> HandlerContext:
    mock_runtime = MagicMock(spec=ResolvedRuntime)
    mock_runtime.serving = MagicMock(spec=ServingConfig)
    mock_gateway = MagicMock(spec=StorageGateway)
    mock_graph_runtime = MagicMock()

    return HandlerContext(
        config=MagicMock(spec=CliConfig),
        operation_id="test.op",
        _params=params,
        _runtime=mock_runtime,
        _gateway=mock_gateway,
        _graph_runtime=mock_graph_runtime,
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

| File | Reason | Status |
|------|--------|--------|
| `handlers/base.py` | Superseded by `handlers/context.py` | Delete in P6 |
| `handlers/protocol.py` | Superseded by `handlers/context.py` | Delete in P6 |
| `execution/context.py` | Superseded by `handlers/context.py` | Delete in P6 |
| `execution/adapter.py` | Superseded by `commands/decorators.py` | Delete in P6 |
| `commands/context.py` | Superseded by decorator internals | ⚠️ graphs.py uses it |
| `rendering/renderers.py` | Merged into `rendering/service.py` | Already deleted (P2) |
| `introspection/registry.py` | LEGACY registry | Delete in P6 |
| `operations/*.py` | LEGACY registrations | Delete in P6 |

**Phase 5 Cleanup (Already Done):**
- Handler module registration sections removed from `handlers/*.py`
- No more `register_operation()` calls in handler files
- Decorator now handles registration with NEW registry

**Special Cases:**
- `commands/context.py` - Still used by `graphs.py` (conditional handler dispatch)
- Either migrate `graphs.py` to use wrapper handler, or retain `context.py` for special cases

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

| Phase | Name | Focus | Duration | Status |
|-------|------|-------|----------|--------|
| Phase 0 | Preparation | Baselines, inventories, scaffolding | 1-2 days | ✅ Complete |
| Phase 1 | Foundation Layer | `HandlerContext`, `bootstrap_cli()` | 3-4 days | ✅ Complete |
| Phase 2 | Rendering Consolidation | Single `UnifiedRenderer` stack | 2-3 days | ✅ Complete |
| Phase 3 | Handler Migration | All handlers use new context | 5-7 days | ✅ Complete |
| Phase 4 | Registry Unification | Dual registry (NEW + LEGACY compat) | 2-3 days | ✅ Complete |
| Phase 5 | Command Decorator | `@cli_command` + migrate commands | 5-7 days | ✅ Complete |
| Phase 6 | Legacy Cleanup | Delete all superseded files | 2-3 days | 🔄 Next |

**Total:** 4-6 weeks (20-29 working days)

### 12.1.1 Phase 5 Implementation Summary

Phase 5 migrated **~36 commands** across **12 command files** to the `@cli_command` decorator:

| File | Commands Migrated |
|------|-------------------|
| `jobs.py` | 5 (list, status, output, cancel, cleanup) |
| `health.py` | 1 (check) |
| `ide.py` | 1 (hints) |
| `docs.py` | 1 (export) |
| `history.py` | 2 (timeseries, etc.) |
| `storage.py` | 4 (validate_macros, generate_macros, profile, etc.) |
| `build.py` | 4 (run, status, history, etc.) |
| `datasets.py` | 5 (lint, list, snapshot, diff, etc.) |
| `plugins.py` | 7 (list, discover, info, paths, new, test, validate) |
| `subsystem.py` | 6 (list, show, profiles, coverage, module_memberships, etc.) |
| `dataset_ops.py` | 3 (list, describe, verify) |
| `ops.py` | 2 (list, call) |

**Not migrated (require Phase 6 attention):**
- `graphs.py` - Conditional handler dispatch requires wrapper handler or custom `__call__`
- `serve.py` - MCP/HTTP serve commands, can be migrated to `@cli_command`

**Other files (not command classes):**
- `context.py` - Legacy `command_context()` function (to be deleted in Phase 6)
- `_common.py` - Utilities and type aliases
- `decorators.py` - Contains the `@cli_command` decorator implementation

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

| Milestone | Phase | Validation | Status |
|-----------|-------|------------|--------|
| New context available | 1 | `HandlerContext` unit tests pass | ✅ |
| Rendering unified | 2 | `renderers.py` deleted, all output works | ✅ |
| All handlers migrated | 3 | No `_get_*_param` helpers remain, all use `ctx.param_*()` | ✅ |
| Registry unified | 4 | NEW registry in `execution/registry.py`, handlers register, LEGACY maintained for compat | ✅ |
| All commands migrated | 5 | 34 commands use `@cli_command`, handler registrations removed | ✅ |
| Migration complete | 6 | All legacy files deleted, full tests pass | ⬜ |

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

### 13.6 Why Dual Registry Architecture (Phase 4)?

**Problem:** The existing codebase has tests and introspection code that depend on the legacy `OperationSpec` type (with `category: OperationCategory` enum), but the new handler architecture needs a different spec type (with `group: str` and resource requirement booleans).

**Decision:** Maintain two separate registries during migration:
- **NEW registry** (`execution/registry.py`): Handler-based registrations with new `OperationSpec`
- **LEGACY registry** (`introspection/registry.py`): Backward-compatible registrations with old `OperationSpec`

**Rationale:**
- Allows gradual migration without breaking existing tests
- Handler modules can register to NEW registry immediately
- Existing tests using `get_operation_registry()` continue to work
- Clean deletion path in Phase 6 (remove LEGACY, keep NEW)
- Avoids complex type aliasing or adapter patterns

**Implementation Notes:**
- `introspection/__init__.py` exports from BOTH registries with distinct names
- Legacy: `OperationRegistry`, `get_operation_registry`, `register_operation`
- New: `HandlerOperationRegistry`, `HandlerOperationSpec`, `get_handler_registry`, etc.
- The `operations/*.py` files continue to register to LEGACY (for tests)
- The `handlers/*.py` files register to NEW (for production)

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

### Before (Pre-Phase 3)

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

### Current State (Phase 3 Complete)

Handlers now use `HandlerContext` with typed accessors. Local helper functions eliminated:

```python
# handlers/jobs.py (Current - Phase 3 complete)
from codeintel.cli.handlers.context import HandlerContext

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
    # ... implementation (same business logic, cleaner param access)
```

### After (Target State - Phase 5 Complete)

Commands use `@cli_command` decorator with `CommandConfig`, eliminating all boilerplate:

```python
# commands/jobs.py (Actual implementation)
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.jobs import jobs_list_handler

# Bundle resource requirements in CommandConfig
_JOBS_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

@cli_command("jobs.list", handler=jobs_list_handler, config=_JOBS_CONFIG)
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True)] = 0
    
    # No __call__ needed - decorator generates it


# handlers/jobs.py (same as Phase 3)
# NOTE: No register_operation() calls - decorator handles registration
def jobs_list_handler(ctx: HandlerContext) -> CliResult[JobsListResult]:
    """List background jobs."""
    status_str = ctx.param_str("status")
    limit = ctx.param_int("limit", 20)
    # ... implementation
```

**Key implementation details:**
1. `CommandConfig` bundles `require_runtime`, `require_gateway`, `require_graph_runtime`, `description`
2. The decorator uses PEP 695 type parameters: `def cli_command[T, R](...)`
3. Class docstring (first line) becomes operation description
4. Handler modules no longer contain `register_operation()` calls
5. Special cases (like `graphs.py` with conditional handlers) keep manual `__call__`

---

## Appendix C: Testing Considerations

### Handler Testing (Recommended Pattern)

```python
from unittest.mock import MagicMock
from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.context import HandlerContext, ParameterError
from codeintel.cli.resolution.types import ResolvedRuntime
from codeintel.storage.gateway import StorageGateway


def _build_test_context(params: dict[str, object]) -> HandlerContext:
    """Create HandlerContext for testing with mocked resources."""
    mock_config = MagicMock(spec=CliConfig)
    mock_runtime = MagicMock(spec=ResolvedRuntime)
    mock_gateway = MagicMock(spec=StorageGateway)
    mock_graph_runtime = MagicMock()

    return HandlerContext(
        config=mock_config,
        operation_id="test.op",
        _params=params,
        _runtime=mock_runtime,
        _gateway=mock_gateway,
        _graph_runtime=mock_graph_runtime,
    )


def test_jobs_list_handler():
    """Test handler with mocked context."""
    ctx = _build_test_context({"status": "running", "limit": 10})
    
    result = jobs_list_handler(ctx)
    
    assert result.success
    if result.data is not None:
        assert len(result.data.jobs) <= 10


def test_handler_raises_on_missing_required_param():
    """Test ParameterError raised for missing required params."""
    ctx = _build_test_context({})  # No params
    
    with pytest.raises(ParameterError, match="Required parameter 'name'"):
        my_handler(ctx)
```

### Renderer Testing

```python
def test_unified_renderer_json():
    ctx = RenderContext(format=OutputFormat.JSON, color=False, writer=stdout, err_writer=stderr)
    renderer = UnifiedRenderer(ctx)
    
    result = CliResult.ok({"key": "value"})
    exit_code = renderer.render_result(result)
    
    assert exit_code == 0
    assert json.loads(stdout.getvalue())["data"] == {"key": "value"}
```

### Command Testing

With `@cli_command`, commands can be tested at the handler level (easier) or integration tested via Cyclopts test utilities.

**Key Testing Principles:**
1. Create `HandlerContext` directly with mock dependencies - no special test factories needed
2. Use `ParameterError` for expected required parameter failures
3. Test handlers in isolation from command boilerplate
4. Prefer assertion helpers from `tests/_helpers/assertions/` for consistent error messages

---

*End of Specification*
