# Phase 8: CLI Architecture Unification

**Status:** 🚧 IN PROGRESS  
**Last Updated:** December 2024  
**Prerequisites:** Phase 7 Complete

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 8A | Unify OperationSpec | ✅ Complete |
| 8B | Eliminate command_context | ✅ Complete |
| 8C | Consolidate Runtime Resolution | ✅ Complete |
| 8D | Unify Logging Bootstrap | ✅ Complete |
| 8E | Delete OperationExecutor | ✅ Complete |

### Phase 8A Notes

- Unified `OperationSpec` in `execution/registry.py` with optional execution hint fields
- Deleted `executor.OperationSpec` and `OperationCategory` from `executor.py`
- Updated `plugins/registry.py` to import from `execution.registry`
- `OperationExecutor` now has type errors (16 errors) because it used executor-specific
  fields (`is_async`, `is_streaming`, `param_schema`, `retry_policy`, etc.) that are
  not in the unified spec. These will be resolved when the executor is deleted in 8E.

### Phase 8B Notes

- Audited codebase: no usages of `command_context` or `CommandContextError` found
- Deleted `commands/context.py` (~360 lines) - contained legacy context manager
- Deleted `cli/compat.py` (~127 lines) - deprecated compatibility shims
- All deprecated exports (`command_context`, `CommandContextError`, `get_operation_registry`,
  `EnhancedHandlerContext`, `build_handler_context`, `LegacyHandlerContext`) removed
- Total lines deleted: ~487 lines of legacy code

### Phase 8C Notes

- Inlined `RuntimeResolver` directly into `handlers/context.py._resolve_runtime()`
- Deleted `handlers/_lazy_resources.py` (~66 lines) - was just a wrapper around RuntimeResolver
- Updated `execution/context.py` to import directly from `resolution` module
- Deleted `execution/_lazy_deps.py` (~62 lines) - was redundant indirection layer
- Runtime resolution now flows directly: HandlerContext → RuntimeResolver
- Total lines deleted: ~128 lines of indirection code

### Phase 8D Notes

- Updated `handlers/history.py` to use `bootstrap_cli()` instead of `setup_logging()`
- Updated `handlers/__init__.py` to export `bootstrap_cli` instead of `setup_logging`
- Removed from `handlers/_utilities.py`:
  - `_LOGGING_CONFIGURED` global flag
  - `setup_logging()` function (~30 lines)
  - `_determine_log_level()` helper (~15 lines)
  - `VERBOSITY_DEBUG`, `VERBOSITY_INFO` constants (now only in `bootstrap.py`)
- `bootstrap_cli()` is now the single logging/signal initialization point
- Total lines removed: ~50 lines of duplicate logging code

### Phase 8E Notes

- Created simple `execute_operation(spec, params)` function in `registry.py`
- Updated callers to use `execute_operation`:
  - `shell/_shell.py` - interactive shell command execution
  - `jobs/runner.py` - background job execution
  - `project/pipelines.py` - batch operation execution
- Deleted `execution/executor.py` (~734 lines):
  - `OperationExecutor` class (~500 lines)
  - `_ExecutorState` class
  - `get_executor()`, `configure_executor()` functions
  - `run_sync()`, `run_async_operation()` functions
- Updated `execution/__init__.py` exports
- Updated `tests/cli/_harness/__init__.py` to use `execute_operation`
- Fixed test using wrong parameter name (`operation_id` → `op_id`)
- Total lines deleted: ~734 lines of executor code
- Middleware, progress, and types modules preserved for potential future use

## Executive Summary

This phase consolidates the CLI architecture from three parallel execution paths and two competing context types into a single, handler-centric design. The goal is **one spec, one registry, one context, one bootstrap, one resolution path**.

### Current State Problems

1. **Two OperationSpec classes** with incompatible schemas causing type errors
2. **Three execution paths** (`@cli_command`, `command_context`, `OperationExecutor`) with no shared infrastructure
3. **Four runtime resolution implementations** with subtle behavioral differences
4. **Three logging bootstrap mechanisms** that can conflict
5. **Two context types** (`HandlerContext`, `ExecutionContext`) with different param access semantics
6. **800+ lines of unused executor code** that was built but never integrated

### Target State

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Points                              │
│  @cli_command decorates Cyclopts dataclasses                         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Unified OperationSpec                               │
│  Single spec in execution/registry.py                                │
│  • Core: operation_id, name, description, group, handler             │
│  • Resources: require_runtime, require_gateway, require_graph_runtime│
│  • Optional: resilience_config, progress_config                      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Command Pipeline                                    │
│  bootstrap_cli() → HandlerContext → [optional middleware] → handler  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   HandlerContext (THE context)                        │
│  • Rich param_* accessors                                            │
│  • Lazy resources: runtime, gateway, graph_runtime                   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   RuntimeResolver (THE resolver)                      │
│  • Project file discovery + CLI param fallback                       │
│  • Single implementation for all callers                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Narrative

### The Problem: Organic Growth Without Integration

The CLI architecture grew through multiple phases, each adding new capabilities without fully integrating with existing code:

1. **Phase 1-3**: Built `OperationExecutor` with sophisticated middleware, resilience, and progress tracking using `ExecutionContext`
2. **Phase 4-5**: Built `@cli_command` decorator with `HandlerContext` for simpler handler authoring
3. **Phase 6**: Cleaned up legacy patterns but left parallel paths intact
4. **Phase 7**: Removed migration artifacts but didn't address structural duplication

The result is that `@cli_command` (the pattern we actually use) bypasses all the executor infrastructure (middleware, resilience, progress) that was carefully built. Meanwhile, `command_context` exists as a third path that reimplements runtime resolution.

### The Solution: Handler-Centric Unification

We adopt `HandlerContext` as THE context type because:
- It has richer param access (`param_str`, `param_int`, `param_enum`, etc.)
- It's what handlers actually receive
- It already has lazy resource loading

We make `@cli_command` THE execution path because:
- It's the pattern all commands use
- It handles Cyclopts integration cleanly
- It can be enhanced to include optional middleware

We make `RuntimeResolver` THE resolution mechanism because:
- It's already canonical
- Other implementations just duplicate its logic
- Single source of truth prevents drift

### What Gets Deleted

| Component | Lines | Reason |
|-----------|-------|--------|
| `executor.py:OperationSpec` | ~50 | Duplicate spec, unused |
| `executor.py:OperationCategory` | ~10 | Unused enum |
| `executor.py:OperationExecutor` | ~650 | Never called by CLI |
| `commands/context.py` | ~360 | Reimplements resolution |
| `execution/_lazy_deps.py` | ~50 | Duplicate of handlers/_lazy_resources |
| `cli/compat.py` | ~130 | External shim with no consumers |
| `handlers/_utilities.py:setup_logging` | ~40 | Duplicate of bootstrap |
| **Total** | **~1,290** | Dead/duplicate code |

### What Gets Enhanced

| Component | Enhancement |
|-----------|-------------|
| `registry.OperationSpec` | Add optional `resilience_config`, `progress_config` |
| `@cli_command` | Optional middleware integration |
| `bootstrap_cli()` | Becomes single logging/signal setup |
| `HandlerContext` | Minor cleanup, inline lazy loading |

---

## Phase 8A: Unify OperationSpec

**Goal:** Single `OperationSpec` class that serves both registry and execution needs.

### Current State

```python
# execution/executor.py - UNUSED
@dataclass(frozen=True)
class OperationSpec[T]:
    operation_id: str
    handler: AnyHandler[T]
    category: OperationCategory = OperationCategory.READ
    param_schema: ValidationSchema | None = None
    requires_progress: bool = False
    estimated_duration: float | None = None
    retryable: bool = False
    retry_policy: RetryPolicy | None = None
    timeout: float | None = None
    description: str = ""
    is_async: bool | None = None
    is_streaming: bool | None = None

# execution/registry.py - USED
@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    name: str
    description: str
    handler: Callable[[HandlerContext], CliResult[Any]]
    group: str
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    tags: tuple[str, ...] = ()
    hidden: bool = False
```

### Target State

```python
# execution/registry.py - UNIFIED
@dataclass(frozen=True)
class OperationSpec:
    """Unified specification for CLI operations.
    
    Core fields are required. Resource requirements default to True
    for backward compatibility. Execution hints are optional for
    future middleware integration.
    """
    # Core identification
    operation_id: str
    name: str
    description: str
    handler: Callable[[HandlerContext], CliResult[Any]]
    group: str
    
    # Resource requirements
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    
    # Metadata
    tags: tuple[str, ...] = ()
    hidden: bool = False
    
    # Execution hints (optional, for future middleware)
    timeout: float | None = None
    retryable: bool = False
    estimated_duration: float | None = None
```

### Implementation Steps

#### 8A.1: Enhance registry.OperationSpec

**File:** `src/codeintel/cli/execution/registry.py`

```python
# Add new optional fields to existing OperationSpec
@dataclass(frozen=True)
class OperationSpec:
    # ... existing fields ...
    
    # Execution hints (optional, for future middleware)
    timeout: float | None = None
    retryable: bool = False
    estimated_duration: float | None = None
```

#### 8A.2: Update plugins/registry.py imports

**File:** `src/codeintel/cli/plugins/registry.py`

Change:
```python
from codeintel.cli.execution import OperationSpec  # executor version
```

To:
```python
from codeintel.cli.execution.registry import OperationSpec  # registry version
```

Update `PluginProtocol.get_operations()` return type annotation.

#### 8A.3: Delete executor.OperationSpec and OperationCategory

**File:** `src/codeintel/cli/execution/executor.py`

- Delete `class OperationCategory(Enum)` (lines 72-80)
- Delete `class OperationSpec[T]` (lines 82-137)
- Update any internal references in executor to not use these

#### 8A.4: Update execution/__init__.py exports

Remove exports of deleted classes, ensure `OperationSpec` comes from registry.

#### 8A.5: Fix type errors

Run pyright, fix any remaining type mismatches from the consolidation.

### Verification

```bash
# Type check
uv run pyright src/codeintel/cli/execution/ src/codeintel/cli/plugins/

# Import test
uv run python -c "from codeintel.cli.execution.registry import OperationSpec; print('OK')"

# Run related tests
uv run pytest tests/cli/unit/test_operation_handlers.py -v
```

### Rollback

If issues arise, revert the changes to `registry.py` and `plugins/registry.py`.

---

## Phase 8B: Eliminate command_context

**Goal:** Remove `command_context` and `cli/compat.py`, consolidating into `@cli_command`.

### Current State

`commands/context.py` provides `command_context()` which:
1. Loads configuration
2. Sets up logging (via `setup_logging()`)
3. Resolves runtime (reimplements `RuntimeResolver` logic)
4. Creates `HandlerContext`
5. Creates `UnifiedRenderer`

This duplicates what `@cli_command` already does, just with a different API.

### Target State

- `command_context()` deleted
- `cli/compat.py` deleted (it only re-exports `command_context`)
- All commands use `@cli_command` decorator pattern

### Implementation Steps

#### 8B.1: Audit command_context usage

```bash
rg "command_context|from codeintel.cli.commands.context" src/ tests/
```

Identify all callers that need migration.

#### 8B.2: Migrate any remaining callers

For each caller using `command_context`:
- Convert to `@cli_command` decorator pattern, OR
- If programmatic use, call handler directly with manually constructed `HandlerContext`

#### 8B.3: Delete commands/context.py

**File:** `src/codeintel/cli/commands/context.py`

Delete the entire file (360 lines).

#### 8B.4: Delete cli/compat.py

**File:** `src/codeintel/cli/compat.py`

Delete the entire file (127 lines).

#### 8B.5: Update commands/__init__.py

Remove any references to `command_context` or `CommandContextError`.

#### 8B.6: Delete related tests

**File:** `tests/cli/handlers/test_command_context.py` (if exists after Phase 7)

### Verification

```bash
# Ensure no dangling imports
rg "command_context|CommandContextError" src/ tests/

# Import test
uv run python -c "from codeintel.cli.commands import *; print('OK')"

# Full CLI import
uv run python -c "from codeintel.cli import app; print('OK')"
```

### Rollback

Restore deleted files from git if needed.

---

## Phase 8C: Consolidate Runtime Resolution

**Goal:** Single runtime resolution path through `RuntimeResolver`.

### Current State

Runtime resolution happens in FOUR places:

1. **`resolution/runtime.py:RuntimeResolver.resolve()`** - Canonical
2. **`commands/context.py:_resolve_runtime_from_params()`** - Deleted in 8B
3. **`handlers/_lazy_resources.py:lazy_resolve_runtime()`** - Creates mini ExecutionContext
4. **`execution/_lazy_deps.py:lazy_resolve_runtime()`** - For ExecutionContext

### Target State

- `RuntimeResolver.resolve()` is the single implementation
- `handlers/_lazy_resources.py` simplified to directly use RuntimeResolver
- `execution/_lazy_deps.py` deleted (ExecutionContext usage reduced)

### Implementation Steps

#### 8C.1: Simplify handlers/_lazy_resources.py

**File:** `src/codeintel/cli/handlers/_lazy_resources.py`

Current:
```python
def lazy_resolve_runtime(...) -> ResolvedRuntime:
    exec_params: dict[str, object] = dict(params)
    if project_root is not None:
        exec_params["project_root"] = project_root
    if database_path is not None:
        exec_params["db_path"] = database_path
    exec_ctx = ExecutionContext(operation_id=operation_id, params=exec_params)
    return RuntimeResolver.resolve(exec_ctx)
```

Target - inline into `HandlerContext._resolve_runtime()`:
```python
# In handlers/context.py
def _resolve_runtime(self) -> ResolvedRuntime:
    """Resolve runtime using RuntimeResolver."""
    from codeintel.cli.resolution.runtime import resolve_runtime_from_params
    return resolve_runtime_from_params(
        project_root=self.project_root,
        params=self._params,
    )
```

#### 8C.2: Add params-based resolution to RuntimeResolver

**File:** `src/codeintel/cli/resolution/runtime.py`

Add a function that doesn't require `ExecutionContext`:

```python
def resolve_runtime_from_params(
    params: dict[str, object],
    project_root: Path | None = None,
) -> ResolvedRuntime:
    """Resolve runtime from parameter dict.
    
    Convenience function for HandlerContext that doesn't require
    constructing an ExecutionContext.
    """
    # Build minimal context
    effective_params = dict(params)
    if project_root is not None:
        effective_params["project_root"] = project_root
    
    # Use existing resolution logic
    ...
```

#### 8C.3: Delete execution/_lazy_deps.py

**File:** `src/codeintel/cli/execution/_lazy_deps.py`

Delete the file. Update `execution/context.py` to inline or use RuntimeResolver directly.

#### 8C.4: Delete handlers/_lazy_resources.py

**File:** `src/codeintel/cli/handlers/_lazy_resources.py`

Delete the file after inlining into `HandlerContext`.

#### 8C.5: Update ExecutionContext.require_runtime()

Either:
- Inline the resolution logic, OR
- Call `RuntimeResolver.resolve()` directly (will require careful import handling)

### Verification

```bash
# Type check
uv run pyright src/codeintel/cli/resolution/ src/codeintel/cli/handlers/

# Resolution tests
uv run pytest tests/cli/unit/test_resolution_integration.py -v

# Full handler tests
uv run pytest tests/cli/handlers/ -v
```

### Rollback

Restore deleted files, revert RuntimeResolver changes.

---

## Phase 8D: Unify Logging Bootstrap

**Goal:** Single logging bootstrap through `bootstrap_cli()`.

### Current State

```
┌─────────────────────────────────────────────────────────────────┐
│ execution/bootstrap.py:bootstrap_cli()                          │
│ • _BootstrapState class with lock                               │
│ • Thread-safe, idempotent                                       │
│ • Configures logging + signal handlers                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ handlers/_utilities.py:setup_logging()                          │
│ • _LOGGING_CONFIGURED global                                    │
│ • Idempotent via flag                                           │
│ • Only configures logging                                       │
└─────────────────────────────────────────────────────────────────┘
```

Both can be called, potentially with different verbosity settings, causing confusion.

### Target State

- `bootstrap_cli()` is THE bootstrap function
- `setup_logging()` deleted or becomes thin wrapper around `bootstrap_cli()`
- All entry points use `bootstrap_cli()`

### Implementation Steps

#### 8D.1: Audit setup_logging usage

```bash
rg "setup_logging|from codeintel.cli.handlers._utilities import" src/
```

#### 8D.2: Update callers to use bootstrap_cli

For each caller of `setup_logging()`:
- Change to `bootstrap_cli(verbosity=N)`
- Adjust for return value (bootstrap returns CliConfig)

#### 8D.3: Delete setup_logging from _utilities.py

**File:** `src/codeintel/cli/handlers/_utilities.py`

Remove:
- `_LOGGING_CONFIGURED` global
- `setup_logging()` function
- `_determine_log_level()` helper (if only used by setup_logging)

Keep:
- `VERBOSITY_DEBUG`, `VERBOSITY_INFO` constants (move to bootstrap.py if needed)
- `get_handler_logger()`
- `open_handler_gateway()`
- `resolved_to_project_runtime()`

#### 8D.4: Update handlers/__init__.py exports

Remove `setup_logging` from exports if it was exported.

#### 8D.5: Consolidate verbosity constants

Ensure `VERBOSITY_DEBUG` and `VERBOSITY_INFO` are defined in one place (bootstrap.py) and re-exported where needed.

### Verification

```bash
# No more setup_logging references
rg "setup_logging" src/

# Import test
uv run python -c "from codeintel.cli.execution.bootstrap import bootstrap_cli; print('OK')"

# CLI startup test
uv run python -m codeintel.cli --help
```

### Rollback

Restore `setup_logging()` function if callers still need it.

---

## Phase 8E: Decide Executor Future

**Goal:** Either delete the unused executor or create integration path.

### Decision Point

The `OperationExecutor` class (~650 lines) provides:
- Middleware stack execution
- Resilience (retry with circuit breaker)
- Progress tracking
- Async/streaming handler support

BUT: No CLI command uses it. `@cli_command` calls handlers directly.

### Option A: Delete OperationExecutor (RECOMMENDED)

**Rationale:**
- 650+ lines of unused code
- Middleware can be added as decorators if needed later
- Resilience is rarely needed for CLI commands
- Progress tracking can be added to HandlerContext

**Implementation:**

#### 8E.A.1: Delete OperationExecutor class

**File:** `src/codeintel/cli/execution/executor.py`

Delete:
- `class OperationExecutor` (~500 lines)
- `_ExecutorState` class
- `get_executor()`, `configure_executor()` functions
- `run_sync()`, `run_async_operation()` functions

Keep:
- Any utility functions that might be useful elsewhere

#### 8E.A.2: Update execution/__init__.py

Remove exports of deleted components.

#### 8E.A.3: Delete or reduce middleware.py

If middleware is only used by executor, delete the file.
Otherwise, keep for potential future use.

#### 8E.A.4: Simplify resilience modules

The retry/circuit_breaker/resilience modules may become unused.
Either delete them or mark as "utility modules for future use".

### Option B: Integrate Executor with @cli_command

**Rationale:**
- Preserve investment in middleware/resilience infrastructure
- Enable optional middleware for complex operations

**Implementation:**

#### 8E.B.1: Create executor adapter in decorators.py

```python
def _execute_via_executor(
    handler: Callable[[HandlerContext], CliResult[R]],
    ctx: HandlerContext,
    spec: OperationSpec,
) -> CliResult[R]:
    """Execute handler through OperationExecutor for middleware support."""
    executor = get_executor()
    # Create ExecutionContext from HandlerContext
    exec_ctx = ExecutionContext.for_sync(
        spec.operation_id,
        ctx.params,
        ctx.output_format,
    )
    # Execute with middleware
    result = executor.execute(spec, exec_ctx)
    return result.result
```

#### 8E.B.2: Add use_executor flag to CommandConfig

```python
@dataclass(frozen=True)
class CommandConfig:
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    description: str | None = None
    use_executor: bool = False  # NEW: route through executor
```

#### 8E.B.3: Update _execute_command to optionally use executor

```python
def _execute_command(...) -> None:
    # ... existing setup ...
    
    if use_executor:
        result = _execute_via_executor(handler, ctx, spec)
    else:
        result = handler(ctx)  # Direct call
    
    # ... rendering ...
```

### Recommendation

**Choose Option A (Delete)** because:
1. CLI commands are typically short-lived, don't need circuit breakers
2. Retry logic is better handled at the operation level (e.g., network calls)
3. Progress tracking can be added to HandlerContext if needed
4. Simpler codebase is easier to maintain
5. If needed later, middleware can be reimplemented in a simpler form

---

## Implementation Order

```
Phase 8A: Unify OperationSpec
    ↓
Phase 8B: Eliminate command_context  
    ↓
Phase 8C: Consolidate Runtime Resolution
    ↓
Phase 8D: Unify Logging Bootstrap
    ↓
Phase 8E: Delete OperationExecutor (Option A)
```

Each phase is independently deployable and testable.

---

## Verification Checklist

After all phases complete:

- [ ] Single `OperationSpec` class in `execution/registry.py`
- [ ] No `command_context` or `cli/compat.py`
- [ ] Single runtime resolution path through `RuntimeResolver`
- [ ] Single logging bootstrap through `bootstrap_cli()`
- [ ] No unused executor code (Option A) OR integrated executor (Option B)
- [ ] All pyright errors resolved
- [ ] All tests pass
- [ ] CLI commands work correctly

```bash
# Full verification
uv run ruff check src/codeintel/cli/ --fix
uv run pyright src/codeintel/cli/
uv run pytest tests/cli/ -v
uv run python -m codeintel.cli --help
```

---

## Files Summary

### To Delete (~1,290 lines)

| File | Lines | Phase |
|------|-------|-------|
| `commands/context.py` | 360 | 8B |
| `cli/compat.py` | 127 | 8B |
| `execution/_lazy_deps.py` | ~50 | 8C |
| `handlers/_lazy_resources.py` | ~66 | 8C |
| `executor.py` (OperationExecutor) | ~650 | 8E |
| `handlers/_utilities.py` (setup_logging) | ~40 | 8D |

### To Modify

| File | Changes | Phase |
|------|---------|-------|
| `execution/registry.py` | Add optional fields to OperationSpec | 8A |
| `plugins/registry.py` | Fix imports | 8A |
| `execution/__init__.py` | Update exports | 8A, 8E |
| `resolution/runtime.py` | Add params-based resolver | 8C |
| `handlers/context.py` | Inline lazy resolution | 8C |
| `execution/context.py` | Simplify resolution | 8C |
| `handlers/_utilities.py` | Remove setup_logging | 8D |
| `handlers/__init__.py` | Update exports | 8D |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking external plugins | Low | Medium | Version bump, deprecation warnings |
| Circular imports | Medium | High | Careful import ordering, lazy imports |
| Test failures | Medium | Low | Run tests after each phase |
| Performance regression | Low | Low | Profile before/after |

---

## Success Metrics

1. **Code reduction**: ~1,290 lines deleted
2. **Type safety**: 0 pyright errors in CLI
3. **Test coverage**: Maintained or improved
4. **Simplicity**: Single path for each concern
5. **Maintainability**: Clear ownership, no duplication
