# CLI Unified Context - Phase 2 Detailed Implementation Plan

> **Purpose**: Comprehensive, step-by-step implementation plan for migrating all remaining handlers to the unified `ExecutionContext` pattern, simplifying Cyclopts commands, and cleaning up deprecated code.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Handler Migration Strategy](#handler-migration-strategy)
4. [Step 1: Build Handlers Migration](#step-1-build-handlers-migration)
5. [Step 2: Common Handlers Migration](#step-2-common-handlers-migration)
6. [Step 3: Docs Handlers Migration](#step-3-docs-handlers-migration)
7. [Step 4: Datasets Handlers Migration](#step-4-datasets-handlers-migration)
8. [Step 5: Subsystem Handlers Migration](#step-5-subsystem-handlers-migration)
9. [Step 6: History Handlers Migration](#step-6-history-handlers-migration)
10. [Step 7: IDE Handlers Migration](#step-7-ide-handlers-migration)
11. [Step 8: Graphs Handlers Migration](#step-8-graphs-handlers-migration)
12. [Step 9: Command Simplification](#step-9-command-simplification)
13. [Step 10: Cleanup Deprecated Code](#step-10-cleanup-deprecated-code)
14. [Verification Checklist](#verification-checklist)
15. [Rollback Plan](#rollback-plan)

---

## Overview

### Scope

Phase 2 completes the CLI Unified Context architecture by:
1. Migrating all remaining handlers to accept `ExecutionContext`
2. Simplifying Cyclopts commands to use `execute_command()` pattern
3. Removing deprecated code (duplicate `RuntimeCliOptions`, `build_runtime_from_cli`, etc.)

### Phase 1 Foundation (Completed)

Phase 1 established the infrastructure that Phase 2 builds upon:

| Component | Location | Status |
|-----------|----------|--------|
| `RuntimeResolver` | `resolution/runtime.py` | ✅ Complete |
| `GatewayManager` | `resolution/gateway.py` | ✅ Complete |
| `ResolvedRuntime` | `resolution/types.py` | ✅ Complete |
| `ResolutionError` | `resolution/errors.py` | ✅ Complete |
| `CommonOptions` | `options/common.py` | ✅ Complete |
| `ExecutionContext.require_runtime()` | `execution/context.py` | ✅ Complete |
| `ExecutionContext.require_gateway()` | `execution/context.py` | ✅ Complete |
| Proof of Concept Handlers | `storage_handlers.py`, `ops_handlers.py` | ✅ Complete |

### Phase 2 Effort Estimate

| Step | Files | Handlers | Estimated Effort |
|------|-------|----------|------------------|
| Build Handlers | 1 file | ~8 handlers | 2-3 hours |
| Common Handlers | 1 file | ~6 handlers | 2-3 hours |
| Docs Handlers | 1 file | ~10 handlers | 3-4 hours |
| Datasets Handlers | 1 file | ~8 handlers | 2-3 hours |
| Subsystem Handlers | 1 file | ~4 handlers | 1-2 hours |
| History Handlers | 1 file | ~3 handlers | 1-2 hours |
| IDE Handlers | 1 file | ~2 handlers | 1 hour |
| Graphs Handlers | 1 file | ~3 handlers | 1-2 hours |
| Command Simplification | ~18 files | ~40 commands | 4-6 hours |
| Cleanup | ~15 files | - | 2-3 hours |
| **Total** | **~25 files** | **~44 handlers** | **19-29 hours** |

### Success Criteria

- [ ] All handlers accept `ExecutionContext` as their context parameter
- [ ] All handlers return `CliResult[T]` instead of `None`
- [ ] All Cyclopts commands use `execute_command()` or `CycloptsAdapter`
- [ ] No duplicate `RuntimeCliOptions` definitions remain
- [ ] No duplicate `build_runtime_from_cli` implementations remain
- [ ] All deprecated context types removed
- [ ] Zero pyright/pyrefly/ruff errors
- [ ] All CLI tests pass

---

## Prerequisites

Before starting Phase 2, verify Phase 1 infrastructure is complete and working:

```bash
# Verify Phase 1 infrastructure
uv run python -c "
from codeintel.cli.resolution import resolve_runtime, ResolvedRuntime, ResolutionError, GatewayManager
from codeintel.cli.options import CommonOptions
from codeintel.cli.execution.context import ExecutionContext
print('Phase 1 infrastructure: OK')
"

# Verify all CLI tests pass
uv run pytest tests/cli/ -q

# Verify quality checks pass
uv run ruff check src/codeintel/cli/
uv run pyright src/codeintel/cli/
uv run pyrefly check src/codeintel/cli/
```

---

## Handler Migration Strategy

### Universal Migration Pattern

Every handler follows the same transformation pattern:

**Before (Legacy Pattern)**:
```python
def some_handler(
    options: SomeOptions,
    ctx: SomeContext,
) -> None:
    """Handler docstring."""
    runtime = build_runtime_from_cli(ctx.runtime_options)
    gateway = open_gateway(StorageConfig.for_readonly(runtime.db_path))
    
    target = options.target
    verbose = ctx.verbose
    
    # ... business logic ...
    
    if output_format == OutputFormat.JSON:
        print(json.dumps(result))
    else:
        print(result)
```

**After (ExecutionContext Pattern)**:
```python
def some_handler(ctx: ExecutionContext) -> CliResult[SomeResult]:
    """Handler docstring.

    Parameters
    ----------
    ctx
        Execution context with params:
        - target: The target to process
        - verbose: Verbosity level

    Returns
    -------
    CliResult[SomeResult]
        Handler result.
    """
    runtime = ctx.require_runtime()
    gateway = ctx.require_gateway()
    
    target = ctx.get_str_param("target")
    
    # ... business logic (unchanged) ...
    
    return CliResult.ok(SomeResult(
        # ... result fields ...
    ))
```

### Key Transformations

| Before | After |
|--------|-------|
| `build_runtime_from_cli(opts)` | `ctx.require_runtime()` |
| `open_gateway(config)` | `ctx.require_gateway()` |
| `options.field` | `ctx.get_str_param("field")` / `ctx.params.get("field")` |
| `ctx.verbose` | `ctx.verbosity` |
| `ctx.output_format` | `ctx.output_format` (same) |
| `return None` | `return CliResult.ok(result)` |
| Direct `print()` | Return `CliResult`, let renderer handle output |

### Result Type Creation

For each handler, create a corresponding result dataclass:

```python
@dataclass
class SomeResult:
    """Result from some_handler.

    Parameters
    ----------
    field1
        Description of field1.
    field2
        Description of field2.
    """

    field1: str
    field2: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "field1": self.field1,
            "field2": self.field2,
        }
```

### Preserving Legacy Handlers

During migration, preserve legacy handlers for backward compatibility:

```python
# Legacy handler (deprecated, will be removed in cleanup phase)
def some_handler_legacy(
    options: SomeOptions,
    ctx: SomeContext,
) -> None:
    """Deprecated: Use some_handler instead."""
    exec_ctx = _build_execution_context_from_legacy(ctx)
    result = some_handler(exec_ctx)
    _render_result(result, ctx.output_format)


# New handler (preferred)
def some_handler(ctx: ExecutionContext) -> CliResult[SomeResult]:
    """Handler docstring."""
    ...
```

---

## Step 1: Build Handlers Migration

### Target File

`src/codeintel/cli/build_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `build_run_handler` | `(opts: BuildRunOptions, ctx: BuildRunContext)` | High |
| `build_status_handler` | `(opts, ctx)` | Medium |
| `build_clean_handler` | `(opts, ctx)` | Low |
| `build_info_handler` | `(ctx)` | Low |
| `build_targets_handler` | `(ctx)` | Low |
| `build_config_handler` | `(opts, ctx)` | Medium |
| `build_validate_handler` | `(opts, ctx)` | Medium |
| `build_watch_handler` | `(opts, ctx)` | High |

### Result Types to Create

```python
@dataclass
class BuildRunResult:
    """Result from build run."""
    success: bool
    targets_built: list[str]
    duration_seconds: float
    errors: list[str]

@dataclass
class BuildStatusResult:
    """Result from build status."""
    is_up_to_date: bool
    last_build: str | None
    pending_targets: list[str]

@dataclass
class BuildInfoResult:
    """Result from build info."""
    version: str
    build_dir: str
    db_path: str
    targets_available: list[str]
```

### Migration Example: build_run_handler

**Before**:
```python
def build_run_handler(
    options: BuildRunOptions,
    ctx: BuildRunContext,
) -> None:
    """Run the build pipeline."""
    runtime_options = ctx.runtime_options or RuntimeCliOptions()
    runtime = build_runtime_from_cli(runtime_options)
    targets = options.targets or []
    
    # ... build logic ...
```

**After**:
```python
def build_run_handler(ctx: ExecutionContext) -> CliResult[BuildRunResult]:
    """Run the build pipeline.

    Parameters
    ----------
    ctx
        Execution context with params:
        - targets: List of build targets (optional)
        - parallel: Enable parallel builds
        - force: Force rebuild all targets

    Returns
    -------
    CliResult[BuildRunResult]
        Build result with success status and details.

    Raises
    ------
    RuntimeError
        If build fails critically.
    """
    runtime = ctx.require_runtime()
    
    targets_raw = ctx.params.get("targets")
    targets: list[str] = list(targets_raw) if targets_raw else []
    parallel = ctx.get_bool_param("parallel", default=False)
    force = ctx.get_bool_param("force", default=False)
    
    # ... build logic (unchanged) ...
    
    return CliResult.ok(BuildRunResult(
        success=True,
        targets_built=built_targets,
        duration_seconds=elapsed,
        errors=[],
    ))
```

### Verification Checkpoint

```bash
# After completing Step 1
uv run ruff check src/codeintel/cli/build_handlers.py --fix
uv run pyright src/codeintel/cli/build_handlers.py
uv run pytest tests/cli/ -k "build" -v
```

---

## Step 2: Common Handlers Migration

### Target File

`src/codeintel/cli/common_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `project_info_handler` | `(project_root, output_format)` | Low |
| `project_init_handler` | `(repo, commit, ...)` | Medium |
| `version_handler` | `()` | Low |
| `config_show_handler` | `(output_format)` | Low |
| `config_validate_handler` | `()` | Low |
| `health_check_handler` | `(project_root)` | Medium |

### Key Removal: Duplicate `build_runtime_from_cli`

This file contains one of the duplicate `build_runtime_from_cli` implementations. After migration, this function should be marked deprecated and eventually removed.

```python
# DEPRECATED - Remove after all handlers migrated
def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Deprecated: Use ctx.require_runtime() instead."""
    import warnings
    warnings.warn(
        "build_runtime_from_cli is deprecated. Use ctx.require_runtime() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation ...
```

### Verification Checkpoint

```bash
# After completing Step 2
uv run ruff check src/codeintel/cli/common_handlers.py --fix
uv run pyright src/codeintel/cli/common_handlers.py
uv run pytest tests/cli/ -k "common or project or config" -v
```

---

## Step 3: Docs Handlers Migration

### Target File

`src/codeintel/cli/docs_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `docs_build_handler` | `(opts, ctx)` | High |
| `docs_serve_handler` | `(opts, ctx)` | Medium |
| `docs_clean_handler` | `(opts, ctx)` | Low |
| `docs_status_handler` | `(opts, ctx)` | Low |
| `docs_index_handler` | `(opts, ctx)` | Medium |
| `docs_search_handler` | `(query, opts, ctx)` | Medium |
| `docs_export_handler` | `(format, opts, ctx)` | Medium |
| `docs_validate_handler` | `(opts, ctx)` | Medium |
| `docs_list_handler` | `(opts, ctx)` | Low |
| `docs_preview_handler` | `(target, opts, ctx)` | Medium |

### Known Issue: OutputFormat Shadow

This file defines its own `OutputFormat` enum that shadows `cli_types.OutputFormat`. Fix during migration:

```python
# REMOVE this local definition
# class OutputFormat(str, Enum):
#     TEXT = "text"
#     JSON = "json"

# USE the canonical import
from codeintel.cli.cli_types import OutputFormat
```

### Verification Checkpoint

```bash
# After completing Step 3
uv run ruff check src/codeintel/cli/docs_handlers.py --fix
uv run pyright src/codeintel/cli/docs_handlers.py
uv run pytest tests/cli/ -k "docs" -v
```

---

## Step 4: Datasets Handlers Migration

### Target File

`src/codeintel/cli/datasets_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `dataset_list_handler` | Already has structured version in ops_handlers | ✅ Done |
| `dataset_describe_handler` | Already has structured version in ops_handlers | ✅ Done |
| `dataset_verify_handler` | Already has structured version in ops_handlers | ✅ Done |
| `dataset_export_handler` | `(table_key, format, opts, ctx)` | Medium |
| `dataset_import_handler` | `(source, table_key, opts, ctx)` | Medium |
| `dataset_schema_handler` | `(table_key, opts, ctx)` | Low |
| `dataset_sample_handler` | `(table_key, count, opts, ctx)` | Low |
| `dataset_stats_handler` | `(table_key, opts, ctx)` | Medium |

### Key Removal: Duplicate `RuntimeCliOptions`

This file has its own `RuntimeCliOptions` definition. After migration, remove it:

```python
# REMOVE - Use CommonOptions or ctx.params instead
# @dataclass
# class RuntimeCliOptions:
#     project_root: Path | None = None
#     repo: str | None = None
#     ...
```

### Verification Checkpoint

```bash
# After completing Step 4
uv run ruff check src/codeintel/cli/datasets_handlers.py --fix
uv run pyright src/codeintel/cli/datasets_handlers.py
uv run pytest tests/cli/ -k "dataset" -v
```

---

## Step 5: Subsystem Handlers Migration

### Target File

`src/codeintel/cli/subsystem_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `subsystem_list_handler` | `(options: SubsystemListOptions)` | Low |
| `subsystem_show_handler` | `(subsystem_id, options)` | Low |
| `subsystem_profile_handler` | `(subsystem_id, options)` | Medium |
| `subsystem_coverage_handler` | `(subsystem_id, options)` | Medium |

### Key Removal: Minimal `RuntimeCliOptions`

This file has a minimal `RuntimeCliOptions` with only `project_root`. Remove after migration:

```python
# REMOVE - Use ctx.params instead
# @dataclass
# class RuntimeCliOptions:
#     project_root: Path | None = None
```

### Verification Checkpoint

```bash
# After completing Step 5
uv run ruff check src/codeintel/cli/subsystem_handlers.py --fix
uv run pyright src/codeintel/cli/subsystem_handlers.py
uv run pytest tests/cli/ -k "subsystem" -v
```

---

## Step 6: History Handlers Migration

### Target File

`src/codeintel/cli/history_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `history_list_handler` | `(opts, ctx)` | Low |
| `history_show_handler` | `(entry_id, opts, ctx)` | Low |
| `history_clear_handler` | `(opts, ctx)` | Low |

### Verification Checkpoint

```bash
# After completing Step 6
uv run ruff check src/codeintel/cli/history_handlers.py --fix
uv run pyright src/codeintel/cli/history_handlers.py
uv run pytest tests/cli/ -k "history" -v
```

---

## Step 7: IDE Handlers Migration

### Target File

`src/codeintel/cli/ide_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `ide_symbols_handler` | `(query, opts, ctx)` | Medium |
| `ide_goto_handler` | `(symbol, opts, ctx)` | Medium |

### Key Removal: Minimal `RuntimeCliOptions`

Same pattern as subsystem_handlers - has minimal `RuntimeCliOptions`.

### Verification Checkpoint

```bash
# After completing Step 7
uv run ruff check src/codeintel/cli/ide_handlers.py --fix
uv run pyright src/codeintel/cli/ide_handlers.py
uv run pytest tests/cli/ -k "ide" -v
```

---

## Step 8: Graphs Handlers Migration

### Target File

`src/codeintel/cli/graphs_handlers.py`

### Handlers to Migrate

| Handler | Current Signature | Complexity |
|---------|-------------------|------------|
| `graph_info_handler` | `(opts, ctx)` | Low |
| `graph_export_handler` | `(format, opts, ctx)` | Medium |
| `graph_analyze_handler` | `(opts, ctx)` | High |

### Verification Checkpoint

```bash
# After completing Step 8
uv run ruff check src/codeintel/cli/graphs_handlers.py --fix
uv run pyright src/codeintel/cli/graphs_handlers.py
uv run pytest tests/cli/ -k "graph" -v
```

---

## Step 9: Command Simplification

### Target Files

All `cyclopts_*.py` files:
- `cyclopts_build.py`
- `cyclopts_docs.py`
- `cyclopts_datasets.py`
- `cyclopts_subsystem.py`
- `cyclopts_history.py`
- `cyclopts_ide.py`
- `cyclopts_graphs.py`
- `cyclopts_ops.py`
- `cyclopts_storage.py`
- `cyclopts_common.py`
- `cyclopts_serve.py`
- And others

### Command Simplification Pattern

**Before**:
```python
@build_app.command(name="run")
@dataclass
class BuildRunCli:
    """Run the build pipeline."""

    targets: list[str] | None = None
    runtime: Annotated[RuntimeCLI | None, Parameter(name="*")] = None
    output: Annotated[OutputFormatCLI | None, Parameter(name="*")] = None
    parallel: bool = False
    force: bool = False

    def __call__(self) -> None:
        runtime_opts, verbose, output_format = make_handler_context(
            self.runtime or RuntimeCLI(),
            self.output or OutputFormatCLI(),
            default_output=OutputFormat.TEXT,
        )
        options = BuildRunOptions(
            targets=self.targets,
            parallel=self.parallel,
            force=self.force,
        )
        ctx_opts = BuildRunContext(
            runtime_options=runtime_opts,
            verbose=verbose,
            output_format=output_format,
        )
        run_handler(build_run_handler, options, ctx_opts)
```

**After**:
```python
@build_app.command(name="run")
@dataclass
class BuildRunCli:
    """Run the build pipeline."""

    targets: Annotated[
        list[str] | None,
        Parameter(help="Build targets to run."),
    ] = None
    parallel: Annotated[
        bool,
        Parameter(name="--parallel", help="Enable parallel builds."),
    ] = False
    force: Annotated[
        bool,
        Parameter(name="--force", help="Force rebuild all targets."),
    ] = False
    options: Annotated[CommonOptions, Parameter(name="*")] = field(
        default_factory=CommonOptions
    )

    def __call__(self) -> None:
        CycloptsAdapter("build.run", build_run_handler)(self)
```

### Migration Steps per Command

1. **Replace option bundles**: `RuntimeCLI` + `OutputFormatCLI` → `CommonOptions`
2. **Remove intermediate context creation**: Delete `make_handler_context()` call
3. **Remove intermediate options creation**: Delete `SomeOptions(...)` instantiation
4. **Delegate to adapter**: Replace `run_handler(...)` with `CycloptsAdapter(...)(self)`

### Verification Checkpoint

```bash
# After completing Step 9
uv run ruff check src/codeintel/cli/cyclopts_*.py --fix
uv run pyright src/codeintel/cli/cyclopts_*.py
uv run pytest tests/cli/ -v --tb=short
```

---

## Step 10: Cleanup Deprecated Code

### Code to Remove

#### 1. Duplicate `RuntimeCliOptions` Classes

| File | Status |
|------|--------|
| `common_handlers.py` | Remove alias to `RuntimeOptions` |
| `datasets_handlers.py` | Remove full class definition |
| `subsystem_handlers.py` | Remove minimal class definition |
| `ide_handlers.py` | Remove minimal class definition |
| `build_handlers.py` | Remove class definition if present |

#### 2. Duplicate `build_runtime_from_cli` Functions

| File | Status |
|------|--------|
| `cyclopts_common.py` | Remove (use `resolve_runtime`) |
| `common_handlers.py` | Remove (use `resolve_runtime`) |
| `datasets_handlers.py` | Remove (use `resolve_runtime`) |
| `subsystem_handlers.py` | Remove (use `resolve_runtime`) |
| `ide_handlers.py` | Remove (use `resolve_runtime`) |
| `build_handlers.py` | Remove (use `resolve_runtime`) |
| `ops_handlers.py` | Remove `_build_runtime_or_error` |

#### 3. Old Context Types

| File | Type to Remove |
|------|----------------|
| `build_handlers.py` | `BuildRunContext`, `BuildRunOptions` |
| `handlers/base.py` | `HandlerContext` (replaced by `ExecutionContext`) |
| Various | Any handler-specific context types |

#### 4. Deprecated Helper Functions

| File | Function |
|------|----------|
| `cyclopts_common.py` | `make_handler_context()` |
| `cyclopts_common.py` | `runtime_cli_to_options()` |
| `cyclopts_common.py` | `get_verbose()` |
| `cyclopts_common.py` | `get_output_format()` |

### Cleanup Procedure

1. **Identify usages**: `grep` for each deprecated symbol
2. **Verify no remaining usages**: All handlers should use `ExecutionContext`
3. **Remove code**: Delete deprecated classes/functions
4. **Update `__all__`**: Remove from exports
5. **Run full test suite**: Verify nothing breaks

### Verification Checkpoint

```bash
# After completing Step 10
# Verify no deprecated symbols remain
grep -r "RuntimeCliOptions" src/codeintel/cli/ --include="*.py" | grep -v "# DEPRECATED"
grep -r "build_runtime_from_cli" src/codeintel/cli/ --include="*.py" | grep -v "# DEPRECATED"
grep -r "make_handler_context" src/codeintel/cli/ --include="*.py"
grep -r "BuildRunContext" src/codeintel/cli/ --include="*.py"

# Full quality checks
uv run ruff check src/codeintel/cli/
uv run pyright src/codeintel/cli/
uv run pyrefly check src/codeintel/cli/

# Full test suite
uv run pytest tests/cli/ -q
```

---

## Verification Checklist

### Per-Handler Checklist

For each handler migrated:

- [ ] Signature changed to `(ctx: ExecutionContext) -> CliResult[T]`
- [ ] Uses `ctx.require_runtime()` instead of `build_runtime_from_cli()`
- [ ] Uses `ctx.require_gateway()` instead of direct `open_gateway()`
- [ ] Uses `ctx.get_*_param()` for parameter access
- [ ] Returns `CliResult.ok(result)` instead of `None`
- [ ] Has corresponding result dataclass with `to_dict()` method
- [ ] Docstring updated with Parameters and Returns sections
- [ ] All raised exceptions documented

### Per-Command Checklist

For each Cyclopts command simplified:

- [ ] Uses `CommonOptions` instead of `RuntimeCLI` + `OutputFormatCLI`
- [ ] Uses `CycloptsAdapter("operation.id", handler)(self)` pattern
- [ ] No intermediate context/options creation
- [ ] Works correctly with `--help`
- [ ] Works correctly with `--json` flag

### Final Verification

- [ ] `uv run ruff check src/codeintel/cli/` passes
- [ ] `uv run pyright src/codeintel/cli/` passes
- [ ] `uv run pyrefly check src/codeintel/cli/` passes
- [ ] `uv run pytest tests/cli/ -q` passes (all tests)
- [ ] No duplicate `RuntimeCliOptions` definitions
- [ ] No duplicate `build_runtime_from_cli` implementations
- [ ] No references to `make_handler_context`
- [ ] No references to old context types (`BuildRunContext`, etc.)

---

## Rollback Plan

### Level 1: Individual Handler Rollback

If a single handler migration causes issues:
- Revert only that handler file
- Keep legacy handler available
- Commands can use either handler

### Level 2: Handler Group Rollback

If an entire handler group (e.g., build_handlers) has issues:
- Revert the handler file
- Revert corresponding cyclopts_*.py changes
- Other handler groups continue working

### Level 3: Full Phase 2 Rollback

If fundamental issues are discovered:
```bash
# Revert all Phase 2 changes
git revert HEAD~N  # Where N is the number of Phase 2 commits
```

Phase 1 infrastructure remains intact and functional.

---

## Incremental Implementation Strategy

### Recommended Order

1. **Start with simplest handlers** (history, ide) to build confidence
2. **Progress to medium complexity** (subsystem, graphs, storage)
3. **Handle high complexity last** (build, docs, datasets)
4. **Simplify commands in parallel** as handlers are migrated
5. **Cleanup as final step** after all migrations complete

### PR Strategy

| PR | Scope | Dependencies |
|----|-------|--------------|
| PR 1 | History + Subsystem handlers | None |
| PR 2 | IDE + Graphs handlers | None |
| PR 3 | Build handlers | None |
| PR 4 | Docs handlers | None |
| PR 5 | Datasets handlers | None |
| PR 6 | Common handlers | None |
| PR 7 | Command simplification (all) | PRs 1-6 |
| PR 8 | Cleanup deprecated code | PR 7 |

Each PR should:
- Be independently testable
- Not break existing functionality
- Include updated tests
- Pass all quality checks

---

## Appendix A: Result Type Templates

### Standard Result Type

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SomeOperationResult:
    """Result from some operation.

    Parameters
    ----------
    status
        Operation status (success, failed, skipped).
    items_processed
        Number of items processed.
    errors
        List of error messages if any.
    metadata
        Additional operation metadata.
    """

    status: str
    items_processed: int
    errors: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "status": self.status,
            "items_processed": self.items_processed,
            "errors": self.errors,
            "metadata": self.metadata,
        }
```

### List Result Type

```python
@dataclass
class ItemListResult:
    """Result containing a list of items.

    Parameters
    ----------
    items
        List of item dictionaries.
    count
        Total number of items.
    """

    items: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "items": self.items,
            "count": self.count,
        }
```

---

## Appendix B: Common Migration Issues

### Issue 1: Handler Uses Global State

**Symptom**: Handler accesses module-level variables or singletons

**Solution**: Pass through `ctx.params` or use `ctx.config`

```python
# Before
def handler():
    config = get_global_config()  # Bad

# After
def handler(ctx: ExecutionContext):
    config = ctx.config  # Good - from context
```

### Issue 2: Handler Returns Multiple Types

**Symptom**: Handler has multiple return paths with different types

**Solution**: Create union result type or use `CliResult.fail()` for errors

```python
# After
def handler(ctx: ExecutionContext) -> CliResult[SuccessResult]:
    if error_condition:
        return CliResult.fail(ProblemDetail(
            type="error:type",
            title="Error Title",
            detail="Error details",
            status=400,
        ))
    return CliResult.ok(SuccessResult(...))
```

### Issue 3: Handler Needs Different Gateway Modes

**Symptom**: Handler needs read-write gateway for some operations, read-only for others

**Solution**: Call `require_gateway()` with appropriate `read_only` parameter per operation

```python
def handler(ctx: ExecutionContext) -> CliResult[SomeResult]:
    if ctx.get_bool_param("write_mode", default=False):
        gateway = ctx.require_gateway(read_only=False)
    else:
        gateway = ctx.require_gateway(read_only=True)
```

### Issue 4: Handler Has Complex Parameter Types

**Symptom**: Handler expects lists, dicts, or custom types from params

**Solution**: Access raw `ctx.params` and perform type conversion

```python
def handler(ctx: ExecutionContext) -> CliResult[SomeResult]:
    # For list parameters
    targets_raw = ctx.params.get("targets")
    targets: list[str] = list(targets_raw) if targets_raw else []
    
    # For enum parameters
    mode_str = ctx.get_str_param("mode", "default")
    mode = SomeEnum(mode_str)
    
    # For path parameters
    path_str = ctx.get_str_param("path")
    path = Path(path_str) if path_str else None
```

---

*Document Version: 1.0*
*Created: 2025-01-09*
*Depends on: Phase 1 Implementation (cli-unified-context-phase1-implementation.md)*

