# Phase 5: Command Decorator & Migration — Detailed Implementation Plan

> **Phase:** 5 of 6  
> **Duration:** 5-7 days  
> **Risk Level:** Medium  
> **Dependencies:** Phase 4 complete ✅  
> **Parallelizable:** Yes (per command file)  
> **Last Updated:** December 2024 (Post-Phase 4)  

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Decorator Design](#3-decorator-design)
4. [Migration Pattern](#4-migration-pattern)
5. [Command Migration Schedule](#5-command-migration-schedule)
6. [Detailed Tasks](#6-detailed-tasks)
7. [File Changes](#7-file-changes)
8. [Testing Requirements](#8-testing-requirements)
9. [Verification Checklist](#9-verification-checklist)
10. [Exit Criteria](#10-exit-criteria)
11. [Rollback Procedure](#11-rollback-procedure)

---

## 1. Objectives

Phase 5 creates the `@cli_command` decorator and migrates all commands:

1. **Create `@cli_command` decorator** — Declarative command binding
2. **Migrate all command files** — Remove boilerplate `__call__` methods
3. **Auto-register operations** — Decorator registers with registry
4. **Eliminate manual wiring** — No more RuntimeCLI/OutputFormatCLI handling

---

## 2. Prerequisites

### 2.1 Phase Dependencies

- [x] Phase 4 complete (registry unified) ✅
- [x] All handlers migrated (Phase 3) ✅
- [x] Handler signatures finalized: `(ctx: HandlerContext) -> CliResult[T]`

### 2.2 Environment

- [ ] All existing tests passing
- [ ] Clean git working tree

### 2.3 Phase 3 & 4 Context (Informing Phase 5)

All handlers now use the unified `HandlerContext` with typed accessors:

```python
# Current handler pattern (Phase 3 complete)
from codeintel.cli.handlers.context import HandlerContext

def my_handler(ctx: HandlerContext) -> CliResult[MyResult]:
    name = ctx.param_str("name")           # Optional param
    count = ctx.param_int("count", 10)     # With default
    target = ctx.require_str("target")     # Required (raises ParameterError)
    # ... business logic
```

**Phase 4 Outcome:** Two registries now exist:
- **NEW registry** (`execution/registry.py`): For handler-based operations with new `OperationSpec`
- **LEGACY registry** (`introspection/registry.py`): Backward compat for existing tests

The `@cli_command` decorator **MUST** register with the NEW registry in `execution/registry.py`, using the new `OperationSpec` type with `group`, `require_runtime`, etc. fields.

```python
# @cli_command imports from execution/registry.py (NEW registry)
from codeintel.cli.execution.registry import OperationSpec, register_operation
```

The decorator will create `HandlerContext` with `_params` populated from command dataclass fields.

---

## 2.4 Phase 4 Outcome Summary

Phase 4 created a dual registry architecture:

| Registry | Location | OperationSpec Type | Populated By |
|----------|----------|-------------------|--------------|
| **NEW** | `execution/registry.py` | `group`, `require_runtime`, `require_gateway`, etc. | `handlers/*.py` module-level registrations |
| **LEGACY** | `introspection/registry.py` | `category` (enum), `param_schema`, etc. | `operations/*.py` module-level registrations |

**Import paths:**
```python
# NEW registry (for @cli_command and handlers)
from codeintel.cli.execution.registry import OperationSpec, register_operation, get_registry

# LEGACY registry (for backward-compatible tests)
from codeintel.cli.introspection.registry import register_operation as legacy_register
from codeintel.cli.introspection import get_operation_registry  # Returns LEGACY registry
```

**Key point for Phase 5:** The `@cli_command` decorator MUST use the NEW registry imports.

---

## 3. Decorator Design

### 3.1 Decorator Signature

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
    """Decorator for CLI command dataclasses."""
```

### 3.2 Generated `__call__` Behavior

The decorator generates a `__call__` method that:

1. Calls `bootstrap_cli()` with verbosity
2. Extracts parameters from dataclass fields
3. Resolves runtime paths from command fields
4. Creates `HandlerContext`
5. Invokes handler within context manager
6. Renders result via `UnifiedRenderer`
7. Handles exit codes

### 3.3 Parameter Extraction

The decorator introspects the dataclass to build the params dict:

```python
def _extract_params(command_instance: Any) -> dict[str, Any]:
    """Extract parameters from command dataclass fields."""
    params = {}
    for field_info in dataclasses.fields(command_instance):
        name = field_info.name
        # Skip standard fields
        if name in ("output_format", "verbose", "json"):
            continue
        value = getattr(command_instance, name)
        params[name] = value
    return params
```

### 3.4 Auto-Registration (NEW Registry)

When the decorator is applied, it registers the operation with the **NEW registry** (`execution/registry.py`):

```python
# During decoration - imports from NEW registry
from codeintel.cli.execution.registry import OperationSpec, register_operation

register_operation(OperationSpec(
    operation_id=operation_id,
    name=cls.__name__,
    description=description or cls.__doc__ or "",
    handler=handler,
    group=operation_id.split(".")[0],
    require_runtime=require_runtime,
    require_gateway=require_gateway,
    require_graph_runtime=require_graph_runtime,
))
```

**Important:** The decorator uses the NEW `OperationSpec` from `execution/registry.py`, NOT the legacy spec from `execution/executor.py`. The key differences:
- NEW: `group: str` (CLI command group name)
- NEW: `require_runtime`, `require_gateway`, `require_graph_runtime` (booleans)
- LEGACY: `category: OperationCategory` (enum)
- LEGACY: `param_schema`, `requires_progress` (different fields)

---

## 4. Migration Pattern

### 4.1 Before Migration

```python
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT

    def __call__(self) -> None:
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)
        params = {"status": self.status, "limit": self.limit}
        
        with command_context(
            "jobs.list",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)
```

### 4.2 After Migration

```python
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
    
    # NO __call__ method - decorator generates it
```

### 4.3 Key Changes

1. **Add `@cli_command` decorator** (above `@app.command`)
2. **Add `verbose` field** (if not present)
3. **Remove `__call__` method entirely**
4. **Remove imports** of `RuntimeCLI`, `OutputFormatCLI`, `command_context`

---

## 5. Command Migration Schedule

### 5.1 Migration Order

| Day | Priority | Files | Notes |
|-----|----------|-------|-------|
| 1-2 | 1 | `jobs.py` | POC - simplest |
| 2 | 2 | `health.py` | Simple, no runtime |
| 2-3 | 3 | `ops.py` | Uses runtime |
| 3 | 4 | `storage.py` | |
| 3-4 | 5 | `history.py` | |
| 4 | 6 | `build.py` | Complex |
| 4-5 | 7 | `graphs.py` | Uses graph_runtime |
| 5 | 8 | `docs.py` | |
| 5-6 | 9 | `ide.py` | |
| 6 | 10 | `datasets.py` | |
| 6 | 11 | `dataset_ops.py` | |
| 6-7 | 12 | `plugins.py` | |
| 7 | 13 | `subsystem.py` | |
| 7 | 14 | `serve.py` | |
| 7 | 15 | `config.py` | |
| 7 | 16 | `completions.py` | |

### 5.2 Command File Details

| File | Commands | Complexity |
|------|----------|------------|
| `jobs.py` | 5 | Low |
| `health.py` | 1 | Low |
| `ops.py` | Multiple | Medium |
| `storage.py` | Multiple | Medium |
| `history.py` | Multiple | Medium |
| `build.py` | Multiple | High |
| `graphs.py` | Multiple | High |
| `docs.py` | Multiple | Medium |
| `ide.py` | Multiple | Medium |
| `datasets.py` | Multiple | Medium |
| `dataset_ops.py` | Multiple | Medium |
| `plugins.py` | Multiple | Medium |
| `subsystem.py` | Multiple | Medium |
| `serve.py` | Multiple | Medium |
| `config.py` | Multiple | Low |
| `completions.py` | Multiple | Low |

---

## 6. Detailed Tasks

### Task P5-1: Create `commands/decorators.py`

**Duration:** 8 hours

**File:** `src/codeintel/cli/commands/decorators.py`

```python
"""Declarative command binding via @cli_command decorator.

This module provides the @cli_command decorator that eliminates boilerplate
from CLI command classes. Instead of manually implementing __call__, the
decorator generates it based on:

- Handler function to invoke
- Resource requirements
- Command dataclass fields

Note: This decorator registers with the NEW registry in execution/registry.py,
NOT the legacy registry in introspection/registry.py.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from codeintel.cli.execution.bootstrap import bootstrap_cli
# IMPORTANT: Import from NEW registry (execution/registry.py), not introspection
from codeintel.cli.execution.registry import OperationSpec, register_operation
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.service import get_renderer
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.core import CliResult

T = TypeVar("T")

# Fields that are standard infrastructure, not command params
_INFRASTRUCTURE_FIELDS = frozenset({
    "output_format",
    "verbose",
    "json",
    "project",
    "project_root",
    "db_path",
    "database_path",
    "repo",
    "commit",
    "build_dir",
    "repo_root",
})


def cli_command(
    operation_id: str,
    *,
    handler: Callable[[HandlerContext], CliResult[Any]],
    require_runtime: bool = True,
    require_gateway: bool = True,
    require_graph_runtime: bool = False,
    description: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator for CLI command dataclasses.

    Generates a __call__ method that handles all CLI infrastructure:

    1. Bootstrap CLI (logging, config)
    2. Extract parameters from dataclass fields
    3. Create HandlerContext
    4. Invoke handler
    5. Render result
    6. Handle exit code

    Also registers the operation with the global OperationRegistry.

    Parameters
    ----------
    operation_id
        Unique operation identifier (e.g., "jobs.list").
    handler
        Handler function to invoke.
    require_runtime
        Whether handler needs ResolvedRuntime.
    require_gateway
        Whether handler needs StorageGateway.
    require_graph_runtime
        Whether handler needs GraphRuntime.
    description
        Optional description (defaults to class docstring).

    Returns
    -------
    Callable[[type[T]], type[T]]
        Class decorator.

    Examples
    --------
    >>> @cli_command("jobs.list", handler=jobs_list_handler, require_runtime=False)
    ... @jobs_app.command(name="list")
    ... @dataclass
    ... class JobsListCommand:
    ...     '''List background jobs.'''
    ...     status: str | None = None
    ...     limit: int = 20
    ...     output_format: OutputFormat = OutputFormat.TEXT
    ...     verbose: int = 0
    """

    def decorator(cls: type[T]) -> type[T]:
        # Extract description from docstring if not provided
        op_description = description or cls.__doc__ or f"Execute {operation_id}"

        # Register operation
        register_operation(OperationSpec(
            operation_id=operation_id,
            name=cls.__name__,
            description=op_description.strip(),
            handler=handler,
            group=operation_id.split(".")[0],
            require_runtime=require_runtime,
            require_gateway=require_gateway,
            require_graph_runtime=require_graph_runtime,
        ))

        # Generate __call__ method
        def __call__(self: Any) -> None:
            _execute_command(
                command=self,
                operation_id=operation_id,
                handler=handler,
                require_runtime=require_runtime,
            )

        # Attach to class
        cls.__call__ = __call__

        return cls

    return decorator


def _execute_command(
    command: Any,
    operation_id: str,
    handler: Callable[[HandlerContext], CliResult[Any]],
    require_runtime: bool,
) -> None:
    """Execute a CLI command.

    Parameters
    ----------
    command
        Command dataclass instance.
    operation_id
        Operation identifier.
    handler
        Handler function.
    require_runtime
        Whether runtime is required.
    """
    # Extract verbosity
    verbosity = getattr(command, "verbose", 0)

    # Bootstrap CLI
    config = bootstrap_cli(verbosity=verbosity)

    # Extract output format
    output_format = _get_output_format(command)

    # Extract parameters
    params = _extract_params(command)

    # Extract runtime paths
    project_root = _get_path_field(command, "project", "project_root")
    database_path = _get_path_field(command, "db_path", "database_path")

    # Create context (params become _params for typed accessor methods)
    ctx = HandlerContext(
        config=config,
        operation_id=operation_id,
        output_format=output_format,
        verbosity=verbosity,
        project_root=project_root,
        database_path=database_path,
        _params=params,  # Handlers access via ctx.param_str(), ctx.require_int(), etc.
    )

    # Execute handler
    try:
        with ctx:
            result = handler(ctx)
    except Exception as e:
        # Log and re-raise unexpected errors
        ctx.logger.exception("Handler raised exception: %s", e)
        raise

    # Render result
    renderer = get_renderer(output_format)
    exit_code = renderer.render_result(result)

    if exit_code != 0:
        sys.exit(exit_code)


def _get_output_format(command: Any) -> OutputFormat:
    """Get output format from command instance.

    Parameters
    ----------
    command
        Command dataclass instance.

    Returns
    -------
    OutputFormat
        Resolved output format.
    """
    # Check for explicit format field
    if hasattr(command, "output_format"):
        fmt = command.output_format
        if isinstance(fmt, OutputFormat):
            return fmt
        return OutputFormat(str(fmt))

    # Check for --json flag
    if hasattr(command, "json") and command.json:
        return OutputFormat.JSON

    return OutputFormat.TEXT


def _extract_params(command: Any) -> dict[str, Any]:
    """Extract parameters from command dataclass fields.

    Infrastructure fields (output_format, verbose, etc.) are excluded.
    All other fields become handler parameters.

    Parameters
    ----------
    command
        Command dataclass instance.

    Returns
    -------
    dict[str, Any]
        Parameter dictionary.
    """
    if not dataclasses.is_dataclass(command):
        return {}

    params: dict[str, Any] = {}

    for field_info in dataclasses.fields(command):
        name = field_info.name

        # Skip infrastructure fields
        if name in _INFRASTRUCTURE_FIELDS:
            continue

        value = getattr(command, name)
        params[name] = value

    return params


def _get_path_field(command: Any, *field_names: str) -> Path | None:
    """Get Path value from first matching field.

    Parameters
    ----------
    command
        Command dataclass instance.
    field_names
        Field names to try in order.

    Returns
    -------
    Path | None
        Path value or None.
    """
    for name in field_names:
        if hasattr(command, name):
            value = getattr(command, name)
            if value is not None:
                if isinstance(value, Path):
                    return value
                return Path(str(value))
    return None


__all__ = [
    "cli_command",
]
```

---

### Task P5-2: Implement Parameter Extraction

**Duration:** 4 hours

Included in P5-1. Ensure robust parameter extraction:

**Test Cases:**
- Dataclass with simple fields
- Dataclass with default values
- Dataclass with None values
- Dataclass with Path fields
- Dataclass with Enum fields
- Non-dataclass (error handling)

---

### Task P5-3: Implement `__call__` Generation

**Duration:** 4 hours

Included in P5-1. The generated `__call__`:

1. Must work with Cyclopts
2. Must handle all output formats
3. Must propagate exceptions correctly
4. Must clean up resources on error

---

### Task P5-4: Implement Auto-Registration

**Duration:** 2 hours

Included in P5-1. Registration happens at decoration time:

```python
# When @cli_command is applied to a class, it registers immediately
register_operation(OperationSpec(...))
```

---

### Task P5-5: Migrate `commands/jobs.py` (POC)

**Duration:** 2 hours

**File:** `src/codeintel/cli/commands/jobs.py`

**Before:**
```python
import sys
from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.jobs import jobs_list_handler
from codeintel.cli.rendering.types import OutputFormat

jobs_app = App(name="jobs", help="Manage background jobs")


@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    
    status: Annotated[str | None, Parameter(help="Filter by status")] = None
    limit: Annotated[int, Parameter(help="Maximum jobs")] = 20
    output_format: Annotated[OutputFormat, Parameter(name="--format")] = OutputFormat.TEXT

    def __call__(self) -> None:
        runtime_cli = RuntimeCLI()
        output_cli = OutputFormatCLI(output_format=self.output_format)
        params = {"status": self.status, "limit": self.limit}
        
        with command_context(
            "jobs.list",
            runtime_cli,
            output_cli,
            params=params,
            require_runtime=False,
        ) as (ctx, renderer):
            result = jobs_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)
```

**After:**
```python
from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.handlers.jobs import (
    jobs_cancel_handler,
    jobs_cleanup_handler,
    jobs_list_handler,
    jobs_output_handler,
    jobs_status_handler,
)
from codeintel.cli.rendering.types import OutputFormat

jobs_app = App(name="jobs", help="Manage background jobs")


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
    
    status: Annotated[
        Literal["pending", "running", "completed", "failed", "cancelled"] | None,
        Parameter(help="Filter by status"),
    ] = None
    limit: Annotated[int, Parameter(help="Maximum jobs to show")] = 20
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True)] = 0


@cli_command(
    "jobs.status",
    handler=jobs_status_handler,
    require_runtime=False,
    require_gateway=False,
)
@jobs_app.command(name="status")
@dataclass
class JobsStatusCommand:
    """Get status of a background job."""
    
    job_id: Annotated[str, Parameter(help="Job ID")]
    output_format: Annotated[
        OutputFormat,
        Parameter(name="--format", help="Output format"),
    ] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name="-v", count=True)] = 0


# ... similar for other job commands
```

**Removed:**
- `sys` import
- `OutputFormatCLI` import
- `RuntimeCLI` import
- `command_context` import
- All `__call__` methods

**Added:**
- `cli_command` import
- `@cli_command` decorator on each command
- `verbose` field on each command

---

### Task P5-6: Verify POC Works End-to-End

**Duration:** 2 hours

```bash
# Test jobs commands
codeintel jobs list
codeintel jobs list --format json
codeintel jobs list -v
codeintel jobs status test-id
codeintel jobs cancel test-id
codeintel jobs cleanup

# Run tests
uv run pytest tests/cli/commands/test_jobs.py -v
```

---

### Task P5-7: Migrate Remaining Command Files

**Duration:** 12 hours

Migrate each remaining command file following the pattern from P5-5.

**Per-file checklist:**
- [ ] Add `cli_command` import
- [ ] Remove `sys`, `RuntimeCLI`, `OutputFormatCLI`, `command_context` imports
- [ ] Add `@cli_command` decorator to each command class
- [ ] Add `verbose` field to each command class
- [ ] Remove all `__call__` methods
- [ ] Run tests for that file

**Files to migrate:**
1. `commands/health.py`
2. `commands/ops.py`
3. `commands/storage.py`
4. `commands/history.py`
5. `commands/build.py`
6. `commands/graphs.py`
7. `commands/docs.py`
8. `commands/ide.py`
9. `commands/datasets.py`
10. `commands/dataset_ops.py`
11. `commands/plugins.py`
12. `commands/subsystem.py`
13. `commands/serve.py`
14. `commands/config.py`
15. `commands/completions.py`

---

### Task P5-8: Full CLI Smoke Test

**Duration:** 2 hours

Test all major command groups:

```bash
# Jobs
codeintel jobs list
codeintel jobs list --format json

# Health
codeintel health check

# Ops
codeintel ops list

# Storage
codeintel storage info

# Build (with test project)
codeintel build --help

# Graphs
codeintel graphs --help

# Docs
codeintel docs --help

# IDE
codeintel ide --help

# Datasets
codeintel datasets list

# Plugins
codeintel plugins list

# Subsystem
codeintel subsystem --help

# Serve
codeintel serve --help

# Config
codeintel config show

# Completions
codeintel completions install --help
```

---

### Task P5-9: Full Test Suite

**Duration:** 2 hours

```bash
# Full CLI tests
uv run pytest tests/cli/ -v --tb=short

# Verify no manual __call__ methods remain
rg "def __call__\(self\)" src/codeintel/cli/commands/

# Verify all commands have @cli_command
rg "@cli_command" src/codeintel/cli/commands/
```

---

### Task P5-10: Documentation Update

**Duration:** 2 hours

Update any developer documentation:

1. Document the new decorator pattern
2. Provide migration guide for custom commands
3. Update CLI architecture docs

---

## 7. File Changes

### 7.0 Phase 4 Completed Files (Context)

These files were created/modified in Phase 4 and are now available for Phase 5:

| File | Status | Purpose |
|------|--------|---------|
| `execution/registry.py` | ✅ Created | NEW `OperationRegistry` and `OperationSpec` |
| `introspection/__init__.py` | ✅ Modified | Re-exports both NEW and LEGACY registries |
| `handlers/*.py` | ✅ Modified | All 12 handler modules register with NEW registry |
| `operations/*.py` | ✅ Modified | Continue registering with LEGACY for backward compat |
| `tests/cli/execution/test_registry.py` | ✅ Created | Tests for NEW registry |

### 7.1 New Files Created

| File | Purpose |
|------|---------|
| `commands/decorators.py` | `@cli_command` decorator (registers with NEW registry) |
| `tests/cli/commands/test_decorators.py` | Decorator unit tests |

### 7.2 Files Modified

| File | Changes |
|------|---------|
| `commands/jobs.py` | Add decorator, remove `__call__` |
| `commands/health.py` | Add decorator, remove `__call__` |
| `commands/ops.py` | Add decorator, remove `__call__` |
| `commands/storage.py` | Add decorator, remove `__call__` |
| `commands/history.py` | Add decorator, remove `__call__` |
| `commands/build.py` | Add decorator, remove `__call__` |
| `commands/graphs.py` | Add decorator, remove `__call__` |
| `commands/docs.py` | Add decorator, remove `__call__` |
| `commands/ide.py` | Add decorator, remove `__call__` |
| `commands/datasets.py` | Add decorator, remove `__call__` |
| `commands/dataset_ops.py` | Add decorator, remove `__call__` |
| `commands/plugins.py` | Add decorator, remove `__call__` |
| `commands/subsystem.py` | Add decorator, remove `__call__` |
| `commands/serve.py` | Add decorator, remove `__call__` |
| `commands/config.py` | Add decorator, remove `__call__` |
| `commands/completions.py` | Add decorator, remove `__call__` |

### 7.3 Code Removed

From each command file:
- `sys` import (if only used for `sys.exit`)
- `RuntimeCLI` import
- `OutputFormatCLI` import
- `command_context` import
- All `__call__` method implementations

---

## 8. Testing Requirements

### 8.1 Decorator Unit Tests

**File:** `tests/cli/commands/test_decorators.py`

```python
"""Tests for @cli_command decorator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import pytest
from cyclopts import Parameter

from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
# Import from NEW registry (execution/registry.py)
from codeintel.cli.execution.registry import get_registry, reset_registry
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from codeintel.cli.handlers.context import HandlerContext


def dummy_handler(ctx: HandlerContext) -> CliResult[dict[str, bool]]:
    """Return success result for testing.
    
    Parameters
    ----------
    ctx
        Handler context (unused in test).
    
    Returns
    -------
    CliResult[dict[str, bool]]
        Success result with test data.
    """
    _ = ctx  # Acknowledge unused
    return CliResult.ok({"test": True})


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset NEW registry before each test."""
    reset_registry()


class TestCliCommandDecorator:
    """Tests for @cli_command decorator."""

    def test_generates_call_method(self) -> None:
        """Decorator generates __call__ method."""
        @cli_command("test.op", handler=dummy_handler, require_runtime=False)
        @dataclass
        class TestCommand:
            output_format: OutputFormat = OutputFormat.TEXT
            verbose: int = 0
        
        cmd = TestCommand()
        assert hasattr(cmd, "__call__")

    def test_registers_operation(self) -> None:
        """Decorator registers operation."""
        @cli_command("test.register", handler=dummy_handler, require_runtime=False)
        @dataclass
        class TestCommand:
            output_format: OutputFormat = OutputFormat.TEXT
            verbose: int = 0
        
        registry = get_registry()
        assert "test.register" in registry

    def test_uses_docstring_as_description(self) -> None:
        """Decorator uses class docstring as description."""
        @cli_command("test.doc", handler=dummy_handler, require_runtime=False)
        @dataclass
        class TestCommand:
            """This is the description."""
            output_format: OutputFormat = OutputFormat.TEXT
            verbose: int = 0
        
        registry = get_registry()
        spec = registry.get("test.doc")
        assert spec is not None
        assert spec.description == "This is the description."

    def test_extracts_params(self) -> None:
        """Decorator extracts non-infrastructure params."""
        @cli_command("test.params", handler=dummy_handler, require_runtime=False)
        @dataclass
        class TestCommand:
            name: str = "test"
            count: int = 10
            output_format: OutputFormat = OutputFormat.TEXT
            verbose: int = 0
        
        # Params should include name and count, not output_format or verbose
        # This is tested via the handler receiving correct params
```

### 8.2 Integration Testing

Test each migrated command:

```bash
# Per-command testing
uv run pytest tests/cli/commands/test_{command}.py -v
```

### 8.3 Regression Testing

All CLI tests must pass:

```bash
uv run pytest tests/cli/ -v
```

---

## 9. Verification Checklist

### 9.0 Phase 4 Prerequisites Verified

- [x] NEW registry exists at `execution/registry.py`
- [x] `OperationSpec` has: `operation_id`, `name`, `description`, `handler`, `group`, `require_runtime`, `require_gateway`, `require_graph_runtime`, `tags`, `hidden`
- [x] `get_registry()` returns NEW registry singleton
- [x] `register_operation()` registers with NEW registry
- [x] All handler modules have module-level registrations
- [x] Tests pass: `uv run pytest tests/cli/execution/test_registry.py`

### 9.1 Decorator Implementation

- [ ] `commands/decorators.py` created
- [ ] `@cli_command` decorator imports from `execution/registry.py` (NOT introspection)
- [ ] `@cli_command` decorator works
- [ ] Parameter extraction correct
- [ ] Auto-registration uses NEW `OperationSpec` type
- [ ] Exit code handling correct

### 9.2 Command Migration

- [ ] All command files migrated
- [ ] No manual `__call__` methods remain
- [ ] All commands have `verbose` field
- [ ] All commands have `output_format` field

### 9.3 Functionality

- [ ] All CLI commands work
- [ ] Help text renders correctly
- [ ] JSON output mode works
- [ ] Error handling works

---

## 10. Exit Criteria

Phase 5 is complete when:

| Criterion | Status |
|-----------|--------|
| `commands/decorators.py` implemented | ⬜ |
| Decorator unit tests pass | ⬜ |
| `commands/jobs.py` migrated | ⬜ |
| `commands/health.py` migrated | ⬜ |
| `commands/ops.py` migrated | ⬜ |
| `commands/storage.py` migrated | ⬜ |
| `commands/history.py` migrated | ⬜ |
| `commands/build.py` migrated | ⬜ |
| `commands/graphs.py` migrated | ⬜ |
| `commands/docs.py` migrated | ⬜ |
| `commands/ide.py` migrated | ⬜ |
| `commands/datasets.py` migrated | ⬜ |
| `commands/dataset_ops.py` migrated | ⬜ |
| `commands/plugins.py` migrated | ⬜ |
| `commands/subsystem.py` migrated | ⬜ |
| `commands/serve.py` migrated | ⬜ |
| `commands/config.py` migrated | ⬜ |
| `commands/completions.py` migrated | ⬜ |
| No `__call__` methods remain | ⬜ |
| All CLI smoke tests pass | ⬜ |
| All unit tests pass | ⬜ |

---

## 11. Rollback Procedure

**Per-command rollback:**

If a single command migration fails:

```bash
git checkout HEAD~1 -- src/codeintel/cli/commands/{command}.py
```

**Full phase rollback:**

1. Revert all command file changes
2. Delete `commands/decorators.py`
3. Verify tests pass with old code

**Note:** Individual command files are independent, so partial rollbacks are safe.

---

**Previous Phase:** [Phase 4: Registry Unification](./PHASE_4_REGISTRY.md)  
**Next Phase:** [Phase 6: Legacy Cleanup](./PHASE_6_CLEANUP.md)
