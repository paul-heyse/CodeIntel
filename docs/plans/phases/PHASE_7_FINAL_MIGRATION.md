# Phase 7: Final Migration - Complete Legacy Elimination

**Status:** 🔴 NOT STARTED  
**Priority:** HIGH  
**Estimated Effort:** 6-8 hours  
**Prerequisites:** Phase 6 Complete  

## Executive Summary

This phase eliminates **all** legacy patterns, deprecated code, and scattered backward-compatibility shims from the CLI module. Upon completion:

1. **Zero legacy code** exists within the CLI module
2. **Single source of truth** for each concept (one context type, one registry, one execution path)
3. **External compatibility** handled at exactly one well-documented boundary module
4. **No migration comments** or phase references remain in the code

---

## Table of Contents

1. [Goals and Non-Goals](#goals-and-non-goals)
2. [Current State Analysis](#current-state-analysis)
3. [Target Architecture](#target-architecture)
4. [Phase 7A: Migrate graphs.py and serve.py](#phase-7a-migrate-graphspy-and-servepy)
5. [Phase 7B: Delete commands/context.py](#phase-7b-delete-commandscontextpy)
6. [Phase 7C: Unify Handler Utilities](#phase-7c-unify-handler-utilities)
7. [Phase 7D: Eliminate Legacy Context Types](#phase-7d-eliminate-legacy-context-types)
8. [Phase 7E: Clean Execution Layer](#phase-7e-clean-execution-layer)
9. [Phase 7F: Clean Introspection Layer](#phase-7f-clean-introspection-layer)
10. [Phase 7G: External Compatibility Boundary](#phase-7g-external-compatibility-boundary)
11. [Phase 7H: Remove Migration Artifacts](#phase-7h-remove-migration-artifacts)
12. [Verification Checklist](#verification-checklist)
13. [Rollback Plan](#rollback-plan)

---

## Goals and Non-Goals

### Goals

1. **Complete migration** of all CLI commands to `@cli_command` decorator pattern
2. **Single context type** (`HandlerContext` from `handlers/context.py`) used everywhere
3. **Single registry** (`OperationRegistry` from `execution/registry.py`) with no legacy fields
4. **Single execution path** - all commands flow through `@cli_command` → handler
5. **External compatibility** isolated to one boundary module (`cli/compat.py`)
6. **Clean codebase** - no "legacy", "deprecated", "migration", "Phase N" comments

### Non-Goals

- Changing the plugin system's support for manifest-based plugins (keep it)
- Modifying code outside `src/codeintel/cli/` (except tests)
- Breaking external consumers without providing a compatibility shim

---

## Current State Analysis

### Legacy Items Inventory

| Item | Location | Status | Action |
|------|----------|--------|--------|
| `command_context` | `commands/context.py` | Used by graphs.py, serve.py | **DELETE after migration** |
| `EnhancedHandlerContext` | `handlers/protocol.py` | Unused after Phase 6 | **DELETE** |
| `HandlerProtocol` | `handlers/protocol.py` | Unused | **DELETE** |
| `handler_context()` | `handlers/protocol.py` | Unused | **DELETE** |
| Old `HandlerContext` | `handlers/base.py` | Aliased as `LegacyHandlerContext` | **DELETE** |
| `build_handler_context()` | `handlers/base.py` | Unused | **DELETE** |
| `_resolved_to_project_runtime()` | 5 handler files | Duplicated | **CONSOLIDATE** |
| Legacy `OperationSpec` fields | `execution/registry.py` | For executor compat | **DELETE** |
| `get_operation_registry()` | `introspection/__init__.py` | Legacy alias | **SIMPLIFY** |
| `_get_executor` export | `introspection/__init__.py` | Unused | **DELETE** |
| `@operation` decorator | `execution/adapter.py` | Old pattern | **DELETE** |
| `CycloptsAdapter` | `execution/adapter.py` | Old pattern | **DELETE** |
| Migration comments | Throughout | Phase 1-6 references | **DELETE** |

### Files to Delete (Complete)

```
src/codeintel/cli/commands/context.py        # After 7A migration
src/codeintel/cli/handlers/protocol.py       # Legacy contexts
src/codeintel/cli/execution/adapter.py       # Old adapter pattern
```

### Files to Modify Significantly

```
src/codeintel/cli/commands/graphs.py         # Migrate to @cli_command
src/codeintel/cli/commands/serve.py          # Migrate to @cli_command
src/codeintel/cli/commands/_common.py        # Remove context import
src/codeintel/cli/handlers/base.py           # Keep only utilities
src/codeintel/cli/handlers/__init__.py       # Remove legacy exports
src/codeintel/cli/execution/registry.py      # Remove legacy fields
src/codeintel/cli/introspection/__init__.py  # Simplify exports
tests/cli/_harness/__init__.py               # Remove dual-path execution
```

---

## Target Architecture

After Phase 7, the CLI module will have this clean structure:

```
codeintel/cli/
├── __init__.py                    # Public API + external compat shim
├── compat.py                      # NEW: External compatibility boundary
├── commands/
│   ├── __init__.py
│   ├── _common.py                 # RuntimeCLI, OutputFormatCLI (no context)
│   ├── decorators.py              # @cli_command (THE pattern)
│   ├── build.py                   # @cli_command decorated
│   ├── datasets.py                # @cli_command decorated
│   ├── docs.py                    # @cli_command decorated
│   ├── graphs.py                  # @cli_command decorated (migrated)
│   ├── help_commands.py           # @cli_command decorated
│   ├── history.py                 # @cli_command decorated
│   ├── ide.py                     # @cli_command decorated
│   ├── jobs.py                    # @cli_command decorated
│   ├── ops.py                     # @cli_command decorated
│   ├── plugins.py                 # @cli_command decorated
│   ├── serve.py                   # @cli_command decorated (migrated)
│   ├── storage.py                 # @cli_command decorated
│   └── subsystems.py              # @cli_command decorated
├── execution/
│   ├── __init__.py
│   ├── bootstrap.py               # CLI startup
│   ├── context.py                 # ExecutionContext (internal)
│   ├── executor.py                # OperationExecutor (simplified)
│   └── registry.py                # OperationSpec, OperationRegistry (clean)
├── handlers/
│   ├── __init__.py                # Clean exports only
│   ├── _utilities.py              # NEW: setup_logging, get_handler_logger, etc.
│   ├── context.py                 # HandlerContext (THE context)
│   ├── build.py                   # Handler functions
│   ├── datasets.py                # Handler functions
│   ├── docs.py                    # Handler functions
│   ├── graphs.py                  # Handler functions
│   ├── history.py                 # Handler functions
│   ├── ide.py                     # Handler functions
│   ├── jobs.py                    # Handler functions
│   ├── macros.py                  # Handler functions
│   ├── ops.py                     # Handler functions
│   ├── plugins.py                 # Handler functions
│   ├── storage.py                 # Handler functions
│   └── subsystems.py              # Handler functions
├── introspection/
│   ├── __init__.py                # Clean: just discovery + help
│   ├── discovery.py               # Operation discovery
│   ├── help.py                    # Help rendering
│   └── params.py                  # Parameter introspection
└── ...
```

**Deleted files:**
- `commands/context.py` - Replaced by handlers/context.py
- `handlers/protocol.py` - Legacy contexts
- `handlers/base.py` - Split: utilities → `_utilities.py`, classes → deleted
- `execution/adapter.py` - Old adapter pattern

---

## Phase 7A: Migrate graphs.py and serve.py

### Objective

Convert the last two command files using manual `__call__` to the `@cli_command` decorator pattern.

### 7A.1: Migrate `commands/graphs.py`

**Current Implementation:**
```python
# commands/graphs.py - CURRENT (LEGACY)
from codeintel.cli.commands.context import command_context

@dataclass
class GraphPluginsCommand:
    def __call__(self) -> None:
        with command_context("graph.plugins", ...) as (ctx, renderer):
            if self.plan:
                result = graph_plugins_plan_handler(ctx)
            else:
                result = graph_plugins_list_handler(ctx)
            renderer.render_result(result)
```

**Target Implementation:**
```python
# commands/graphs.py - TARGET
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.handlers.graphs import (
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
)

@graphs_app.command(name="plugins-list")
@cli_command("graph.plugins.list", handler=graph_plugins_list_handler)
@dataclass
class GraphPluginsListCommand:
    """List registered graph plugins."""
    names: Annotated[list[str] | None, ...] = None
    include_disabled: Annotated[bool, ...] = True
    output_format: OutputFmt = OutputFormat.TEXT
    verbose: Verbose = 0

@graphs_app.command(name="plugins-plan")
@cli_command("graph.plugins.plan", handler=graph_plugins_plan_handler)
@dataclass
class GraphPluginsPlanCommand:
    """Display an execution plan for graph plugins."""
    names: Annotated[list[str] | None, ...] = None
    enable: Annotated[list[str] | None, ...] = None
    disable: Annotated[list[str] | None, ...] = None
    selection_policy: Annotated[SelectionPolicy, ...] = SelectionPolicy.LENIENT
    dependency_policy: Annotated[DependencyPolicy, ...] = DependencyPolicy.STRICT
    output_format: OutputFmt = OutputFormat.TEXT
    verbose: Verbose = 0
```

**Changes Required:**

1. Split `GraphPluginsCommand` into two separate commands:
   - `GraphPluginsListCommand` (default behavior)
   - `GraphPluginsPlanCommand` (--plan flag becomes separate command)

2. Apply `@cli_command` decorator to each

3. Remove `__call__` method from both classes

4. Remove import: `from codeintel.cli.commands.context import command_context`

5. Update `graphs_app` command names if needed for backward CLI compatibility

**Handler Updates (if needed):**

The handlers in `handlers/graphs.py` already accept `HandlerContext` - no changes needed.

### 7A.2: Migrate `commands/serve.py`

**Current Implementation:**
```python
# commands/serve.py - CURRENT (LEGACY)
from codeintel.cli.commands.context import command_context

@dataclass
class ServeHttpCommand:
    def __call__(self) -> None:
        with command_context("serve.http", ...) as (ctx, renderer):
            result = serve_http_handler(ctx)
            renderer.render_result(result)
```

**Target Implementation:**
```python
# commands/serve.py - TARGET
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.handlers.ops import serve_http_handler, serve_mcp_handler

@serve_app.command(name="http")
@cli_command("serve.http", handler=serve_http_handler)
@dataclass
class ServeHttpCommand:
    """Start the HTTP server."""
    host: Annotated[str, ...] = "127.0.0.1"
    port: Annotated[int, ...] = 8000
    auto_pipeline: Annotated[bool, ...] = False
    reload: Annotated[bool, ...] = False
    root: ProjectRoot = None
    verbose: Verbose = 0

@serve_app.command(name="mcp")
@cli_command("serve.mcp", handler=serve_mcp_handler)
@dataclass
class ServeMcpCommand:
    """Start the MCP server."""
    auto_pipeline: Annotated[bool, ...] = False
    root: ProjectRoot = None
    verbose: Verbose = 0
```

**Changes Required:**

1. Add `@cli_command` decorator to both command classes
2. Remove `__call__` method from both classes
3. Remove import: `from codeintel.cli.commands.context import command_context`
4. Verify handlers accept `HandlerContext` (they should already)

### 7A.3: Update `commands/_common.py`

Remove the re-export of `command_context`:

```python
# REMOVE this line:
from codeintel.cli.commands.context import command_context

# REMOVE from __all__:
"command_context",
```

### 7A Verification

```bash
# Run after 7A:
uv run python -c "from codeintel.cli.commands.graphs import graphs_app; print('OK')"
uv run python -c "from codeintel.cli.commands.serve import serve_app; print('OK')"
uv run pytest tests/cli/test_graphs*.py -v
uv run pytest tests/cli/test_serve*.py -v
```

---

## Phase 7B: Delete commands/context.py

### Objective

After 7A, `commands/context.py` has no internal consumers. Delete it entirely.

### Actions

1. **Verify no imports remain:**
   ```bash
   rg "from codeintel.cli.commands.context import" src/
   rg "from codeintel.cli.commands import.*context" src/
   ```

2. **Delete the file:**
   ```bash
   rm src/codeintel/cli/commands/context.py
   ```

3. **Update `commands/__init__.py`:**
   Remove any export of `command_context` or `CommandContextError`.

4. **External compatibility:**
   If external code depends on `command_context`, add to `cli/compat.py` (Phase 7G).

### 7B Verification

```bash
uv run python -c "import codeintel.cli.commands; print('OK')"
uv run ruff check src/codeintel/cli/commands/
```

---

## Phase 7C: Unify Handler Utilities

### Objective

Consolidate duplicated utilities and establish a clean separation between:
- **Utilities** (`_utilities.py`) - Functions that remain
- **Legacy classes** - Delete entirely

### 7C.1: Create `handlers/_utilities.py`

Extract utilities from `handlers/base.py`:

```python
# handlers/_utilities.py - NEW FILE
"""Handler utilities - logging, gateway management.

This module provides shared utilities for CLI handlers:
- Logging setup and configuration
- Gateway management for database access
"""

from __future__ import annotations

import logging
from pathlib import Path

from codeintel.cli.config import CliConfig, load_config
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

LOG = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False
VERBOSITY_DEBUG = 2
VERBOSITY_INFO = 1


def setup_logging(
    verbosity: int = 0,
    *,
    config: CliConfig | None = None,
    force: bool = False,
) -> None:
    """Configure logging for CLI handlers.

    Parameters
    ----------
    verbosity
        Verbosity level (0=warnings, 1=info, 2+=debug).
    config
        Optional CLI configuration for default log level.
    force
        If True, reconfigure even if already configured.
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603

    if _LOGGING_CONFIGURED and not force:
        return

    level = _determine_log_level(verbosity, config)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=force,
    )
    _LOGGING_CONFIGURED = True


def _determine_log_level(verbosity: int, config: CliConfig | None) -> int:
    """Determine log level from verbosity and config."""
    if verbosity >= VERBOSITY_DEBUG:
        return logging.DEBUG
    if verbosity >= VERBOSITY_INFO:
        return logging.INFO
    if config is not None:
        return getattr(logging, config.log_level, logging.WARNING)
    return logging.WARNING


def get_handler_logger(name: str) -> logging.Logger:
    """Get a logger for a handler.

    Parameters
    ----------
    name
        Logger name (typically operation_id).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    return logging.getLogger(f"codeintel.cli.handlers.{name}")


def open_handler_gateway(
    db_path: Path,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open a gateway for handler use.

    Parameters
    ----------
    db_path
        Path to the database file.
    read_only
        Whether to open in read-only mode.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    storage_config = StorageConfig(db_path=db_path, read_only=read_only)
    return open_gateway(storage_config)


__all__ = [
    "VERBOSITY_DEBUG",
    "VERBOSITY_INFO",
    "get_handler_logger",
    "open_handler_gateway",
    "setup_logging",
]
```

### 7C.2: Consolidate `_resolved_to_project_runtime()`

This helper appears in 5 files. Create a single source:

```python
# Add to handlers/_utilities.py:

from codeintel.cli.project import ProjectRuntime
from codeintel.cli.resolution.types import ResolvedRuntime


def resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    """Convert ResolvedRuntime to ProjectRuntime.

    This conversion is needed when interfacing with code that expects
    the older ProjectRuntime type.

    Parameters
    ----------
    runtime
        Resolved runtime from the new resolution system.

    Returns
    -------
    ProjectRuntime
        Equivalent ProjectRuntime instance.
    """
    return ProjectRuntime(
        root=runtime.root,
        project=runtime.project,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        cfg=runtime.config,
        serving=runtime.serving,
    )
```

Then update all handler files to import from `_utilities`:

**Files to update:**
- `handlers/ops.py`
- `handlers/datasets.py`
- `handlers/build.py`
- `handlers/docs.py`

```python
# In each file, replace:
def _resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    ...

# With:
from codeintel.cli.handlers._utilities import resolved_to_project_runtime
```

### 7C.3: Delete `handlers/base.py`

After extracting utilities, delete the entire file:

```bash
rm src/codeintel/cli/handlers/base.py
```

### 7C Verification

```bash
uv run python -c "from codeintel.cli.handlers._utilities import setup_logging; print('OK')"
uv run ruff check src/codeintel/cli/handlers/
uv run pyright src/codeintel/cli/handlers/
```

---

## Phase 7D: Eliminate Legacy Context Types

### Objective

Delete `handlers/protocol.py` and remove all references to:
- `EnhancedHandlerContext`
- `HandlerProtocol`
- `handler_context()`
- `LegacyHandlerContext`

### 7D.1: Verify No Usage

```bash
rg "EnhancedHandlerContext" src/
rg "HandlerProtocol" src/
rg "handler_context\(" src/
rg "LegacyHandlerContext" src/
```

Expected: Only in `handlers/__init__.py` and `handlers/protocol.py`

### 7D.2: Update `handlers/context.py`

Remove TYPE_CHECKING import of `EnhancedHandlerContext` if present:

```python
# REMOVE if present:
if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext
```

### 7D.3: Delete `handlers/protocol.py`

```bash
rm src/codeintel/cli/handlers/protocol.py
```

### 7D.4: Update `handlers/__init__.py`

Remove all legacy exports:

```python
# REMOVE these imports:
from codeintel.cli.handlers.base import (
    HandlerContext as LegacyHandlerContext,
)
from codeintel.cli.handlers.base import build_handler_context
from codeintel.cli.handlers.protocol import (
    EnhancedHandlerContext,
    HandlerProtocol,
    handler_context,
)

# REMOVE from __all__:
"EnhancedHandlerContext",
"HandlerProtocol",
"LegacyHandlerContext",
"build_handler_context",
"handler_context",
```

Update imports to use `_utilities`:

```python
from codeintel.cli.handlers._utilities import (
    get_handler_logger,
    open_handler_gateway,
    setup_logging,
)
```

### 7D Verification

```bash
uv run python -c "from codeintel.cli.handlers import HandlerContext; print('OK')"
uv run ruff check src/codeintel/cli/handlers/
uv run pyright src/codeintel/cli/handlers/
```

---

## Phase 7E: Clean Execution Layer

### Objective

Remove legacy fields from `OperationSpec` and delete the old `execution/adapter.py`.

### 7E.1: Remove Legacy Fields from `OperationSpec`

**Current:**
```python
@dataclass
class OperationSpec:
    operation_id: str
    name: str
    description: str
    handler: Callable[..., CliResult[T]]
    group: str
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    tags: tuple[str, ...] = ()
    hidden: bool = False
    
    # Legacy fields for executor compatibility - DELETE THESE
    is_async: bool = False
    is_streaming: bool = False
    param_schema: object | None = None
    requires_progress: bool = False
    estimated_duration: float | None = None
    retryable: bool = True
    retry_policy: object | None = None
    timeout: float | None = None
    category: OperationCategory = OperationCategory.READ
```

**Target:**
```python
@dataclass
class OperationSpec(Generic[T]):
    """Specification for a CLI operation.

    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "build.run").
    name
        Human-readable name.
    description
        Operation description.
    handler
        Handler function accepting HandlerContext.
    group
        Grouping for help display.
    require_runtime
        Whether operation needs resolved runtime.
    require_gateway
        Whether operation needs database gateway.
    require_graph_runtime
        Whether operation needs graph runtime.
    tags
        Optional tags for categorization.
    hidden
        Whether to hide from help output.
    """

    operation_id: str
    name: str
    description: str
    handler: Callable[[HandlerContext], CliResult[T]]
    group: str
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    tags: tuple[str, ...] = ()
    hidden: bool = False
```

### 7E.2: Update Test Harness

Update `tests/cli/_harness/__init__.py` to remove dual-path execution:

```python
# REMOVE: Legacy executor-based execution path
# KEEP: Only handler-based execution

def execute(self) -> ExecutionResult:
    """Execute the operation via handler."""
    spec = self._get_spec()
    ctx = self._build_handler_context(spec)
    
    try:
        result = spec.handler(ctx)
        return ExecutionResult(
            success=result.is_ok,
            result=result,
            error=None,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            result=None,
            error=str(e),
        )
    finally:
        ctx.close()
```

### 7E.3: Delete `execution/adapter.py`

```bash
rm src/codeintel/cli/execution/adapter.py
```

### 7E.4: Update `execution/__init__.py`

Remove exports of deleted items:
```python
# REMOVE:
from codeintel.cli.execution.adapter import (
    CycloptsAdapter,
    adapt_cyclopts_command,
    operation,
)
```

### 7E Verification

```bash
uv run python -c "from codeintel.cli.execution.registry import OperationSpec; print('OK')"
uv run ruff check src/codeintel/cli/execution/
uv run pyright src/codeintel/cli/execution/
uv run pytest tests/cli/ -v -k "not integration"
```

---

## Phase 7F: Clean Introspection Layer

### Objective

Simplify the introspection module to remove legacy re-exports.

### 7F.1: Update `introspection/__init__.py`

**Remove:**
```python
# REMOVE these:
from codeintel.cli.execution.executor import get_executor as _get_executor

def get_operation_registry() -> OperationRegistry:
    """Legacy compatibility..."""
    return get_registry()
```

**Simplify to:**
```python
"""CLI introspection utilities.

Provide operation discovery, help rendering, and parameter inspection.
"""

from __future__ import annotations

from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
    reset_registry,
)
from codeintel.cli.introspection.discovery import (
    OperationInfo,
    get_operation_info,
    list_all_operations,
    list_operations_by_group,
    search_operations,
)
from codeintel.cli.introspection.help import HelpRenderer
from codeintel.cli.introspection.params import (
    CliParamSpec,
    ParamSource,
    ValidationSchema,
    Validator,
    build_cli_param_spec,
    build_cli_param_specs_for_operation,
)

__all__ = [
    "CliParamSpec",
    "HelpRenderer",
    "OperationInfo",
    "OperationRegistry",
    "OperationSpec",
    "ParamSource",
    "ValidationSchema",
    "Validator",
    "build_cli_param_spec",
    "build_cli_param_specs_for_operation",
    "get_operation_info",
    "get_registry",
    "list_all_operations",
    "list_operations_by_group",
    "register_operation",
    "reset_registry",
    "search_operations",
]
```

### 7F Verification

```bash
uv run python -c "from codeintel.cli.introspection import get_registry; print('OK')"
uv run ruff check src/codeintel/cli/introspection/
```

---

## Phase 7G: External Compatibility Boundary

### Objective

Create a single, well-documented boundary module for external code that depends on legacy patterns.

### 7G.1: Create `cli/compat.py`

```python
"""External compatibility shims.

This module provides compatibility for external code that depends on
legacy CLI patterns. Internal code MUST NOT use this module.

All exports are deprecated and will be removed in a future version.

Migration Guide
---------------
- `command_context` → Use `@cli_command` decorator
- `get_operation_registry` → Use `get_registry` from `execution.registry`
- `EnhancedHandlerContext` → Use `HandlerContext` from `handlers.context`
- `build_handler_context` → Use `handler_context_manager` from `handlers.context`

Version History
---------------
- Added: v2.0 (Phase 7)
- Planned removal: v3.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import contextmanager

    from codeintel.cli.handlers.context import HandlerContext
    from codeintel.cli.rendering.service import UnifiedRenderer


def __getattr__(name: str) -> object:
    """Lazy attribute access with deprecation warnings."""
    if name == "command_context":
        warnings.warn(
            "command_context is deprecated. Use @cli_command decorator instead. "
            "See: docs/migration/cli-commands.md",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import on demand to avoid circular imports
        from codeintel.cli.commands.context import command_context as _cmd_ctx
        return _cmd_ctx
    
    if name == "get_operation_registry":
        warnings.warn(
            "get_operation_registry is deprecated. Use get_registry from "
            "codeintel.cli.execution.registry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.execution.registry import get_registry
        return get_registry
    
    if name == "EnhancedHandlerContext":
        warnings.warn(
            "EnhancedHandlerContext is deprecated. Use HandlerContext from "
            "codeintel.cli.handlers.context instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.handlers.context import HandlerContext
        return HandlerContext
    
    if name == "build_handler_context":
        warnings.warn(
            "build_handler_context is deprecated. Use handler_context_manager from "
            "codeintel.cli.handlers.context instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.handlers.context import handler_context_manager
        return handler_context_manager
    
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    # All exports are deprecated - do not add new items
    "EnhancedHandlerContext",
    "build_handler_context",
    "command_context",
    "get_operation_registry",
]
```

### 7G.2: Update `cli/__init__.py`

Add documentation pointing to compat module:

```python
"""CodeIntel CLI package.

Public API
----------
- HandlerContext: Context for CLI handlers
- CliResult: Result type for CLI operations
- @cli_command: Decorator for command definitions

External Compatibility
----------------------
If you depend on legacy patterns (command_context, EnhancedHandlerContext),
import from codeintel.cli.compat instead. These are deprecated and will
be removed in v3.0.
"""

# ... existing imports ...

# Note: Legacy exports moved to cli/compat.py
# External code using old patterns should import from there
```

### 7G.3: Document External Migration

Create or update migration documentation:

```markdown
# docs/migration/cli-commands.md

## Migrating from Legacy CLI Patterns

### command_context → @cli_command

**Before (deprecated):**
```python
from codeintel.cli.commands.context import command_context

@dataclass
class MyCommand:
    def __call__(self):
        with command_context("my.op", runtime_cli, output_cli) as (ctx, renderer):
            result = my_handler(ctx)
            renderer.render_result(result)
```

**After:**
```python
from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.handlers.my_module import my_handler

@my_app.command(name="op")
@cli_command("my.op", handler=my_handler)
@dataclass
class MyCommand:
    param: str
    verbose: int = 0
```

The `@cli_command` decorator handles all infrastructure automatically.
```

### 7G Verification

```bash
# Verify compat module works
uv run python -c "from codeintel.cli.compat import command_context; print('OK')"

# Verify deprecation warning is raised
uv run python -W error::DeprecationWarning -c "from codeintel.cli.compat import command_context" 2>&1 | grep -q "DeprecationWarning" && echo "Warning raised correctly"
```

---

## Phase 7H: Remove Migration Artifacts

### Objective

Remove all comments referencing migration, legacy, or phases from the codebase.

### 7H.1: Remove Phase References

Search and update:
```bash
rg -l "Phase [0-9]" src/codeintel/cli/
rg -l "Phase1|Phase2|Phase3|Phase4|Phase5|Phase6" src/codeintel/cli/
```

**Update each file** to remove or rewrite comments:

- "Phase 6 Migration" → Remove entirely
- "Phase 1+" → Remove, the code is now canonical
- "NEW: Unified handler context (Phase 1+)" → "Handler context"

### 7H.2: Remove Legacy Comments

Search and update:
```bash
rg -l "legacy|LEGACY" src/codeintel/cli/ --ignore-case
rg -l "backward.?compat" src/codeintel/cli/ --ignore-case
rg -l "deprecated" src/codeintel/cli/ --ignore-case
```

**Guidelines:**
- Remove comments like "Legacy context types - retained for backward compatibility"
- Keep deprecation warnings in `compat.py` (they warn external users)
- Update docstrings to describe current behavior, not migration history

### 7H.3: Clean Up `__all__` Lists

Simplify and alphabetize `__all__` in all modules:

```python
# Example: handlers/__init__.py
__all__ = [
    # Result types
    "BuildHistoryResult",
    "BuildStatusResult",
    ...
    # Context
    "HandlerContext",
    "HandlerContextOptions",
    "ParameterError",
    # Handlers
    "build_history_handler",
    "build_status_handler",
    ...
    # Utilities
    "get_handler_logger",
    "handler_context_manager",
    "open_handler_gateway",
    "setup_logging",
]
```

### 7H.4: Archive Phase Documents

Move completed phase documents to archive:

```bash
mkdir -p docs/plans/phases/archive/
mv docs/plans/phases/PHASE_[0-6]*.md docs/plans/phases/archive/
```

Update `docs/plans/phases/README.md` to reflect completion.

### 7H Verification

```bash
# Verify no migration comments remain
rg "Phase [0-9]" src/codeintel/cli/ && echo "FAIL: Phase references found" || echo "OK"
rg "legacy" src/codeintel/cli/ --ignore-case | grep -v "compat.py" | grep -v "plugins/" && echo "FAIL: Legacy references found" || echo "OK"

# Verify code still works
uv run ruff check src/codeintel/cli/
uv run pyright src/codeintel/cli/
uv run pytest tests/cli/ -v
```

---

## Verification Checklist

### Pre-Flight Checks (Before Starting)

- [ ] All tests passing: `uv run pytest tests/cli/ -v`
- [ ] Clean lint: `uv run ruff check src/codeintel/cli/`
- [ ] Clean types: `uv run pyright src/codeintel/cli/`
- [ ] Git working tree clean

### Phase 7A Verification

- [ ] `graphs.py` uses `@cli_command` decorator
- [ ] `serve.py` uses `@cli_command` decorator
- [ ] No `__call__` methods in command classes
- [ ] No import from `commands/context.py`
- [ ] Tests pass: `uv run pytest tests/cli/test_graph*.py tests/cli/test_serve*.py -v`

### Phase 7B Verification

- [ ] `commands/context.py` deleted
- [ ] No imports of deleted file remain
- [ ] CLI still functional: `uv run python -m codeintel.cli --help`

### Phase 7C Verification

- [ ] `handlers/_utilities.py` created with all utilities
- [ ] `_resolved_to_project_runtime` consolidated (single location)
- [ ] All handler files import from `_utilities`
- [ ] `handlers/base.py` deleted

### Phase 7D Verification

- [ ] `handlers/protocol.py` deleted
- [ ] No `EnhancedHandlerContext` references
- [ ] No `LegacyHandlerContext` references
- [ ] `handlers/__init__.py` exports only current types

### Phase 7E Verification

- [ ] `OperationSpec` has no legacy fields
- [ ] `execution/adapter.py` deleted
- [ ] Test harness uses handler-only execution
- [ ] All tests pass

### Phase 7F Verification

- [ ] `introspection/__init__.py` simplified
- [ ] No `get_operation_registry()` function (or simple alias)
- [ ] No `_get_executor` export

### Phase 7G Verification

- [ ] `cli/compat.py` created
- [ ] All legacy exports raise `DeprecationWarning`
- [ ] Migration documentation created

### Phase 7H Verification

- [ ] No "Phase N" comments in `src/codeintel/cli/`
- [ ] No "legacy" comments (except compat.py)
- [ ] No "migration" comments
- [ ] Phase documents archived

### Final Verification

```bash
# Complete verification script
echo "=== Final Phase 7 Verification ==="

echo "1. Checking for deleted files..."
test -f src/codeintel/cli/commands/context.py && echo "FAIL: context.py exists" || echo "OK: context.py deleted"
test -f src/codeintel/cli/handlers/protocol.py && echo "FAIL: protocol.py exists" || echo "OK: protocol.py deleted"
test -f src/codeintel/cli/handlers/base.py && echo "FAIL: base.py exists" || echo "OK: base.py deleted"
test -f src/codeintel/cli/execution/adapter.py && echo "FAIL: adapter.py exists" || echo "OK: adapter.py deleted"

echo "2. Checking for legacy imports..."
rg "from codeintel.cli.commands.context import" src/ && echo "FAIL" || echo "OK"
rg "from codeintel.cli.handlers.protocol import" src/ && echo "FAIL" || echo "OK"
rg "from codeintel.cli.handlers.base import" src/ && echo "FAIL" || echo "OK"

echo "3. Checking for legacy comments..."
rg "Phase [0-6]" src/codeintel/cli/ && echo "FAIL" || echo "OK"

echo "4. Running lint..."
uv run ruff check src/codeintel/cli/

echo "5. Running type check..."
uv run pyright src/codeintel/cli/

echo "6. Running tests..."
uv run pytest tests/cli/ -v --tb=short

echo "=== Verification Complete ==="
```

---

## Rollback Plan

### If Phase 7A Fails

```bash
git checkout HEAD -- src/codeintel/cli/commands/graphs.py
git checkout HEAD -- src/codeintel/cli/commands/serve.py
```

### If Any Phase Fails

```bash
# Full rollback
git checkout HEAD -- src/codeintel/cli/
git checkout HEAD -- tests/cli/
```

### Incremental Rollback

Each phase can be rolled back independently by restoring the specific files modified in that phase. Commit after each phase to enable granular rollback.

---

## Execution Order

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 7A: Migrate graphs.py and serve.py                    │
│ - Convert to @cli_command                                   │
│ - Remove __call__ methods                                   │
│ - Test thoroughly                                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7B: Delete commands/context.py                        │
│ - Verify no imports                                         │
│ - Delete file                                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7C: Unify Handler Utilities                           │
│ - Create _utilities.py                                      │
│ - Consolidate _resolved_to_project_runtime                  │
│ - Delete handlers/base.py                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7D: Eliminate Legacy Context Types                    │
│ - Delete handlers/protocol.py                               │
│ - Clean handlers/__init__.py                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7E: Clean Execution Layer                             │
│ - Remove legacy OperationSpec fields                        │
│ - Update test harness                                       │
│ - Delete execution/adapter.py                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7F: Clean Introspection Layer                         │
│ - Simplify introspection/__init__.py                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7G: External Compatibility Boundary                   │
│ - Create cli/compat.py                                      │
│ - Document migration path                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7H: Remove Migration Artifacts                        │
│ - Remove phase comments                                     │
│ - Remove legacy comments                                    │
│ - Archive phase documents                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

Upon completion of Phase 7:

1. **Zero legacy files:** No `context.py`, `protocol.py`, `base.py`, or `adapter.py`
2. **Single context type:** Only `HandlerContext` from `handlers/context.py`
3. **Single registry:** Only `OperationRegistry` from `execution/registry.py`
4. **Clean OperationSpec:** No legacy fields (is_async, param_schema, etc.)
5. **All commands decorated:** 100% use `@cli_command`
6. **External compat isolated:** Only in `cli/compat.py`
7. **No migration comments:** Zero grep hits for "Phase", "legacy", "migration" in src/cli/
8. **All tests passing:** `pytest tests/cli/` green
9. **Clean static analysis:** `ruff` and `pyright` clean

---

## Appendix: File Inventory

### Files to Delete

| File | Phase | Reason |
|------|-------|--------|
| `commands/context.py` | 7B | Replaced by handlers/context.py |
| `handlers/protocol.py` | 7D | Legacy context types |
| `handlers/base.py` | 7C | Split into _utilities.py |
| `execution/adapter.py` | 7E | Old adapter pattern |

### Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `handlers/_utilities.py` | 7C | Shared utilities |
| `cli/compat.py` | 7G | External compatibility |

### Files with Significant Changes

| File | Phase | Changes |
|------|-------|---------|
| `commands/graphs.py` | 7A | Add @cli_command, remove __call__ |
| `commands/serve.py` | 7A | Add @cli_command, remove __call__ |
| `commands/_common.py` | 7A | Remove context import |
| `handlers/__init__.py` | 7D | Remove legacy exports |
| `execution/registry.py` | 7E | Remove legacy fields |
| `introspection/__init__.py` | 7F | Simplify exports |
| `tests/cli/_harness/__init__.py` | 7E | Handler-only execution |
