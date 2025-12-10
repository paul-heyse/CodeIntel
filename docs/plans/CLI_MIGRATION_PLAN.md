# CLI Unified Architecture Migration Plan

> **Status:** In Progress (Phase 2 Complete)  
> **Target Architecture:** [CLI_UNIFIED_ARCHITECTURE.md](./CLI_UNIFIED_ARCHITECTURE.md)  
> **Estimated Duration:** 4-6 weeks (20-29 working days)  
> **Last Updated:** 2025-12-10

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Target State Summary](#3-target-state-summary)
4. [Migration Strategy](#4-migration-strategy)
5. [Phase 0: Preparation](#5-phase-0-preparation)
6. [Phase 1: Foundation Layer](#6-phase-1-foundation-layer)
7. [Phase 2: Rendering Consolidation](#7-phase-2-rendering-consolidation)
8. [Phase 3: Handler Migration](#8-phase-3-handler-migration)
9. [Phase 4: Registry Unification](#9-phase-4-registry-unification)
10. [Phase 5: Command Decorator & Migration](#10-phase-5-command-decorator--migration)
11. [Phase 6: Legacy Cleanup](#11-phase-6-legacy-cleanup)
12. [Risk Management](#12-risk-management)
13. [Testing Strategy](#13-testing-strategy)
14. [Rollback Procedures](#14-rollback-procedures)
15. [Success Criteria](#15-success-criteria)
16. [Appendices](#16-appendices)

---

## 1. Executive Summary

### 1.1 Objective

Migrate the CodeIntel CLI from its current architecture—characterized by duplicated context types, parallel rendering stacks, and boilerplate-heavy command wiring—to a unified, handler-centric architecture that provides:

- **Single context type** (`HandlerContext`) for all operations
- **Single rendering implementation** (`UnifiedRenderer`)
- **Declarative command binding** via `@cli_command` decorator
- **Unified operation registry** for introspection and execution
- **Zero legacy code** at completion (no compatibility shims)

### 1.2 Scope

**In Scope:**
- All files under `src/codeintel/cli/`
- Associated test files under `tests/cli/`
- Handler, command, rendering, execution, and introspection modules

**Out of Scope:**
- Core domain logic in `src/codeintel/` (non-CLI)
- External integrations (MCP, HTTP serving) except their CLI entry points
- Documentation site generation (except CLI help text)

### 1.3 Key Constraints

1. **No feature regressions** — All existing CLI functionality must continue working
2. **Incremental migration** — Each phase must leave the system in a working state
3. **Test coverage maintained** — No reduction in test coverage during migration
4. **Clean end state** — Final architecture has zero legacy/compatibility code

### 1.4 Phase Overview

| Phase | Name | Duration | Primary Deliverable |
|-------|------|----------|---------------------|
| 0 | Preparation | 1-2 days | Migration tooling and baseline |
| 1 | Foundation Layer | 3-4 days | `HandlerContext` + `bootstrap_cli()` |
| 2 | Rendering Consolidation | 2-3 days | Single `UnifiedRenderer` stack |
| 3 | Handler Migration | 5-7 days | All handlers on new context |
| 4 | Registry Unification | 2-3 days | Single `OperationRegistry` |
| 5 | Command Decorator | 5-7 days | `@cli_command` + all commands migrated |
| 6 | Legacy Cleanup | 2-3 days | All legacy code deleted |

---

## 2. Current State Assessment

### 2.1 Context Type Proliferation

The CLI currently has **four distinct context types** that evolved independently:

| Type | Location | Purpose | Key Properties |
|------|----------|---------|----------------|
| `HandlerContext` | `handlers/base.py` | Basic handler context | `config`, `execution`, `project_root`, `verbosity` |
| `EnhancedHandlerContext` | `handlers/protocol.py` | Extended context with resources | Above + lazy `gateway`, `graph_runtime` |
| `ExecutionContext` | `execution/context.py` | Executor-oriented context | `operation_id`, `params`, `require_runtime()` |
| Context Manager | `commands/context.py` | Command-layer context | Creates `EnhancedHandlerContext` + renderer |

**Problem:** Different code paths use different contexts, leading to inconsistent parameter access, resource lifecycle management, and logging behavior.

### 2.2 Rendering Stack Duplication

Two parallel rendering implementations exist:

| Implementation | Location | Consumers |
|----------------|----------|-----------|
| `UnifiedRenderer` | `rendering/service.py` | `commands/context.py`, some handlers |
| `RichRenderer`/`PlainRenderer` | `rendering/renderers.py` | `execution/executor.py`, `execution/adapter.py` |

Additionally, `rendering/renderers.py` duplicates:
- `ColumnSpec` and `TableSpec` (canonical versions in `rendering/table.py`)
- `CODEINTEL_THEME` definition

### 2.3 Command Wiring Boilerplate

Each command file contains repetitive patterns:

```python
# Typical command structure (repeated ~50 times)
@app.command(name="...")
@dataclass
class SomeCommand:
    # Parameters...
    output_format: Annotated[OutputFormat, Parameter(...)] = OutputFormat.TEXT

    def __call__(self) -> None:
        runtime_cli = RuntimeCLI(...)
        output_cli = OutputFormatCLI(output_format=self.output_format)
        params = {/* manual extraction */}
        with command_context(...) as (ctx, renderer):
            result = handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)
```

### 2.4 Handler Parameter Parsing Duplication

Multiple handler files define identical parameter extraction functions:

| Function | Defined In |
|----------|------------|
| `_get_str_param()` | `handlers/ops.py`, `handlers/jobs.py`, `handlers/history.py`, `handlers/storage.py` |
| `_get_int_param()` | Same files |
| `_get_bool_param()` | Same files |
| `_require_str_param()` | Same files |
| `_get_path_param()` | `handlers/history.py` |
| `_get_enum_str_param()` | `handlers/history.py` |

### 2.5 Operation Registry Drift

The `operations/*.py` modules register `OperationSpec` instances with placeholder handlers, while actual implementations live in `handlers/*.py`. This creates:

- Metadata duplication (operation names, descriptions in two places)
- Potential for specs and handlers to diverge
- Unclear which is authoritative

### 2.6 Logging/Bootstrap Duplication

Logging is configured in multiple locations:

| Location | Function | Notes |
|----------|----------|-------|
| `handlers/base.py` | `setup_logging()` | Global `_LOGGING_CONFIGURED` flag |
| `commands/context.py` | Inline setup | Calls `setup_logging()` |
| `execution/adapter.py` | Inline setup | Reloads config, calls setup |

---

## 3. Target State Summary

See [CLI_UNIFIED_ARCHITECTURE.md](./CLI_UNIFIED_ARCHITECTURE.md) for complete specification.

### 3.1 Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `HandlerContext` | `handlers/context.py` | Single unified context for all handlers |
| `bootstrap_cli()` | `execution/bootstrap.py` | Idempotent CLI initialization |
| `UnifiedRenderer` | `rendering/service.py` | Single rendering implementation |
| `@cli_command` | `commands/decorators.py` | Declarative command binding |
| `OperationRegistry` | `execution/registry.py` | Single operation metadata store |
| `CommandExecutor` | `execution/executor.py` | Unified execution engine |

### 3.2 Target Module Structure

```
cli/
├── commands/
│   ├── decorators.py          # @cli_command decorator
│   ├── app.py                 # Root Cyclopts app
│   └── {domain}.py            # Command definitions (no __call__)
├── handlers/
│   ├── context.py             # HandlerContext (single context type)
│   └── {domain}.py            # Handler implementations
├── execution/
│   ├── bootstrap.py           # bootstrap_cli()
│   ├── executor.py            # CommandExecutor
│   └── registry.py            # OperationRegistry
├── rendering/
│   ├── service.py             # UnifiedRenderer
│   ├── table.py               # ColumnSpec, TableSpec
│   └── types.py               # OutputFormat, RenderContext
└── core/
    └── results.py             # CliResult
```

### 3.3 Files to be Deleted

The following files will be removed as part of the migration. Some have already been deleted in earlier phases:

| File | Superseded By | Status |
|------|---------------|--------|
| `rendering/renderers.py` | `rendering/service.py` | ✅ **Deleted in Phase 2** |
| `handlers/base.py` | `handlers/context.py` | Pending (Phase 6) |
| `handlers/protocol.py` | `handlers/context.py` | Pending (Phase 6) |
| `execution/context.py` | `handlers/context.py` | Pending (Phase 6) |
| `execution/adapter.py` | `commands/decorators.py` | Pending (Phase 6) |
| `commands/context.py` | `commands/decorators.py` (internals) | Pending (Phase 6) |
| `operations/build_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/dataset_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/docs_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/graph_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/history_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/ide_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/op_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/storage_operations.py` | Registration in handlers | Pending (Phase 6) |
| `operations/subsystem_operations.py` | Registration in handlers | Pending (Phase 6) |
| `introspection/registry.py` | `execution/registry.py` | Pending (Phase 6) |
| `_migration_flags.py` | N/A (temporary scaffolding) | Pending (Phase 6) |

---

## 4. Migration Strategy

### 4.1 Core Principles

1. **Build New Before Removing Old**
   - New components are created as additive changes
   - Consumers migrate incrementally
   - Old code deleted only after all consumers migrated

2. **Maintain Working System**
   - Every commit should pass all tests
   - No "big bang" rewrites
   - Feature flags for incremental rollout during transition

3. **Bottom-Up Dependency Order**
   - Build foundation (context, bootstrap) first
   - Then rendering (independent utility layer)
   - Then handlers (consumers of context)
   - Then commands (consumers of handlers)
   - Then cleanup (removal of old code)

4. **Per-File Granularity**
   - Handler files migrate independently
   - Command files migrate independently
   - Enables parallelization and easy rollback

### 4.2 Critical Path

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                CRITICAL PATH                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 0     Phase 1         Phase 2         Phase 3                        │
│  ┌─────┐    ┌─────────┐     ┌─────────┐     ┌─────────────┐                 │
│  │Prep │───▶│Foundation│────▶│Rendering│────▶│  Handlers   │                 │
│  └─────┘    └─────────┘     └─────────┘     └──────┬──────┘                 │
│                                                    │                         │
│                                                    ▼                         │
│                              Phase 4         Phase 5         Phase 6        │
│                             ┌─────────┐     ┌─────────┐     ┌─────────┐     │
│                             │Registry │────▶│Commands │────▶│ Cleanup │     │
│                             └─────────┘     └─────────┘     └─────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Parallelization Opportunities:**
- Phase 2 (Rendering) can overlap with early Phase 3 (first few handlers)
- Within Phase 3: Handler files can be migrated in parallel
- Within Phase 5: Command files can be migrated in parallel

### 4.3 Temporary Scaffolding

During migration, we employ temporary compatibility mechanisms that are **removed in Phase 6**:

| Scaffolding | Purpose | Created In | Removed In |
|-------------|---------|------------|------------|
| `_feature_flags.py` | Optional gating of new code paths | Phase 0 | Phase 6 |
| `HandlerContext.from_enhanced()` | Bridge old contexts to new | Phase 1 | Phase 6 |
| `commands/context.py` updates | Create `HandlerContext` instead of old type | Phase 3 | Phase 6 |

---

## 5. Phase 0: Preparation

### 5.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 1-2 days |
| **Risk Level** | Low |
| **Parallelizable** | No |
| **Dependencies** | None |

### 5.2 Objectives

1. Establish baseline metrics for test coverage and CLI functionality
2. Create inventory of all handlers, commands, and their dependencies
3. Set up temporary infrastructure for migration tracking
4. Validate all existing tests pass before any changes

### 5.3 Deliverables

#### 5.3.1 Test Baseline Report

Generate and document:
- Total test count for `tests/cli/`
- Coverage percentage for `src/codeintel/cli/`
- List of any currently failing/skipped tests
- CLI smoke test results

**Command:**
```bash
uv run pytest tests/cli/ -v --tb=short > baseline_test_report.txt
uv run pytest tests/cli/ --cov=src/codeintel/cli --cov-report=html:htmlcov_baseline
```

#### 5.3.2 Handler Inventory

Create `docs/plans/cli_migration_inventory.md` documenting:

| Handler File | Handlers | Context Type Used | Param Helpers | Dependencies |
|--------------|----------|-------------------|---------------|--------------|
| `jobs.py` | 5 | `EnhancedHandlerContext` | Local | None |
| `health.py` | 2 | `EnhancedHandlerContext` | Local | None |
| ... | ... | ... | ... | ... |

#### 5.3.3 Command Inventory

| Command File | Commands | Uses `command_context` | Has `__call__` | Runtime Required |
|--------------|----------|------------------------|----------------|------------------|
| `jobs.py` | 5 | Yes | Yes | No |
| `health.py` | 1 | Yes | Yes | No |
| ... | ... | ... | ... | ... |

#### 5.3.4 Feature Flag Infrastructure (Optional)

Create `cli/_feature_flags.py`:

```python
"""Temporary feature flags for migration. DELETE IN PHASE 6."""

from __future__ import annotations

import os

# Enable new HandlerContext in all code paths
USE_NEW_CONTEXT: bool = os.environ.get("CODEINTEL_CLI_V2_CONTEXT", "0") == "1"

# Enable new rendering stack everywhere
USE_UNIFIED_RENDERER: bool = os.environ.get("CODEINTEL_CLI_V2_RENDER", "0") == "1"
```

### 5.4 Entry Criteria

- Clean git working tree
- All CI checks passing on main branch

### 5.5 Exit Criteria

- [ ] Test baseline documented
- [ ] Handler inventory complete
- [ ] Command inventory complete
- [ ] Feature flag module created (if using)
- [ ] All existing tests still pass

### 5.6 Tasks

| ID | Task | Effort | Owner |
|----|------|--------|-------|
| P0-1 | Run full CLI test suite and capture baseline | 1h | |
| P0-2 | Generate coverage report for `cli/` | 1h | |
| P0-3 | Create handler inventory spreadsheet | 2h | |
| P0-4 | Create command inventory spreadsheet | 2h | |
| P0-5 | Create feature flag module | 1h | |
| P0-6 | Document any currently failing tests | 1h | |
| P0-7 | Create migration tracking document | 1h | |

---

## 6. Phase 1: Foundation Layer

### 6.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 3-4 days |
| **Risk Level** | Low |
| **Parallelizable** | Partially (1.1 and 1.2 can run in parallel) |
| **Dependencies** | Phase 0 complete |

### 6.2 Objectives

1. Create the new unified `HandlerContext` class
2. Implement centralized `bootstrap_cli()` function
3. Establish comprehensive test coverage for new components
4. Create adapter for gradual handler migration

### 6.3 Deliverables

#### 6.3.1 `handlers/context.py` — New HandlerContext

**File:** `src/codeintel/cli/handlers/context.py`

**Core Implementation:**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, TypeVar

if TYPE_CHECKING:
    from codeintel.cli.config.model import CliConfig
    from codeintel.cli.resolution.types import ResolvedRuntime


@dataclass(frozen=True)
class HandlerContextOptions:
    """Options for creating a HandlerContext.

    Bundle optional parameters to reduce argument count in factory functions.
    """

    output_format: OutputFormat = OutputFormat.TEXT
    verbosity: int = 0
    project_root: Path | None = None
    database_path: Path | None = None


@dataclass
class HandlerContext:
    """Unified context for all CLI handler operations."""

    # Core configuration
    config: CliConfig
    operation_id: str
    output_format: OutputFormat = OutputFormat.TEXT
    verbosity: int = 0

    # Runtime resolution parameters
    project_root: Path | None = None
    index_path: Path | None = None
    database_path: Path | None = None

    # Internal state (use `object` not `Any` to avoid type issues)
    _params: dict[str, object] = field(default_factory=dict, repr=False)
    _runtime: ResolvedRuntime | None = field(default=None, repr=False)
    _gateway: StorageGateway | None = field(default=None, repr=False)
    _graph_runtime: GraphRuntime | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing resources."""
        self.close()
```

**Required Methods:**

| Method | Purpose |
|--------|---------|
| `param_str(key, default)` | Get string parameter |
| `param_int(key, default)` | Get integer parameter |
| `param_bool(key, *, default)` | Get boolean parameter (keyword-only default) |
| `param_path(key, default)` | Get Path parameter |
| `param_enum(key, enum_type, default)` | Get enum parameter |
| `param_list(key, default)` | Get list[str] parameter |
| `param_tuple(key, default)` | Get tuple[str, ...] parameter |
| `require_str(key)` | Get required string (raises `ParameterError`) |
| `require_int(key)` | Get required integer (raises `ParameterError`) |
| `require_path(key)` | Get required path (raises `ParameterError`) |
| `runtime` | Property: lazy-load `ResolvedRuntime` |
| `gateway` | Property: lazy-load `StorageGateway` |
| `graph_runtime` | Property: lazy-load `GraphRuntime` |
| `logger` | Property: operation-specific logger |
| `db_path` | Property: database path from runtime or fallback |
| `color_enabled` | Property: check if color output is enabled |
| `close()` | Clean up resources (idempotent) |
| `__enter__` / `__exit__` | Context manager protocol |

**Helper Module for Lazy Imports:**

**File:** `src/codeintel/cli/handlers/_lazy_resources.py`

```python
"""Lazy resource loading helpers to avoid circular imports."""

from __future__ import annotations

from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.resolution.runtime import RuntimeResolver


def lazy_resolve_runtime(
    operation_id: str,
    params: dict[str, object],
    project_root: Path | None,
    database_path: Path | None,
) -> ResolvedRuntime:
    """Resolve runtime from handler context parameters."""
    exec_params: dict[str, object] = dict(params)
    if project_root is not None:
        exec_params["project_root"] = project_root
    if database_path is not None:
        exec_params["db_path"] = database_path

    exec_ctx = ExecutionContext(
        operation_id=operation_id,
        params=exec_params,
    )
    return RuntimeResolver.resolve(exec_ctx)
```

**Adapter Methods (temporary, remove in Phase 6):**

```python
@classmethod
def from_enhanced_context(
    cls,
    ctx: EnhancedHandlerContext,
    operation_id: str,
    params: dict[str, object] | None = None,
) -> HandlerContext:
    """Create HandlerContext from legacy EnhancedHandlerContext.

    Raises
    ------
    TypeError
        If ctx is not an EnhancedHandlerContext instance.
    """
```

#### 6.3.2 `execution/bootstrap.py` — CLI Bootstrap

**File:** `src/codeintel/cli/execution/bootstrap.py`

**Core Implementation:**

```python
from __future__ import annotations

import logging
import signal
import sys
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.cli.config import load_config as load_cli_config

if TYPE_CHECKING:
    from types import FrameType
    from codeintel.cli.config.model import CliConfig

LOG = logging.getLogger(__name__)

VERBOSITY_DEBUG = 2
VERBOSITY_INFO = 1


@dataclass
class _BootstrapState:
    """Internal state for bootstrap management.

    Use dataclass instead of global variables to avoid PLW0603 linting errors.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    complete: bool = False
    config: CliConfig | None = None


# Module-level state instance (singleton)
_state = _BootstrapState()


def bootstrap_cli(
    verbosity: int = 0,
    config: CliConfig | None = None,
) -> CliConfig:
    """Initialize CLI subsystems exactly once.

    Idempotent: safe to call multiple times.
    Thread-safe: uses lock for concurrent access.

    Initializes:

    - Logging configuration based on verbosity
    - Signal handlers for graceful shutdown (SIGINT, SIGTERM)

    Parameters
    ----------
    verbosity
        Logging verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
    config
        Optional pre-loaded configuration. If None, loads from environment.

    Returns
    -------
    CliConfig
        The active CLI configuration.
    """
    # Fast path for already initialized
    if _state.complete:
        if _state.config is not None:
            return _state.config
        return load_cli_config(validate=False)

    with _state.lock:
        # Double-check after acquiring lock
        if _state.complete and _state.config is not None:
            return _state.config

        active_config = config if config is not None else load_cli_config(validate=False)
        _configure_logging(verbosity, active_config)
        _register_signal_handlers()

        _state.config = active_config
        _state.complete = True

        LOG.debug("CLI bootstrap complete (verbosity=%d)", verbosity)
        return active_config
```

**Responsibilities:**
1. Load configuration (if not provided)
2. Configure logging based on verbosity
3. Register signal handlers for graceful shutdown
4. Cache config in `_BootstrapState` dataclass (avoids global statements)

#### 6.3.3 Test Coverage

**New Test Files:**

| File | Coverage Target |
|------|-----------------|
| `tests/cli/handlers/test_context.py` | `HandlerContext` unit tests |
| `tests/cli/execution/test_bootstrap.py` | `bootstrap_cli()` unit tests |

**Test Cases for HandlerContext:**

- [ ] Construction with minimal parameters
- [ ] Construction with all parameters
- [ ] `param_str` with default
- [ ] `param_str` without default (returns None)
- [ ] `param_int` with conversion
- [ ] `param_int` with invalid value (error)
- [ ] `param_bool` with various truthy/falsy values
- [ ] `param_path` resolves relative paths
- [ ] `require_str` raises on missing
- [ ] `require_int` raises on missing
- [ ] Lazy `runtime` property creates runtime once
- [ ] Lazy `gateway` property creates gateway once
- [ ] `close()` cleans up resources
- [ ] Context manager protocol works
- [ ] `from_enhanced_context` adapter works

**Test Cases for bootstrap_cli:**

- [ ] First call initializes and returns config
- [ ] Second call is no-op and returns same config
- [ ] Thread-safety under concurrent calls
- [ ] Verbosity levels affect logging
- [ ] Custom config passed through

### 6.4 Entry Criteria

- Phase 0 complete
- Handler inventory available
- Test baseline established

### 6.5 Exit Criteria

- [ ] `handlers/context.py` implemented
- [ ] `execution/bootstrap.py` implemented
- [ ] Unit tests passing with >90% coverage
- [ ] All existing CLI tests still pass (no changes to them yet)
- [ ] Documentation strings complete
- [ ] Type checking clean (pyright, pyrefly)
- [ ] Linting clean (ruff)

### 6.6 Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| P1-1 | Create `handlers/context.py` skeleton | 2h | - |
| P1-2 | Implement param accessor methods | 4h | P1-1 |
| P1-3 | Implement lazy resource properties | 4h | P1-1 |
| P1-4 | Implement context manager protocol | 2h | P1-3 |
| P1-5 | Implement `from_enhanced_context` adapter | 2h | P1-4 |
| P1-6 | Create `execution/bootstrap.py` | 4h | - |
| P1-7 | Write unit tests for `HandlerContext` | 4h | P1-5 |
| P1-8 | Write unit tests for `bootstrap_cli` | 2h | P1-6 |
| P1-9 | Integration test: new context in isolation | 2h | P1-7, P1-8 |
| P1-10 | Code review and refinement | 2h | P1-9 |

### 6.7 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Resource lifecycle differs from old context | Medium | Medium | Extensive testing, adapter maintains compatibility |
| Lazy loading edge cases | Low | Low | Test with and without resource access |
| Thread-safety issues in bootstrap | Low | High | Use threading.Lock, test concurrent calls |

---

## 7. Phase 2: Rendering Consolidation

### 7.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2-3 days |
| **Risk Level** | Medium |
| **Parallelizable** | No |
| **Dependencies** | Phase 1 complete (or can overlap with early P1) |

### 7.2 Objectives

1. Ensure `UnifiedRenderer` has all capabilities needed
2. Migrate all rendering consumers to use `UnifiedRenderer`
3. Delete `rendering/renderers.py`
4. Eliminate duplicate type definitions

### 7.3 Current Rendering Landscape

**`rendering/service.py` (UnifiedRenderer):**
- JSON output
- JSONL output
- Rich table output
- Plain text output
- Uses `RenderContext`, `TableSpec`

**`rendering/renderers.py` (to be deleted):**
- `RichRenderer` class
- `PlainRenderer` class
- `get_renderer()` factory function
- `render_cli_result()` helper
- Duplicate `ColumnSpec`, `TableSpec`
- Duplicate `CODEINTEL_THEME`

### 7.4 Deliverables

#### 7.4.1 Audit and Gap Analysis

Document any capabilities in `renderers.py` not present in `service.py`:

| Capability | In `renderers.py` | In `service.py` | Action |
|------------|-------------------|-----------------|--------|
| JSON rendering | ✓ | ✓ | None |
| JSONL rendering | ✓ | ✓ | None |
| Rich tables | ✓ | ✓ | Verify feature parity |
| Plain text tables | ✓ | ✓ | Verify feature parity |
| Error rendering | ✓ | ✓ | Verify feature parity |
| Progress bars | ✓ | ? | Add if missing |
| `get_renderer()` | ✓ | ✗ | Add to service.py |
| `render_cli_result()` | ✓ | ✗ | Add to service.py |

#### 7.4.2 Update `rendering/service.py`

Add missing functions to achieve API compatibility:

```python
from __future__ import annotations

import sys
from typing import TextIO


def get_renderer(
    output_format: OutputFormat = OutputFormat.TEXT,
    *,
    color: bool | None = None,
    writer: TextIO | None = None,
    err_writer: TextIO | None = None,
) -> UnifiedRenderer:
    """Get a renderer for the specified output format.

    Factory function that creates UnifiedRenderer instances with appropriate
    settings based on output format, environment, and TTY detection.

    Parameters
    ----------
    output_format
        Desired output format (TEXT, JSON, or JSONL).
    color
        Override color detection. If None, auto-detect based on TTY.
    writer
        Output stream (defaults to sys.stdout).
    err_writer
        Error stream (defaults to sys.stderr).

    Returns
    -------
    UnifiedRenderer
        Configured renderer instance.
    """
    if writer is not None or err_writer is not None:
        is_tty = (writer or sys.stdout).isatty()
        use_color = color if color is not None else (is_tty and output_format == OutputFormat.TEXT)
        ctx = RenderContext(
            format=output_format,
            color=use_color,
            writer=writer or sys.stdout,
            err_writer=err_writer or sys.stderr,
            is_tty=is_tty,
        )
    else:
        ctx = RenderContext.auto_detect(
            format_override=output_format,
            color_override=color,
        )
    return UnifiedRenderer(ctx)


def render_cli_result[T](
    result: CliResult[T],
    renderer: UnifiedRenderer | None = None,
    *,
    table_spec: TableSpec | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
) -> int:
    """Render a CliResult and return exit code.

    Convenience function that creates a renderer if not provided and renders
    the result appropriately. Uses PEP 695 type parameter syntax.
    """
    if renderer is None:
        renderer = get_renderer(output_format)

    if table_spec is not None and result.success and isinstance(result.data, list):
        renderer.render_table(result.data, table_spec)
        return 0

    return renderer.render_result(result)
```

**Note:** `specs.py` already existed with pre-built table specifications — no new file needed.

#### 7.4.3 Migrate Consumers

**Files to Update:**

| File | Current Import | New Import |
|------|----------------|------------|
| `execution/executor.py` | `from .renderers import ...` | `from .service import ...` |
| `execution/adapter.py` | `from .renderers import ...` | `from .service import ...` |

#### 7.4.4 Delete `rendering/renderers.py`

After all consumers migrated:
1. Delete the file
2. Remove from `rendering/__init__.py` exports
3. Update any remaining imports (search codebase)

#### 7.4.5 Consolidate Type Definitions

Ensure single source of truth:

| Type | Canonical Location | Action |
|------|-------------------|--------|
| `ColumnSpec` | `rendering/table.py` | Keep |
| `TableSpec` | `rendering/table.py` | Keep |
| `OutputFormat` | `rendering/types.py` | Keep |
| `RenderContext` | `rendering/types.py` | Keep |
| `CODEINTEL_THEME` | `rendering/service.py` | Keep single definition |

### 7.5 Entry Criteria

- Phase 1 substantially complete (new context exists)
- Rendering audit complete

### 7.6 Exit Criteria

- [ ] `UnifiedRenderer` has all required capabilities
- [ ] `get_renderer()` available in `service.py`
- [ ] `render_cli_result()` available in `service.py`
- [ ] `execution/executor.py` uses `service.py`
- [ ] `execution/adapter.py` uses `service.py`
- [ ] `rendering/renderers.py` deleted
- [ ] No duplicate `ColumnSpec`/`TableSpec` definitions
- [ ] All tests pass
- [ ] All CLI commands render correctly

### 7.7 Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| P2-1 | Audit `renderers.py` vs `service.py` capabilities | 2h | - |
| P2-2 | Add missing functions to `service.py` | 4h | P2-1 |
| P2-3 | Update `execution/executor.py` imports | 2h | P2-2 |
| P2-4 | Update `execution/adapter.py` imports | 2h | P2-2 |
| P2-5 | Search for any other consumers | 1h | P2-4 |
| P2-6 | Delete `rendering/renderers.py` | 1h | P2-5 |
| P2-7 | Update `rendering/__init__.py` | 1h | P2-6 |
| P2-8 | Verify all CLI commands work | 2h | P2-7 |
| P2-9 | Run full test suite | 1h | P2-8 |

### 7.8 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Feature gap in `UnifiedRenderer` | Medium | Medium | Thorough audit before migration |
| Subtle rendering differences | Medium | Low | Visual inspection of common outputs |
| Missed consumer | Low | Medium | Grep for all imports before deletion |

---

## 8. Phase 3: Handler Migration

### 8.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 5-7 days |
| **Risk Level** | Medium |
| **Parallelizable** | Yes (per handler file) |
| **Dependencies** | Phase 1 complete, Phase 2 substantially complete |

### 8.2 Objectives

1. Migrate all handler files to use new `HandlerContext`
2. Remove all local `_get_*_param` helper functions
3. Standardize handler signatures
4. Update `commands/context.py` to create new context type

### 8.3 Handler Migration Pattern

**Before:**
```python
from codeintel.cli.handlers.protocol import EnhancedHandlerContext

def _get_str_param(ctx: EnhancedHandlerContext, key: str, default: str | None = None) -> str | None:
    value = ctx.params.get(key)
    return str(value) if value is not None else default

def _get_int_param(ctx: EnhancedHandlerContext, key: str, default: int = 0) -> int:
    value = ctx.params.get(key)
    if value is None:
        return default
    return int(value)

def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:
    name = _get_str_param(ctx, "name")
    limit = _get_int_param(ctx, "limit", 20)
    # ... handler logic
```

**After:**
```python
from codeintel.cli.handlers.context import HandlerContext

def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
    name = ctx.param_str("name")
    limit = ctx.param_int("limit", 20)
    # ... handler logic
```

### 8.4 Migration Order

Handlers are migrated in order of increasing complexity and dependency:

#### Tier 1: No Runtime Required (Days 1-2)

| File | Handlers | Complexity | Notes |
|------|----------|------------|-------|
| `handlers/jobs.py` | 5 | Low | Background job management |
| `handlers/health.py` | 2 | Low | Health checks |

#### Tier 2: Runtime Required, Simple (Days 2-3)

| File | Handlers | Complexity | Notes |
|------|----------|------------|-------|
| `handlers/ops.py` | 8 | Medium | Operation management |
| `handlers/storage.py` | 3 | Medium | Storage inspection |

#### Tier 3: Runtime Required, Medium (Days 3-4)

| File | Handlers | Complexity | Notes |
|------|----------|------------|-------|
| `handlers/history.py` | 1+ | Medium | History queries |
| `handlers/build.py` | Multiple | Medium | Build operations |
| `handlers/docs.py` | Multiple | Medium | Documentation generation |

#### Tier 4: Full Resource Access (Days 4-5)

| File | Handlers | Complexity | Notes |
|------|----------|------------|-------|
| `handlers/graphs.py` | Multiple | Higher | Uses `graph_runtime` |
| `handlers/ide.py` | Multiple | Medium | IDE integration |
| `handlers/datasets.py` | Multiple | Medium | Dataset management |

#### Tier 5: Plugin/Extension (Days 5-6)

| File | Handlers | Complexity | Notes |
|------|----------|------------|-------|
| `handlers/plugins.py` | Multiple | Medium | Plugin management |
| `handlers/subsystem.py` | Multiple | Medium | Subsystem operations |

### 8.5 Per-Handler-File Checklist

For each handler file:

- [ ] Update import: `from codeintel.cli.handlers.context import HandlerContext`
- [ ] Remove import: `from codeintel.cli.handlers.protocol import EnhancedHandlerContext`
- [ ] Update all handler signatures: `ctx: HandlerContext`
- [ ] Replace `_get_str_param(ctx, key)` → `ctx.param_str(key)`
- [ ] Replace `_get_int_param(ctx, key, default)` → `ctx.param_int(key, default)`
- [ ] Replace `_get_bool_param(ctx, key, default)` → `ctx.param_bool(key, default)`
- [ ] Replace `_require_str_param(ctx, key)` → `ctx.require_str(key)`
- [ ] Replace `_get_path_param(ctx, key)` → `ctx.param_path(key)`
- [ ] Replace `ctx.params.get(key)` → appropriate `ctx.param_*()` method
- [ ] Delete all local `_get_*_param` function definitions
- [ ] Update any `setup_logging` calls if present
- [ ] Run tests for this handler file
- [ ] Verify lint/type checks pass

### 8.6 Update commands/context.py

Early in Phase 3, update `commands/context.py` to create `HandlerContext`:

```python
# Before
from codeintel.cli.handlers.protocol import EnhancedHandlerContext

# ... in command_context() ...
ctx = EnhancedHandlerContext(...)

# After
from codeintel.cli.handlers.context import HandlerContext

# ... in command_context() ...
ctx = HandlerContext(
    config=config,
    operation_id=operation_id,
    output_format=output_format,
    verbosity=verbosity,
    project_root=project_root,
    index_path=index_path,
    database_path=database_path,
    _params=combined_params,
)
```

### 8.7 Entry Criteria

- Phase 1 complete (`HandlerContext` exists and tested)
- Phase 2 substantially complete (rendering consolidated)

### 8.8 Exit Criteria

- [ ] All handler files migrated
- [ ] No `_get_*_param` functions remain in handlers
- [ ] All handlers use `HandlerContext` type
- [ ] `commands/context.py` creates `HandlerContext`
- [ ] All handler tests pass
- [ ] All CLI integration tests pass
- [ ] Type checking clean
- [ ] Linting clean

### 8.9 Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| P3-1 | Update `commands/context.py` to create `HandlerContext` | 2h | Phase 1 |
| P3-2 | Migrate `handlers/jobs.py` | 2h | P3-1 |
| P3-3 | Migrate `handlers/health.py` | 2h | P3-1 |
| P3-4 | Migrate `handlers/ops.py` | 3h | P3-1 |
| P3-5 | Migrate `handlers/storage.py` | 2h | P3-1 |
| P3-6 | Migrate `handlers/history.py` | 2h | P3-1 |
| P3-7 | Migrate `handlers/build.py` | 3h | P3-1 |
| P3-8 | Migrate `handlers/docs.py` | 3h | P3-1 |
| P3-9 | Migrate `handlers/graphs.py` | 4h | P3-1 |
| P3-10 | Migrate `handlers/ide.py` | 3h | P3-1 |
| P3-11 | Migrate `handlers/datasets.py` | 3h | P3-1 |
| P3-12 | Migrate `handlers/plugins.py` | 2h | P3-1 |
| P3-13 | Migrate `handlers/subsystem.py` | 2h | P3-1 |
| P3-14 | Full test suite validation | 2h | P3-2 through P3-13 |
| P3-15 | Code review and cleanup | 2h | P3-14 |

### 8.10 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Param type conversion differences | Medium | Medium | Comprehensive param accessor tests |
| Resource lifecycle changes | Medium | Medium | Extensive integration testing |
| Missed param access patterns | Low | Low | Grep for `ctx.params` after migration |

---

## 9. Phase 4: Registry Unification

### 9.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2-3 days |
| **Risk Level** | Low |
| **Parallelizable** | No |
| **Dependencies** | Phase 3 complete |

### 9.2 Objectives

1. Create unified `OperationRegistry` in execution layer
2. Migrate operation registrations from `operations/*.py`
3. Update introspection to use new registry
4. Prepare registry for `@cli_command` integration

### 9.3 Deliverables

#### 9.3.1 `execution/registry.py`

**New File:** `src/codeintel/cli/execution/registry.py`

**Core Implementation:**

```python
@dataclass(frozen=True)
class OperationSpec:
    """Specification for a CLI operation."""
    
    operation_id: str
    name: str
    description: str
    handler: Callable[[HandlerContext], CliResult[Any]]
    group: str
    
    # Metadata
    require_runtime: bool = True
    require_gateway: bool = True
    tags: tuple[str, ...] = ()
    
    # Parameter schema (for validation and help)
    parameters: tuple[ParameterSpec, ...] = ()


class OperationRegistry:
    """Central registry for all CLI operations."""
    
    def __init__(self) -> None:
        self._operations: dict[str, OperationSpec] = {}
    
    def register(self, spec: OperationSpec) -> None:
        """Register an operation specification."""
    
    def get(self, operation_id: str) -> OperationSpec | None:
        """Get operation by ID."""
    
    def list_operations(self, group: str | None = None) -> list[OperationSpec]:
        """List all operations, optionally filtered by group."""
    
    def execute(
        self,
        operation_id: str,
        ctx: HandlerContext,
    ) -> CliResult[Any]:
        """Execute an operation by ID."""


# Global singleton
_REGISTRY: OperationRegistry | None = None

def get_registry() -> OperationRegistry:
    """Get the global operation registry."""

def register_operation(spec: OperationSpec) -> None:
    """Register an operation with the global registry."""
```

#### 9.3.2 Operation Registration

Operations will be registered in their handler modules:

```python
# handlers/jobs.py

from codeintel.cli.execution.registry import register_operation, OperationSpec

register_operation(OperationSpec(
    operation_id="jobs.list",
    name="List Jobs",
    description="List background jobs with optional status filtering",
    handler=jobs_list_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
    parameters=(
        ParameterSpec(name="status", type=str, required=False),
        ParameterSpec(name="limit", type=int, default=20),
    ),
))
```

#### 9.3.3 Introspection Update

Update `introspection/__init__.py`:

```python
# Re-export from new location
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
)
```

### 9.4 Entry Criteria

- Phase 3 complete (all handlers migrated)
- Handler signatures stable

### 9.5 Exit Criteria

- [ ] `execution/registry.py` implemented
- [ ] All operations registered in new registry
- [ ] `introspection/__init__.py` re-exports from new registry
- [ ] Help system works with new registry
- [ ] All tests pass

### 9.6 Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| P4-1 | Create `execution/registry.py` | 4h | - |
| P4-2 | Define `OperationSpec` and `ParameterSpec` | 2h | P4-1 |
| P4-3 | Implement `OperationRegistry` class | 4h | P4-2 |
| P4-4 | Add registrations to handler modules | 4h | P4-3 |
| P4-5 | Update `introspection/__init__.py` | 1h | P4-4 |
| P4-6 | Verify help system works | 2h | P4-5 |
| P4-7 | Write registry unit tests | 2h | P4-3 |
| P4-8 | Integration testing | 2h | P4-6 |

### 9.7 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Help system breaks | Medium | Medium | Test help output before and after |
| Missing operation registrations | Low | Medium | Compare against old registry |

---

## 10. Phase 5: Command Decorator & Migration

### 10.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 5-7 days |
| **Risk Level** | Medium |
| **Parallelizable** | Yes (per command file) |
| **Dependencies** | Phase 4 complete |

### 10.2 Objectives

1. Create `@cli_command` decorator
2. Migrate all command files to use decorator
3. Eliminate manual `__call__` boilerplate
4. Auto-register operations with registry

### 10.3 Deliverables

#### 10.3.1 `commands/decorators.py`

**New File:** `src/codeintel/cli/commands/decorators.py`

**Core Implementation:**

```python
def cli_command(
    operation_id: str,
    *,
    handler: Callable[[HandlerContext], CliResult[Any]],
    require_runtime: bool = True,
    require_gateway: bool = True,
    description: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator for CLI command dataclasses.
    
    Generates __call__ method that:
    1. Extracts parameters from dataclass fields
    2. Calls bootstrap_cli()
    3. Creates HandlerContext
    4. Invokes handler
    5. Renders result
    6. Handles exit code
    
    Also registers operation with OperationRegistry.
    
    Parameters
    ----------
    operation_id
        Unique identifier for this operation (e.g., "jobs.list")
    handler
        Handler function to invoke
    require_runtime
        Whether handler needs ResolvedRuntime
    require_gateway
        Whether handler needs StorageGateway
    description
        Operation description (defaults to docstring)
    
    Examples
    --------
    >>> @cli_command("jobs.list", handler=jobs_list_handler, require_runtime=False)
    ... @jobs_app.command(name="list")
    ... @dataclass
    ... class JobsListCommand:
    ...     status: Annotated[str | None, Parameter(...)] = None
    ...     limit: Annotated[int, Parameter(...)] = 20
    ...     output_format: Annotated[OutputFormat, Parameter(...)] = OutputFormat.TEXT
    ...     verbose: Annotated[int, Parameter(name="-v", count=True)] = 0
    """
```

**Generated `__call__` behavior:**

```python
def __call__(self) -> None:
    # 1. Bootstrap
    config = bootstrap_cli(verbosity=self.verbose)
    
    # 2. Extract params from dataclass fields
    params = _extract_params_from_dataclass(self)
    
    # 3. Resolve runtime paths
    project_root, index_path, database_path = _resolve_paths(self, config)
    
    # 4. Create context
    ctx = HandlerContext(
        config=config,
        operation_id=operation_id,
        output_format=self.output_format,
        verbosity=self.verbose,
        project_root=project_root,
        index_path=index_path,
        database_path=database_path,
        _params=params,
    )
    
    # 5. Execute handler
    with ctx:
        result = handler(ctx)
    
    # 6. Render and exit
    renderer = get_renderer(self.output_format)
    exit_code = renderer.render_result(result)
    if exit_code != 0:
        sys.exit(exit_code)
```

#### 10.3.2 Command Migration Pattern

**Before:**
```python
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    status: Annotated[str | None, Parameter(name="--status")] = None
    limit: Annotated[int, Parameter(name="--limit")] = 20
    output_format: Annotated[OutputFormat, Parameter(...)] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name=["-v", "--verbose"], count=True)] = 0
    project: Annotated[Path | None, Parameter(name="--project")] = None

    def __call__(self) -> None:
        runtime_cli = RuntimeCLI(project=self.project)
        output_cli = OutputFormatCLI(output_format=self.output_format)
        params = {"status": self.status, "limit": self.limit}
        
        with command_context(
            operation_id="jobs.list",
            verbose=self.verbose,
            runtime_params=runtime_cli,
            output_params=output_cli,
            extra_params=params,
        ) as (ctx, renderer):
            result = jobs_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)
```

**After:**
```python
@cli_command("jobs.list", handler=jobs_list_handler, require_runtime=False)
@jobs_app.command(name="list")
@dataclass
class JobsListCommand:
    """List background jobs."""
    status: Annotated[str | None, Parameter(name="--status")] = None
    limit: Annotated[int, Parameter(name="--limit")] = 20
    output_format: Annotated[OutputFormat, Parameter(...)] = OutputFormat.TEXT
    verbose: Annotated[int, Parameter(name=["-v", "--verbose"], count=True)] = 0
    project: Annotated[Path | None, Parameter(name="--project")] = None
    # NO __call__ method - decorator generates it
```

### 10.4 Migration Order

Commands are migrated in the same order as their corresponding handlers:

| Priority | File | Commands | Notes |
|----------|------|----------|-------|
| 1 | `commands/jobs.py` | 5 | POC - simplest |
| 2 | `commands/health.py` | 1 | Simple |
| 3 | `commands/ops.py` | Multiple | |
| 4 | `commands/storage.py` | Multiple | |
| 5 | `commands/history.py` | Multiple | |
| 6 | `commands/build.py` | Multiple | |
| 7 | `commands/graphs.py` | Multiple | |
| 8 | `commands/docs.py` | Multiple | |
| 9 | `commands/ide.py` | Multiple | |
| 10 | `commands/datasets.py` | Multiple | |
| 11 | `commands/dataset_ops.py` | Multiple | |
| 12 | `commands/plugins.py` | Multiple | |
| 13 | `commands/subsystem.py` | Multiple | |
| 14 | `commands/serve.py` | Multiple | |
| 15 | `commands/config.py` | Multiple | |
| 16 | `commands/completions.py` | Multiple | |

### 10.5 Per-Command-File Checklist

For each command file:

- [ ] Add import: `from codeintel.cli.commands.decorators import cli_command`
- [ ] Remove imports: `RuntimeCLI`, `OutputFormatCLI` (if no longer needed)
- [ ] Remove import: `command_context` (if no longer needed)
- [ ] For each command class:
  - [ ] Add `@cli_command(...)` decorator above `@app.command()`
  - [ ] Ensure `output_format` field exists
  - [ ] Ensure `verbose` field exists
  - [ ] Delete `__call__` method
- [ ] Run tests for this command file
- [ ] Verify CLI commands work manually
- [ ] Verify lint/type checks pass

### 10.6 Entry Criteria

- Phase 4 complete (registry unified)
- Handler signatures finalized

### 10.7 Exit Criteria

- [ ] `commands/decorators.py` implemented
- [ ] All command files migrated
- [ ] No manual `__call__` methods in command classes
- [ ] All CLI commands work correctly
- [ ] All tests pass
- [ ] Operation auto-registration working

### 10.8 Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| P5-1 | Create `commands/decorators.py` | 8h | Phase 4 |
| P5-2 | Implement param extraction from dataclass | 4h | P5-1 |
| P5-3 | Implement `__call__` generation | 4h | P5-2 |
| P5-4 | Implement auto-registration | 2h | P5-3 |
| P5-5 | Migrate `commands/jobs.py` (POC) | 2h | P5-4 |
| P5-6 | Verify POC works end-to-end | 2h | P5-5 |
| P5-7 | Migrate remaining command files | 12h | P5-6 |
| P5-8 | Full CLI smoke test | 2h | P5-7 |
| P5-9 | Full test suite | 2h | P5-8 |
| P5-10 | Documentation update | 2h | P5-9 |

### 10.9 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Decorator complexity | Medium | Medium | Start with simplest commands |
| Param extraction edge cases | Medium | Medium | Comprehensive testing |
| Cyclopts compatibility issues | Low | High | Test decorator + Cyclopts interaction thoroughly |

---

## 11. Phase 6: Legacy Cleanup

### 11.1 Overview

| Attribute | Value |
|-----------|-------|
| **Duration** | 2-3 days |
| **Risk Level** | Low |
| **Parallelizable** | Partially |
| **Dependencies** | Phase 5 complete |

### 11.2 Objectives

1. Delete all superseded files
2. Remove temporary scaffolding
3. Finalize module structure
4. Update documentation
5. Final validation

### 11.3 Files to Delete

#### 11.3.1 Old Context Types

| File | Reason |
|------|--------|
| `handlers/base.py` | Superseded by `handlers/context.py` |
| `handlers/protocol.py` | Superseded by `handlers/context.py` |
| `execution/context.py` | Superseded by `handlers/context.py` |

#### 11.3.2 Old Command Infrastructure

| File | Reason |
|------|--------|
| `commands/context.py` | Superseded by decorator internals |
| `execution/adapter.py` | Superseded by `commands/decorators.py` |

#### 11.3.3 Operation Placeholders

| File | Reason |
|------|--------|
| `operations/build_operations.py` | Operations registered in handlers |
| `operations/dataset_operations.py` | Operations registered in handlers |
| `operations/docs_operations.py` | Operations registered in handlers |
| `operations/graph_operations.py` | Operations registered in handlers |
| `operations/history_operations.py` | Operations registered in handlers |
| `operations/ide_operations.py` | Operations registered in handlers |
| `operations/op_operations.py` | Operations registered in handlers |
| `operations/storage_operations.py` | Operations registered in handlers |
| `operations/subsystem_operations.py` | Operations registered in handlers |

#### 11.3.4 Old Registry

| File | Reason |
|------|--------|
| `introspection/registry.py` | Moved to `execution/registry.py` |

#### 11.3.5 Temporary Scaffolding

| File | Reason |
|------|--------|
| `_feature_flags.py` | Migration complete |

### 11.4 Deliverables

#### 11.4.1 File Deletion

Execute deletions in this order:

1. **Verify no imports remain**
   ```bash
   rg "from codeintel.cli.handlers.base import" src/
   rg "from codeintel.cli.handlers.protocol import" src/
   rg "from codeintel.cli.execution.context import" src/
   rg "from codeintel.cli.commands.context import" src/
   rg "from codeintel.cli.execution.adapter import" src/
   rg "from codeintel.cli.introspection.registry import" src/
   ```

2. **Delete files**

3. **Update `__init__.py` files**
   - `handlers/__init__.py`: Remove exports of deleted modules
   - `execution/__init__.py`: Remove exports of deleted modules
   - `commands/__init__.py`: Remove exports of deleted modules
   - `operations/__init__.py`: Update or delete if empty
   - `introspection/__init__.py`: Update re-exports

#### 11.4.2 Code Cleanup

- Remove any remaining `from_enhanced_context` calls
- Remove any feature flag conditionals
- Remove any compatibility imports

#### 11.4.3 Documentation Update

- Update any internal docs referencing deleted modules
- Update CLI developer guide (if exists)
- Archive this migration plan

#### 11.4.4 Final Validation

1. **Full test suite**
   ```bash
   uv run pytest tests/cli/ -v
   ```

2. **Type checking**
   ```bash
   uv run pyright --warnings --pythonversion=3.13
   uv run pyrefly check
   ```

3. **Linting**
   ```bash
   uv run ruff check --fix
   ```

4. **CLI smoke test**
   - Test each major command group
   - Verify help text renders
   - Verify JSON/text output modes

5. **Coverage comparison**
   - Compare to Phase 0 baseline
   - Ensure no coverage regression

### 11.5 Entry Criteria

- Phase 5 complete
- All command files migrated
- All tests passing

### 11.6 Exit Criteria

- [ ] All listed files deleted
- [ ] No imports of deleted modules anywhere
- [ ] All `__init__.py` files updated
- [ ] No feature flag code remains
- [ ] Full test suite passes
- [ ] Type checking clean
- [ ] Linting clean
- [ ] CLI smoke test passes
- [ ] Coverage >= baseline

### 11.7 Tasks

| ID | Task | Effort | Dependencies |
|----|------|--------|--------------|
| P6-1 | Verify no imports of files to delete | 2h | Phase 5 |
| P6-2 | Delete old context types | 1h | P6-1 |
| P6-3 | Delete old command infrastructure | 1h | P6-1 |
| P6-4 | Delete operation placeholder files | 1h | P6-1 |
| P6-5 | Delete old registry | 1h | P6-1 |
| P6-6 | Delete feature flags module | 0.5h | P6-1 |
| P6-7 | Update all `__init__.py` files | 2h | P6-2 through P6-6 |
| P6-8 | Remove compatibility code | 2h | P6-7 |
| P6-9 | Full test suite | 1h | P6-8 |
| P6-10 | Type checking | 1h | P6-8 |
| P6-11 | Linting | 0.5h | P6-8 |
| P6-12 | CLI smoke test | 1h | P6-9 |
| P6-13 | Coverage comparison | 1h | P6-9 |
| P6-14 | Documentation updates | 2h | P6-13 |
| P6-15 | Final code review | 2h | P6-14 |

### 11.8 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missed import | Low | Medium | Thorough grep before deletion |
| Test pollution from deleted modules | Low | Low | Run tests after each deletion |

---

## 12. Risk Management

### 12.1 Risk Registry

| ID | Risk | Phase | Likelihood | Impact | Mitigation | Owner |
|----|------|-------|------------|--------|------------|-------|
| R1 | Resource lifecycle differs between contexts | P1, P3 | Medium | Medium | Extensive testing, adapter pattern | |
| R2 | Rendering feature gap | P2 | Medium | Medium | Thorough audit before migration | |
| R3 | Handler param type conversion issues | P3 | Medium | Medium | Comprehensive param tests | |
| R4 | Cyclopts decorator incompatibility | P5 | Low | High | Early POC, consult docs | |
| R5 | Help system breaks | P4 | Medium | Medium | Before/after output comparison | |
| R6 | Test coverage regression | All | Low | High | Continuous coverage tracking | |
| R7 | Performance degradation | P1, P3 | Low | Medium | Benchmark critical paths | |

### 12.2 Risk Response Matrix

| Response | When to Use |
|----------|-------------|
| **Avoid** | Change approach to eliminate risk |
| **Mitigate** | Take actions to reduce likelihood/impact |
| **Transfer** | Assign risk to another party |
| **Accept** | Acknowledge and monitor |

---

## 13. Testing Strategy

### 13.1 Test Levels

| Level | Scope | When Run |
|-------|-------|----------|
| Unit | Individual functions/methods | Every commit |
| Integration | Module interactions | Every PR |
| CLI Smoke | End-to-end commands | Every phase completion |
| Regression | Full suite | Every phase completion |

### 13.2 Test Additions

Each phase adds specific tests:

| Phase | New Tests |
|-------|-----------|
| P1 | `test_context.py`, `test_bootstrap.py` |
| P2 | Rendering comparison tests |
| P3 | Handler migration verification tests |
| P4 | Registry unit tests |
| P5 | Decorator unit tests, command integration tests |
| P6 | Final validation suite |

### 13.3 Coverage Requirements

- **Minimum:** No reduction from baseline
- **Target:** >90% on new code
- **Critical paths:** 100% coverage

### 13.4 CLI Smoke Test Suite

```bash
# Basic functionality
codeintel --help
codeintel --version

# Job commands
codeintel jobs list --output json
codeintel jobs status <job_id>

# Health commands
codeintel health check

# Storage commands
codeintel storage info --output json

# Build commands (with test fixture)
codeintel build --project <test_project>

# Graphs commands
codeintel graphs info --output json
```

---

## 14. Rollback Procedures

### 14.1 Phase-Level Rollback

| Phase | Rollback Procedure |
|-------|-------------------|
| P0 | Delete baseline files (no code changes) |
| P1 | Delete new files (`handlers/context.py`, `execution/bootstrap.py`) |
| P2 | Restore `rendering/renderers.py` from git, revert import changes |
| P3 | Per-handler: revert to previous commit |
| P4 | Delete `execution/registry.py`, restore `introspection/registry.py` |
| P5 | Delete `commands/decorators.py`, restore `__call__` methods |
| P6 | Restore deleted files from git (unlikely needed) |

### 14.2 Emergency Rollback

If critical issues discovered post-deployment:

1. **Identify scope** — Single file or systemic?
2. **Git revert** — Revert to last known good commit
3. **Hotfix branch** — Create branch for targeted fixes
4. **Post-mortem** — Document what went wrong

---

## 15. Success Criteria

### 15.1 Functional Criteria

- [ ] All existing CLI commands work identically
- [ ] Help text renders correctly
- [ ] JSON/JSONL/text output modes work
- [ ] Error messages follow RFC 9457 format
- [ ] Exit codes are correct

### 15.2 Structural Criteria

- [ ] Module structure matches architecture document
- [ ] No files from "Files to Delete" list exist
- [ ] Single `HandlerContext` type
- [ ] Single `UnifiedRenderer` implementation
- [ ] Single `OperationRegistry`
- [ ] All commands use `@cli_command` decorator

### 15.3 Quality Criteria

- [ ] Type checking clean (pyright, pyrefly)
- [ ] Linting clean (ruff)
- [ ] Test coverage >= baseline
- [ ] No suppressed errors without justification
- [ ] Docstrings on all public APIs

### 15.4 Performance Criteria

- [ ] CLI startup time not degraded (< 500ms)
- [ ] Command execution time not degraded
- [ ] Memory usage not significantly increased

---

## 16. Appendices

### 16.1 Appendix A: Handler Inventory Template

| Handler File | Handler Functions | Current Context | Param Helpers Used | Runtime Required | Gateway Required | Graph Runtime Required |
|--------------|-------------------|-----------------|-------------------|------------------|------------------|------------------------|
| `jobs.py` | `jobs_list_handler`, `jobs_status_handler`, ... | `EnhancedHandlerContext` | Local `_get_*` | No | No | No |
| `health.py` | `health_check_handler`, `health_status_handler` | `EnhancedHandlerContext` | Local `_get_*` | No | No | No |
| ... | ... | ... | ... | ... | ... | ... |

### 16.2 Appendix B: Command Inventory Template

| Command File | Command Classes | Has `__call__` | Uses `command_context` | Params Extracted |
|--------------|-----------------|----------------|------------------------|------------------|
| `jobs.py` | `JobsListCommand`, `JobsStatusCommand`, ... | Yes | Yes | Manual |
| `health.py` | `HealthCheckCommand` | Yes | Yes | Manual |
| ... | ... | ... | ... | ... |

### 16.3 Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Handler** | Pure function that implements CLI operation logic |
| **Command** | Cyclopts dataclass defining CLI interface |
| **Context** | State container passed to handlers |
| **Operation** | Named, registered CLI capability |
| **Decorator** | Function that modifies class/function behavior |
| **Registry** | Central store for operation metadata |

### 16.4 Appendix D: Reference Documents

- [CLI_UNIFIED_ARCHITECTURE.md](./CLI_UNIFIED_ARCHITECTURE.md) — Target architecture specification
- [AGENTS.md](../../AGENTS.md) — Project coding standards
- Cyclopts Documentation — https://cyclopts.readthedocs.io/

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-10 | AI Assistant | Initial draft |
| 1.1 | 2025-12-10 | AI Assistant | Updated based on Phase 1 & 2 implementation learnings |

### Implementation Learnings (v1.1)

Key deviations and learnings from Phase 1 and Phase 2:

1. **HandlerContextOptions dataclass** — Created to bundle optional parameters and comply with PLR0913 (max 5 arguments)

2. **_lazy_resources.py module** — Added to avoid circular imports between `handlers/context.py` and `execution/context.py`

3. **_BootstrapState dataclass** — Used instead of global variables to avoid PLW0603 linting errors

4. **Type annotations** — Use `dict[str, object]` instead of `dict[str, Any]` for params to comply with typing rules

5. **PEP 695 generics** — Use `def render_cli_result[T](...)` syntax for function-level type parameters

6. **specs.py already existed** — No need to create pre-built table specs file

7. **renderers.py deleted in Phase 2** — Earlier than originally planned (was scheduled for Phase 6)

8. **TextIO import** — Must be from `typing` module, not `io` module

---

*This document serves as the high-level implementation plan. Detailed phase-specific plans are in the `phases/` directory.*
