# Phase 4: Registry Unification — Detailed Implementation Plan

> **Phase:** 4 of 6  
> **Duration:** 2-3 days  
> **Risk Level:** Low  
> **Dependencies:** Phase 3 complete ✅  
> **Parallelizable:** No  
> **Last Updated:** December 2024 (Post-Phase 3)  

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Current State Analysis](#3-current-state-analysis)
4. [Target Design](#4-target-design)
5. [Detailed Tasks](#5-detailed-tasks)
6. [File Changes](#6-file-changes)
7. [Testing Requirements](#7-testing-requirements)
8. [Verification Checklist](#8-verification-checklist)
9. [Exit Criteria](#9-exit-criteria)
10. [Rollback Procedure](#10-rollback-procedure)

---

## 1. Objectives

Phase 4 unifies the operation registry:

1. **Create unified `OperationRegistry`** — Single location in `execution/registry.py`
2. **Migrate operation registrations** — From `operations/*.py` to handler modules
3. **Update introspection** — Re-export from new location
4. **Prepare for `@cli_command`** — Registry ready for Phase 5 integration

---

## 2. Prerequisites

### 2.1 Phase Dependencies

- [x] Phase 3 complete (all handlers migrated to `HandlerContext`)
- [x] Handler signatures finalized (all use `ctx: HandlerContext -> CliResult[T]`)
- [x] All existing tests passing (145 handler tests)

### 2.2 Environment

- [ ] Clean git working tree
- [ ] Quality checks passing (ruff, pyright, pyrefly)

### 2.3 Phase 3 Outcomes (Context for Phase 4)

Phase 3 migrated all handlers to use the unified `HandlerContext`:

| Handler Module | Handlers Migrated | Notes |
|---------------|-------------------|-------|
| `jobs.py` | 5 | `param_str`, `param_int` |
| `health.py` | 2 | Minimal params |
| `ops.py` | 7 | `require_str`, `param_list` |
| `storage.py` | 3 | `param_enum` for `MacroRequirement` |
| `history.py` | 1 | `param_path`, `param_list` |
| `build.py` | 3 | `param_path`, `param_bool` |
| `docs.py` | 2 | `param_str`, `param_bool`, `param_list` |
| `graphs.py` | 2 | `param_tuple`, `param_bool` |
| `ide.py` | 1 | `require_str` with validation |
| `datasets.py` | 4 | `param_str`, `param_bool` |
| `plugins.py` | 7 | `require_str`, `param_path` |
| `subsystem.py` | 5 | `param_int`, `param_str`, `require_str` |

All handlers now use typed parameter accessors instead of local `_get_*_param` helper functions.

---

## 2.4 Lessons from Phase 3 (Informing Phase 4)

1. **Handler Context Access:** All handlers now use `HandlerContext` with typed accessors. This means `OperationSpec.handler` can be strongly typed as `Callable[[HandlerContext], CliResult[Any]]`.

2. **Resource Requirements:** During Phase 3, we identified which handlers need runtime/gateway/graph_runtime:
   - Handlers using `ctx.gateway` need `require_gateway=True`
   - Handlers using `ctx.runtime` need `require_runtime=True`
   - Handlers using `ctx.graph_runtime` need `require_graph_runtime=True`

3. **Test Patterns:** Handler tests create `HandlerContext` directly with mock dependencies. Registry tests should follow the same pattern.

4. **No Local Helpers:** All `_get_*_param` functions have been removed. Registrations can reference handlers directly without worrying about helper function dependencies.

---

## 3. Current State Analysis

### 3.1 Current Registry Location

**File:** `src/codeintel/cli/introspection/registry.py`

Contains:
- `OperationRegistry` class
- `get_operation_registry()` function
- `register_operation()` function

### 3.2 Operation Specifications

**Location:** `src/codeintel/cli/operations/*.py`

| File | Operations | Status |
|------|------------|--------|
| `op_operations.py` | `op.list` | Registered with placeholder handler |
| `build_operations.py` | Various build ops | Registered |
| `dataset_operations.py` | Various dataset ops | Registered |
| `docs_operations.py` | Various docs ops | Registered |
| `graph_operations.py` | Various graph ops | Registered |
| `history_operations.py` | Various history ops | Registered |
| `ide_operations.py` | Various IDE ops | Registered |
| `storage_operations.py` | Various storage ops | Registered |
| `subsystem_operations.py` | Various subsystem ops | Registered |

### 3.3 Problems with Current Design

1. **Split location** — Registry in `introspection/`, specs in `operations/`
2. **Placeholder handlers** — Some specs don't point to real handlers
3. **Metadata duplication** — Operation descriptions in specs and handlers
4. **Import complexity** — Circular import risk between operations and handlers

---

## 4. Target Design

### 4.1 New Registry Location

**File:** `src/codeintel/cli/execution/registry.py`

Rationale: Registry is used by executor, belongs in execution layer.

### 4.2 Registration Pattern

Operations registered directly in handler modules:

```python
# handlers/jobs.py

from codeintel.cli.execution.registry import register_operation, OperationSpec
from codeintel.cli.handlers.context import HandlerContext

def jobs_list_handler(ctx: HandlerContext) -> CliResult[JobsListResult]:
    """List background jobs."""
    status_str = ctx.param_str("status")
    limit = ctx.param_int("limit", 20)
    # ... implementation


# At module level (after handler definition)
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

**Note:** The handler signature `(ctx: HandlerContext) -> CliResult[T]` is now standardized across all handlers (Phase 3 complete).

### 4.3 OperationSpec Structure

```python
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.core import CliResult

@dataclass(frozen=True)
class OperationSpec:
    """Specification for a CLI operation."""
    
    operation_id: str                                      # Unique ID (e.g., "jobs.list")
    name: str                                              # Display name
    description: str                                       # Help text
    handler: Callable[[HandlerContext], CliResult[Any]]   # Handler function
    group: str                                             # Command group
    
    # Resource requirements
    require_runtime: bool = True
    require_gateway: bool = True
    require_graph_runtime: bool = False
    
    # Metadata
    tags: tuple[str, ...] = ()
    hidden: bool = False
```

**Handler Type:** All handlers now follow the standardized signature `(ctx: HandlerContext) -> CliResult[T]` per Phase 3.

---

## 5. Detailed Tasks

### Task P4-1: Create `execution/registry.py`

**Duration:** 4 hours

**File:** `src/codeintel/cli/execution/registry.py`

```python
"""Unified operation registry for CLI operations.

This module provides the single, canonical registry for all CLI operations.
Operations are registered here and discovered by:

- The @cli_command decorator (Phase 5)
- The help system
- Programmatic execution via execute_operation()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.cli.core import CliResult
    from codeintel.cli.handlers.context import HandlerContext

LOG = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class OperationSpec:
    """Specification for a CLI operation.

    Parameters
    ----------
    operation_id
        Unique identifier (e.g., "jobs.list", "build.run").
    name
        Human-readable display name.
    description
        Help text describing the operation.
    handler
        Handler function to execute.
    group
        Command group (e.g., "jobs", "build").
    require_runtime
        Whether handler needs ResolvedRuntime.
    require_gateway
        Whether handler needs StorageGateway.
    require_graph_runtime
        Whether handler needs GraphRuntime.
    tags
        Optional tags for filtering/categorization.
    hidden
        If True, operation is hidden from help output.

    Examples
    --------
    >>> spec = OperationSpec(
    ...     operation_id="jobs.list",
    ...     name="List Jobs",
    ...     description="List background jobs",
    ...     handler=jobs_list_handler,
    ...     group="jobs",
    ...     require_runtime=False,
    ... )  # doctest: +SKIP
    """

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


@dataclass
class OperationRegistry:
    """Central registry for all CLI operations.

    The registry maintains a mapping of operation IDs to their specifications.
    Operations can be registered, retrieved, and listed.

    Examples
    --------
    >>> registry = OperationRegistry()
    >>> registry.register(OperationSpec(...))  # doctest: +SKIP
    >>> spec = registry.get("jobs.list")  # doctest: +SKIP
    """

    _operations: dict[str, OperationSpec] = field(default_factory=dict)

    def register(self, spec: OperationSpec) -> OperationSpec:
        """Register an operation specification.

        Parameters
        ----------
        spec
            Operation specification to register.

        Returns
        -------
        OperationSpec
            The registered specification (for chaining).

        Raises
        ------
        ValueError
            If operation ID is already registered.
        """
        if spec.operation_id in self._operations:
            msg = f"Operation already registered: {spec.operation_id}"
            raise ValueError(msg)

        self._operations[spec.operation_id] = spec
        LOG.debug("Registered operation: %s", spec.operation_id)
        return spec

    def get(self, operation_id: str) -> OperationSpec | None:
        """Get an operation specification by ID.

        Parameters
        ----------
        operation_id
            Operation identifier.

        Returns
        -------
        OperationSpec | None
            Specification if found, None otherwise.
        """
        return self._operations.get(operation_id)

    def require(self, operation_id: str) -> OperationSpec:
        """Get an operation specification, raising if not found.

        Parameters
        ----------
        operation_id
            Operation identifier.

        Returns
        -------
        OperationSpec
            The operation specification.

        Raises
        ------
        KeyError
            If operation not found.
        """
        spec = self._operations.get(operation_id)
        if spec is None:
            msg = f"Operation not found: {operation_id}"
            raise KeyError(msg)
        return spec

    def list_operations(
        self,
        *,
        group: str | None = None,
        include_hidden: bool = False,
    ) -> list[OperationSpec]:
        """List registered operations.

        Parameters
        ----------
        group
            Optional group filter.
        include_hidden
            If True, include hidden operations.

        Returns
        -------
        list[OperationSpec]
            Matching operations sorted by operation_id.
        """
        ops = list(self._operations.values())

        if group is not None:
            ops = [op for op in ops if op.group == group]

        if not include_hidden:
            ops = [op for op in ops if not op.hidden]

        return sorted(ops, key=lambda op: op.operation_id)

    def list_groups(self) -> list[str]:
        """List all operation groups.

        Returns
        -------
        list[str]
            Sorted list of unique group names.
        """
        groups = {op.group for op in self._operations.values()}
        return sorted(groups)

    def unregister(self, operation_id: str) -> bool:
        """Unregister an operation.

        Parameters
        ----------
        operation_id
            Operation to remove.

        Returns
        -------
        bool
            True if operation was removed.
        """
        if operation_id in self._operations:
            del self._operations[operation_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all registered operations."""
        self._operations.clear()

    def __len__(self) -> int:
        """Return number of registered operations."""
        return len(self._operations)

    def __contains__(self, operation_id: str) -> bool:
        """Check if operation is registered."""
        return operation_id in self._operations


# -----------------------------------------------------------------------------
# Global Registry
# -----------------------------------------------------------------------------

_REGISTRY: OperationRegistry | None = None


def get_registry() -> OperationRegistry:
    """Get the global operation registry.

    Creates the registry on first access (lazy initialization).

    Returns
    -------
    OperationRegistry
        Global registry instance.
    """
    global _REGISTRY  # noqa: PLW0603

    if _REGISTRY is None:
        _REGISTRY = OperationRegistry()

    return _REGISTRY


def register_operation(spec: OperationSpec) -> OperationSpec:
    """Register an operation with the global registry.

    Convenience function that gets the global registry and registers
    the operation.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    OperationSpec
        The registered specification.

    Examples
    --------
    >>> from codeintel.cli.execution.registry import register_operation, OperationSpec
    >>> register_operation(OperationSpec(
    ...     operation_id="my.op",
    ...     name="My Operation",
    ...     description="Does something",
    ...     handler=my_handler,
    ...     group="my",
    ... ))  # doctest: +SKIP
    """
    return get_registry().register(spec)


def reset_registry() -> None:
    """Reset the global registry (for testing only).

    WARNING: This function is for testing purposes only.
    Do not call in production code.
    """
    global _REGISTRY  # noqa: PLW0603
    _REGISTRY = None


__all__ = [
    "OperationRegistry",
    "OperationSpec",
    "get_registry",
    "register_operation",
    "reset_registry",
]
```

---

### Task P4-2: Define OperationSpec and ParameterSpec

**Duration:** 2 hours

Included in P4-1. Ensure `OperationSpec` covers all needed metadata.

**Optional: Add ParameterSpec for schema validation:**

```python
@dataclass(frozen=True)
class ParameterSpec:
    """Specification for an operation parameter.

    Parameters
    ----------
    name
        Parameter name.
    type
        Expected type (str, int, bool, Path, etc.).
    required
        Whether parameter is required.
    default
        Default value if not provided.
    description
        Help text for the parameter.
    """

    name: str
    type: type
    required: bool = False
    default: Any = None
    description: str = ""
```

---

### Task P4-3: Implement OperationRegistry Class

**Duration:** 4 hours

Included in P4-1. Ensure all methods are implemented:

- `register(spec)` — Add operation
- `get(operation_id)` — Get operation (or None)
- `require(operation_id)` — Get operation (or raise)
- `list_operations(group, include_hidden)` — List operations
- `list_groups()` — List groups
- `unregister(operation_id)` — Remove operation
- `clear()` — Remove all
- `__len__`, `__contains__` — Container protocol

---

### Task P4-4: Add Registrations to Handler Modules

**Duration:** 4 hours

For each handler module, add registration at module level.

**Example: `handlers/jobs.py`**

```python
# At the end of the file, after handler definitions

from codeintel.cli.execution.registry import register_operation, OperationSpec

# --- Operation Registrations ---

register_operation(OperationSpec(
    operation_id="jobs.list",
    name="List Jobs",
    description="List background jobs with optional status filtering",
    handler=jobs_list_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
))

register_operation(OperationSpec(
    operation_id="jobs.status",
    name="Job Status",
    description="Get status of a specific background job",
    handler=jobs_status_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
))

register_operation(OperationSpec(
    operation_id="jobs.output",
    name="Job Output",
    description="Get output of a completed background job",
    handler=jobs_output_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
))

register_operation(OperationSpec(
    operation_id="jobs.cancel",
    name="Cancel Job",
    description="Cancel a running background job",
    handler=jobs_cancel_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
))

register_operation(OperationSpec(
    operation_id="jobs.cleanup",
    name="Cleanup Jobs",
    description="Clean up old completed jobs",
    handler=jobs_cleanup_handler,
    group="jobs",
    require_runtime=False,
    require_gateway=False,
))
```

**Repeat for all handler modules (42 total handlers across 12 files):**

| Module | Handlers | Resource Requirements |
|--------|----------|----------------------|
| `handlers/jobs.py` | 5 | No runtime, no gateway |
| `handlers/health.py` | 2 | No runtime, no gateway |
| `handlers/ops.py` | 7 | Mixed (list: no runtime; serve: yes) |
| `handlers/storage.py` | 3 | Runtime for db_path |
| `handlers/history.py` | 1 | Runtime + gateway |
| `handlers/build.py` | 3 | Runtime + gateway |
| `handlers/docs.py` | 2 | Runtime |
| `handlers/graphs.py` | 2 | No runtime, no gateway |
| `handlers/ide.py` | 1 | Runtime + gateway + graph_runtime |
| `handlers/datasets.py` | 4 | Mixed |
| `handlers/plugins.py` | 7 | No runtime, no gateway |
| `handlers/subsystem.py` | 5 | Runtime + gateway + graph_runtime |

**Note:** Resource requirements determine what the `@cli_command` decorator (Phase 5) will configure for lazy loading.

---

### Task P4-5: Update `introspection/__init__.py`

**Duration:** 1 hour

**File:** `src/codeintel/cli/introspection/__init__.py`

Update to re-export from new location:

```python
"""Introspection utilities for CLI commands and operations."""

from __future__ import annotations

# Re-export registry from new location
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
)

# Keep existing introspection exports
from codeintel.cli.introspection.discovery import (
    discover_commands,
    discover_handlers,
)
from codeintel.cli.introspection.help import (
    format_help,
    generate_help_text,
)
from codeintel.cli.introspection.validation import (
    ValidationSchema,
    StringValidator,
)

__all__ = [
    # Registry (from execution)
    "OperationRegistry",
    "OperationSpec",
    "get_registry",
    "register_operation",
    # Discovery
    "discover_commands",
    "discover_handlers",
    # Help
    "format_help",
    "generate_help_text",
    # Validation
    "ValidationSchema",
    "StringValidator",
]
```

---

### Task P4-6: Verify Help System Works

**Duration:** 2 hours

Test that help/introspection still functions:

```bash
# Test help output
codeintel --help
codeintel jobs --help
codeintel ops list

# Test programmatic access
python -c "
from codeintel.cli.execution.registry import get_registry
registry = get_registry()
# Import handlers to trigger registration
import codeintel.cli.handlers.jobs
print(f'Registered: {len(registry)} operations')
for op in registry.list_operations():
    print(f'  {op.operation_id}: {op.name}')
"
```

---

### Task P4-7: Write Registry Unit Tests

**Duration:** 2 hours

**File:** `tests/cli/execution/test_registry.py`

```python
"""Tests for operation registry."""

from __future__ import annotations

import pytest

from codeintel.cli.core import CliResult
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
    reset_registry,
)


def dummy_handler(ctx):
    """Dummy handler for testing."""
    return CliResult.ok({"test": True})


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset global registry before each test."""
    reset_registry()


class TestOperationSpec:
    """Tests for OperationSpec."""

    def test_creation(self) -> None:
        """Create operation spec with required fields."""
        spec = OperationSpec(
            operation_id="test.op",
            name="Test Operation",
            description="A test operation",
            handler=dummy_handler,
            group="test",
        )
        assert spec.operation_id == "test.op"
        assert spec.name == "Test Operation"
        assert spec.require_runtime is True  # Default

    def test_with_all_fields(self) -> None:
        """Create operation spec with all fields."""
        spec = OperationSpec(
            operation_id="test.op",
            name="Test Operation",
            description="A test operation",
            handler=dummy_handler,
            group="test",
            require_runtime=False,
            require_gateway=False,
            require_graph_runtime=True,
            tags=("tag1", "tag2"),
            hidden=True,
        )
        assert spec.require_runtime is False
        assert spec.require_graph_runtime is True
        assert spec.hidden is True


class TestOperationRegistry:
    """Tests for OperationRegistry."""

    def test_register_operation(self) -> None:
        """Register operation successfully."""
        registry = OperationRegistry()
        spec = OperationSpec(
            operation_id="test.op",
            name="Test",
            description="Test",
            handler=dummy_handler,
            group="test",
        )
        result = registry.register(spec)
        assert result is spec
        assert "test.op" in registry

    def test_register_duplicate_raises(self) -> None:
        """Registering duplicate ID raises ValueError."""
        registry = OperationRegistry()
        spec = OperationSpec(
            operation_id="test.op",
            name="Test",
            description="Test",
            handler=dummy_handler,
            group="test",
        )
        registry.register(spec)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_get_existing(self) -> None:
        """Get existing operation."""
        registry = OperationRegistry()
        spec = OperationSpec(
            operation_id="test.op",
            name="Test",
            description="Test",
            handler=dummy_handler,
            group="test",
        )
        registry.register(spec)
        
        result = registry.get("test.op")
        assert result is spec

    def test_get_missing_returns_none(self) -> None:
        """Get missing operation returns None."""
        registry = OperationRegistry()
        assert registry.get("nonexistent") is None

    def test_require_existing(self) -> None:
        """Require existing operation."""
        registry = OperationRegistry()
        spec = OperationSpec(
            operation_id="test.op",
            name="Test",
            description="Test",
            handler=dummy_handler,
            group="test",
        )
        registry.register(spec)
        
        result = registry.require("test.op")
        assert result is spec

    def test_require_missing_raises(self) -> None:
        """Require missing operation raises KeyError."""
        registry = OperationRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.require("nonexistent")

    def test_list_operations(self) -> None:
        """List all operations."""
        registry = OperationRegistry()
        registry.register(OperationSpec(
            operation_id="b.op",
            name="B",
            description="B",
            handler=dummy_handler,
            group="b",
        ))
        registry.register(OperationSpec(
            operation_id="a.op",
            name="A",
            description="A",
            handler=dummy_handler,
            group="a",
        ))
        
        ops = registry.list_operations()
        assert len(ops) == 2
        assert ops[0].operation_id == "a.op"  # Sorted

    def test_list_operations_by_group(self) -> None:
        """List operations filtered by group."""
        registry = OperationRegistry()
        registry.register(OperationSpec(
            operation_id="jobs.list",
            name="List",
            description="List",
            handler=dummy_handler,
            group="jobs",
        ))
        registry.register(OperationSpec(
            operation_id="build.run",
            name="Run",
            description="Run",
            handler=dummy_handler,
            group="build",
        ))
        
        ops = registry.list_operations(group="jobs")
        assert len(ops) == 1
        assert ops[0].operation_id == "jobs.list"

    def test_list_excludes_hidden(self) -> None:
        """List excludes hidden operations by default."""
        registry = OperationRegistry()
        registry.register(OperationSpec(
            operation_id="visible",
            name="Visible",
            description="Visible",
            handler=dummy_handler,
            group="test",
        ))
        registry.register(OperationSpec(
            operation_id="hidden",
            name="Hidden",
            description="Hidden",
            handler=dummy_handler,
            group="test",
            hidden=True,
        ))
        
        ops = registry.list_operations()
        assert len(ops) == 1
        assert ops[0].operation_id == "visible"
        
        ops = registry.list_operations(include_hidden=True)
        assert len(ops) == 2


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_returns_same_instance(self) -> None:
        """get_registry returns same instance."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_register_operation_uses_global(self) -> None:
        """register_operation adds to global registry."""
        spec = OperationSpec(
            operation_id="global.test",
            name="Test",
            description="Test",
            handler=dummy_handler,
            group="test",
        )
        register_operation(spec)
        
        registry = get_registry()
        assert "global.test" in registry
```

---

### Task P4-8: Integration Testing

**Duration:** 2 hours

Test registry integration with handlers and commands:

```bash
# Verify registrations work
python -c "
from codeintel.cli.handlers import jobs, health, ops
from codeintel.cli.execution.registry import get_registry

registry = get_registry()
print(f'Total operations: {len(registry)}')

# Verify job operations registered
assert 'jobs.list' in registry
assert 'jobs.status' in registry

# Verify handler is correct
spec = registry.get('jobs.list')
assert spec.handler.__name__ == 'jobs_list_handler'

# Verify handler signature (Phase 3 guarantee)
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.core import CliResult
import inspect
sig = inspect.signature(spec.handler)
params = list(sig.parameters.values())
assert len(params) == 1
assert params[0].annotation == HandlerContext

print('✓ Integration test passed')
"
```

**Verify handler tests still pass:**
```bash
uv run pytest tests/cli/handlers/ -v --tb=short
```

**Verify no regressions in CLI commands:**
```bash
codeintel --help
codeintel jobs --help
codeintel ops list
```

---

## 6. File Changes

### 6.1 New Files Created

| File | Purpose |
|------|---------|
| `execution/registry.py` | Unified operation registry |
| `tests/cli/execution/test_registry.py` | Registry unit tests |

### 6.2 Files Modified

| File | Changes |
|------|---------|
| `introspection/__init__.py` | Re-export from new location |
| `handlers/jobs.py` | Add operation registrations |
| `handlers/health.py` | Add operation registrations |
| `handlers/ops.py` | Add operation registrations |
| `handlers/storage.py` | Add operation registrations |
| `handlers/history.py` | Add operation registrations |
| `handlers/build.py` | Add operation registrations |
| `handlers/docs.py` | Add operation registrations |
| `handlers/graphs.py` | Add operation registrations |
| `handlers/ide.py` | Add operation registrations |
| `handlers/datasets.py` | Add operation registrations |
| `handlers/plugins.py` | Add operation registrations |
| `handlers/subsystem.py` | Add operation registrations |

### 6.3 Files Marked for Deletion (Phase 6)

| File | Reason |
|------|--------|
| `introspection/registry.py` | Superseded by `execution/registry.py` |
| `operations/build_operations.py` | Registrations moved to handlers |
| `operations/dataset_operations.py` | Registrations moved to handlers |
| `operations/docs_operations.py` | Registrations moved to handlers |
| `operations/graph_operations.py` | Registrations moved to handlers |
| `operations/history_operations.py` | Registrations moved to handlers |
| `operations/ide_operations.py` | Registrations moved to handlers |
| `operations/op_operations.py` | Registrations moved to handlers |
| `operations/storage_operations.py` | Registrations moved to handlers |
| `operations/subsystem_operations.py` | Registrations moved to handlers |

---

## 7. Testing Requirements

### 7.1 Unit Tests

- Registry creation and operations
- OperationSpec creation
- Global registry functions
- Re-export from introspection

### 7.2 Integration Tests

- Handlers register operations on import
- Help system uses new registry
- Operations can be looked up by ID

---

## 8. Verification Checklist

### 8.1 Registry Implementation

- [ ] `execution/registry.py` created
- [ ] `OperationSpec` dataclass complete
- [ ] `OperationRegistry` class complete
- [ ] Global registry functions work
- [ ] Unit tests pass (>90% coverage)

### 8.2 Operation Registration

- [ ] All handler modules have registrations
- [ ] Operations registered at module load
- [ ] No duplicate operation IDs

### 8.3 Introspection Update

- [ ] `introspection/__init__.py` re-exports
- [ ] Help system works with new registry
- [ ] Backward compatibility maintained

---

## 9. Exit Criteria

Phase 4 is complete when:

| Criterion | Status |
|-----------|--------|
| `execution/registry.py` implemented | ⬜ |
| Unit tests for registry (>90% coverage) | ⬜ |
| All handler modules register operations | ⬜ |
| `introspection/__init__.py` updated | ⬜ |
| Help system works | ⬜ |
| All tests pass | ⬜ |
| Quality checks pass | ⬜ |

---

## 10. Rollback Procedure

**Risk Level:** Low (mostly additive)

**To rollback:**

1. Remove registrations from handler modules
2. Delete `execution/registry.py`
3. Revert `introspection/__init__.py`
4. Delete test file

**Note:** Old registry in `introspection/registry.py` still exists until Phase 6, so rollback is safe.

---

**Previous Phase:** [Phase 3: Handler Migration](./PHASE_3_HANDLERS.md)  
**Next Phase:** [Phase 5: Command Decorator](./PHASE_5_DECORATOR.md)
