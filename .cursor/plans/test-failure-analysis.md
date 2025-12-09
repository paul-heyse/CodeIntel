# Test Failure Analysis and Design Improvement Plan

## Executive Summary

Analysis of 7 failing tests reveals two distinct categories of issues:

1. **Architecture Boundary Violation** (1 test): DuckDB imports outside the storage layer
2. **Help Rendering Regression** (6 tests): Our simplification of `build_patched_app` broke multi-location patching

This document provides causal analysis and proposes design changes that not only fix the immediate issues but enhance the codebase's functionality, robustness, extensibility, and maintainability.

---

## Failure Analysis

### Failure Category 1: DuckDB Boundary Violation

**Test**: `tests/architecture/test_duckdb_boundaries.py::test_duckdb_usage_is_localized`

**Error**:
```
Failed: duckdb usage not allowed outside storage/: 
['src/codeintel/cli/storage_handlers.py', 'src/codeintel/ingestion/plugins/repo_scan.py']
```

**Root Cause Analysis**:

Two files import `duckdb` directly instead of using the storage gateway abstraction:

1. **`src/codeintel/cli/storage_handlers.py:9`**:
   ```python
   import duckdb
   ```
   Used for direct profiling operations.

2. **`src/codeintel/ingestion/plugins/repo_scan.py:15`**:
   ```python
   import duckdb
   ```
   Used for batch writes during repository scanning.

**Why This Matters**:
- Violates the hexagonal architecture principle of keeping database concerns isolated
- Creates tight coupling between CLI/ingestion layers and the specific database implementation
- Makes it harder to swap storage backends or mock for testing
- Undermines the gateway abstraction pattern established in the codebase

---

### Failure Category 2: Help Rendering Regression

**Tests**:
- `tests/cli/test_help_defaults_unit.py::test_patched_app_help_with_missing_metadata`
- `tests/cli/test_help_defaults_unit.py::test_help_renders_positional_defaults`
- `tests/cli/test_help_defaults_unit.py::test_help_renders_nested_defaults`
- `tests/cli/test_help_rendering.py::test_docs_export_help_renders`
- `tests/cli/test_help_rendering.py::test_docs_export_help_repeatable`

**Errors**:

For unit tests:
```
AssertionError: '(none)' not found in help output
```

For integration tests:
```
AttributeError: 'NoneType' object has no attribute 'name'
```

**Root Cause Analysis**:

When we simplified `build_patched_app` to use a global patch instead of the `PatchedAppProxy`, we introduced a regression:

1. **Incomplete Patching**: The `apply_help_patch()` function only patches `cyclopts.help.help.create_parameter_help_panel`, but Cyclopts imports this function into multiple locations:
   - `cyclopts.help.help` (the source module)
   - `cyclopts.help` (re-exported in `__init__.py`)
   - `cyclopts.core` (imported inside `_assemble_help_panels` method)

2. **Import Aliasing Problem**: When `cyclopts.core._assemble_help_panels` runs, it does:
   ```python
   from cyclopts.help import create_parameter_help_panel
   ```
   This imports from `cyclopts.help.__init__.py`, which has a different reference than what we patched.

3. **Cyclopts Bug/Assumption**: The underlying Cyclopts code assumes `argument.field_info.default.name` exists for Enum types, crashing when default is `None`:
   ```python
   if is_class_and_subclass(argument.hint, Enum):
       default = argument.parameter.name_transform(argument.field_info.default.name)
   ```

**Why The Original PatchedAppProxy Worked**:

The context manager `_patched_help_renderer()` patched ALL locations via `_iter_patch_targets()`:
```python
def _iter_patch_targets() -> Iterator[tuple[object, str]]:
    yield help_mod, "create_parameter_help_panel"
    for module_name in ("cyclopts.core", "cyclopts.help"):
        # ... patches each module
```

Our simplification to `apply_help_patch()` only patched one location:
```python
def apply_help_patch() -> None:
    help_module.create_parameter_help_panel = create_parameter_help_panel  # Only one!
```

---

## Design Improvements

### Improvement 1: Fix DuckDB Boundary Violations

**Design Principle**: Database technology should be an implementation detail of the storage layer.

#### 1.1 Refactor `storage_handlers.py`

**Current Code** (line 9):
```python
import duckdb
```

**Problem**: Direct duckdb import for profiling operations.

**Solution**: Move profiling operations to the storage layer or use gateway methods.

Create `src/codeintel/storage/helpers/cli_profiling.py`:
```python
"""CLI profiling helpers that work through the gateway abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def profile_query(
    gateway: StorageGateway,
    query: str,
    explain: bool = False,
) -> dict[str, object]:
    """Profile a query through the gateway abstraction.

    Parameters
    ----------
    gateway
        Storage gateway instance.
    query
        SQL query to profile.
    explain
        Whether to return EXPLAIN output.

    Returns
    -------
    dict[str, object]
        Profile results including timing and optionally EXPLAIN output.
    """
    import time

    conn = gateway.connection
    
    start = time.perf_counter()
    result = conn.execute(query).fetchall()
    duration = time.perf_counter() - start
    
    profile = {
        "query": query,
        "duration_seconds": duration,
        "row_count": len(result),
    }
    
    if explain:
        explain_result = conn.execute(f"EXPLAIN {query}").fetchall()
        profile["explain"] = [row[0] for row in explain_result]
    
    return profile
```

Update `storage_handlers.py` to use the gateway:
```python
# Remove: import duckdb
from codeintel.storage.helpers.cli_profiling import profile_query
```

#### 1.2 Refactor `repo_scan.py`

**Current Code** (line 15):
```python
import duckdb
```

**Problem**: Direct duckdb import for batch writes.

**Solution**: Use the gateway's batch write capabilities.

The ingestion layer should use `gateway.write_batch()` or similar methods instead of direct duckdb operations:

```python
# Instead of:
# conn = duckdb.connect(str(db_path))
# conn.execute("INSERT INTO ...")

# Use:
gateway.write_rows("table_name", rows)
# Or use a repository pattern:
repo = SomeRepository(gateway)
repo.insert_batch(rows)
```

### Improvement 2: Fix Help Rendering with Proper Multi-Location Patching

**Design Principle**: Patches must cover all import aliases to be effective.

#### 2.1 Fix `apply_help_patch()` to Patch All Locations

Update `src/codeintel/cli/cyclopts_help.py`:

```python
def apply_help_patch() -> None:
    """Install the hardened help renderer globally for Cyclopts.

    This patches ALL locations where create_parameter_help_panel is imported
    to ensure the patch is effective regardless of how Cyclopts accesses it.
    """
    for module, attr in _iter_patch_targets():
        setattr(module, attr, create_parameter_help_panel)
```

#### 2.2 Create Custom Default Display Class

The `SimpleNamespace(name="(none)")` approach doesn't work well because:
1. Its `__repr__` returns `namespace(name='(none)')` instead of just `(none)`
2. It doesn't trigger Cyclopts to show defaults for None values

Create a proper class:

```python
class _DisplayDefault:
    """A sentinel object that displays a human-readable default in help.

    This class is used to replace None defaults with objects that:
    1. Have a .name attribute for Cyclopts Enum handling
    2. Have __repr__ returning just the display name
    3. Are falsy like None for boolean contexts
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return False  # Falsy like None

    def __eq__(self, other: object) -> bool:
        if other is None:
            return True  # Equal to None for comparison purposes
        if isinstance(other, _DisplayDefault):
            return self.name == other.name
        return False


def _safe_default(argument: Argument) -> object:
    """Return a safe default object for help rendering."""
    default = argument.field_info.default
    if default is not None and hasattr(default, "name"):
        return default

    name = _format_default_value(default, argument_name=str(argument.name))
    return _DisplayDefault(name)
```

### Improvement 3: Add Architecture Boundary Enforcement

**Design Principle**: Architectural constraints should be enforced programmatically.

#### 3.1 Create Import Boundary Module

Create `src/codeintel/_architecture/__init__.py`:
```python
"""Architecture boundary definitions and enforcement utilities.

This module defines the layered architecture boundaries and provides
utilities for enforcing them at test time and runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportBoundary:
    """Defines an import boundary constraint."""

    name: str
    restricted_modules: frozenset[str]
    allowed_paths: frozenset[str]
    description: str


# Define all import boundaries
DUCKDB_BOUNDARY = ImportBoundary(
    name="duckdb",
    restricted_modules=frozenset({"duckdb"}),
    allowed_paths=frozenset({"src/codeintel/storage"}),
    description="DuckDB imports must be confined to storage layer",
)

NETWORKX_BOUNDARY = ImportBoundary(
    name="networkx",
    restricted_modules=frozenset({"networkx", "nx_cugraph"}),
    allowed_paths=frozenset({"src/codeintel/graphs", "src/codeintel/analytics"}),
    description="NetworkX imports must be confined to graphs/analytics layers",
)

ALL_BOUNDARIES = [DUCKDB_BOUNDARY, NETWORKX_BOUNDARY]


def check_boundary(boundary: ImportBoundary, root: Path) -> list[str]:
    """Check for violations of an import boundary.

    Parameters
    ----------
    boundary
        The boundary constraint to check.
    root
        Root directory to scan.

    Returns
    -------
    list[str]
        List of files violating the boundary.
    """
    violations = []
    for path in root.rglob("*.py"):
        # Check if this path is in an allowed location
        str_path = str(path)
        if any(allowed in str_path for allowed in boundary.allowed_paths):
            continue

        # Check for restricted imports
        text = path.read_text(encoding="utf-8")
        for module in boundary.restricted_modules:
            if f"import {module}" in text:
                violations.append(str_path)
                break

    return violations
```

#### 3.2 Parameterize Architecture Tests

Update `tests/architecture/test_duckdb_boundaries.py`:

```python
"""Ensure import boundaries are respected across the codebase."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel._architecture import ALL_BOUNDARIES, check_boundary


@pytest.mark.parametrize("boundary", ALL_BOUNDARIES, ids=lambda b: b.name)
def test_import_boundary_respected(boundary) -> None:
    """Verify import boundaries are respected."""
    root = Path("src")
    violations = check_boundary(boundary, root)
    if violations:
        pytest.fail(f"{boundary.description}: {violations}")
```

### Improvement 4: Add Help Rendering Regression Tests

**Design Principle**: Critical functionality should have targeted regression tests.

#### 4.1 Add Cyclopts Patch Verification Test

Add to `tests/cli/test_help_defaults_unit.py`:

```python
def test_all_cyclopts_locations_are_patched() -> None:
    """Verify the help patch covers all Cyclopts import locations."""
    import cyclopts.help.help as help_mod
    import cyclopts.help as help_pkg

    from codeintel.cli.cyclopts_help import (
        apply_help_patch,
        create_parameter_help_panel,
    )

    # Apply patch
    apply_help_patch()

    # Verify all locations point to our function
    assert help_mod.create_parameter_help_panel is create_parameter_help_panel
    assert help_pkg.create_parameter_help_panel is create_parameter_help_panel


def test_display_default_repr_is_clean() -> None:
    """Verify _DisplayDefault produces clean repr output."""
    from codeintel.cli.cyclopts_help import _DisplayDefault

    none_default = _DisplayDefault("(none)")
    assert repr(none_default) == "(none)"
    assert str(none_default) == "(none)"
    assert none_default.name == "(none)"
    assert not bool(none_default)  # Falsy like None
```

---

## Implementation Order

| Phase | Changes | Impact |
|-------|---------|--------|
| 1 | Fix `apply_help_patch()` to patch all locations | Fixes 5/7 tests immediately |
| 2 | Add `_DisplayDefault` class | Ensures clean help output |
| 3 | Refactor `storage_handlers.py` to use gateway | Fixes 1/7 tests |
| 4 | Refactor `repo_scan.py` to use gateway | Completes boundary fix |
| 5 | Add architecture boundary module | Prevents future violations |
| 6 | Add regression tests | Prevents future regressions |

---

## Best-in-Class Design Enhancements

Beyond fixing the immediate issues, these changes establish:

### Functionality
- Clean separation between database technology and business logic
- Proper help rendering for all parameter types

### Hardness (Robustness)
- Multi-location patching handles Python's import aliasing correctly
- Architecture boundaries are explicitly defined and testable

### Extensibility
- New import boundaries can be added to `_architecture` module
- New display default formats can extend `_DisplayDefault`

### Maintainability
- Single source of truth for boundary definitions
- Parameterized tests reduce duplication
- Clear documentation of why patches are needed

---

## Success Criteria

1. All 7 failing tests pass
2. No new ruff/pyright/pyrefly errors
3. Architecture boundary tests are parameterized and extensible
4. Help rendering works for all default types (None, Enum, bool, str, int)
5. No `import duckdb` outside storage layer

