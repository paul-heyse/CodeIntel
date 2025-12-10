# Phase 2: Rendering Consolidation — Detailed Implementation Plan

> **Phase:** 2 of 6  
> **Duration:** 2-3 days  
> **Risk Level:** Medium  
> **Dependencies:** Phase 1 complete  
> **Parallelizable:** No  

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Current State Analysis](#3-current-state-analysis)
4. [Deliverables](#4-deliverables)
5. [Detailed Tasks](#5-detailed-tasks)
6. [File Changes](#6-file-changes)
7. [Testing Requirements](#7-testing-requirements)
8. [Verification Checklist](#8-verification-checklist)
9. [Exit Criteria](#9-exit-criteria)
10. [Rollback Procedure](#10-rollback-procedure)

---

## 1. Objectives

Phase 2 consolidates the rendering stack:

1. **Audit rendering implementations** — Document capabilities of both stacks
2. **Enhance `UnifiedRenderer`** — Add any missing capabilities from `renderers.py`
3. **Migrate consumers** — Update all imports to use `service.py`
4. **Delete duplicate code** — Remove `renderers.py` entirely

---

## 2. Prerequisites

### 2.1 Phase 1 Artifacts

- [ ] `handlers/context.py` implemented and tested
- [ ] `execution/bootstrap.py` implemented and tested
- [ ] All Phase 1 tests passing

### 2.2 Environment

- [ ] All existing CLI tests passing
- [ ] Clean git working tree

---

## 3. Current State Analysis

### 3.1 Rendering Implementations

| File | Implementation | Status |
|------|----------------|--------|
| `rendering/service.py` | `UnifiedRenderer` | **Keep** (canonical) |
| `rendering/renderers.py` | `RichRenderer`, `PlainRenderer` | **Delete** (duplicate) |

### 3.2 Capability Comparison

| Capability | `service.py` | `renderers.py` | Action |
|------------|--------------|----------------|--------|
| JSON rendering | ✅ `_write_json()` | ✅ `_render_json_to_stdout()` | None |
| JSONL rendering | ✅ `_write_jsonl()` | ❌ | None |
| Rich table rendering | ✅ `_render_rich_table()` | ✅ `RichRenderer.render_table()` | Verify parity |
| Plain table rendering | ✅ `_render_plain_table()` | ✅ `PlainRenderer.render_table()` | Verify parity |
| Error rendering | ✅ `render_error()` | ✅ Both renderers | Verify parity |
| Success message | ✅ `render_message(level="success")` | ✅ `render_success()` | None |
| Warning message | ✅ `render_message(level="warning")` | ✅ `render_warning()` | None |
| Object rendering | ✅ `_render_data()` | ✅ `render_object()` | Verify parity |
| Factory function | ❌ Missing | ✅ `get_renderer()` | **Add** |
| Result rendering | ✅ `render_result()` | ✅ `render_cli_result()` | Verify parity |
| Pre-built table specs | ❌ | ✅ `OPERATION_TABLE_SPEC`, etc. | **Move** |

### 3.3 Type Definitions

| Type | `table.py` | `renderers.py` | Action |
|------|------------|----------------|--------|
| `ColumnSpec` | ✅ Canonical | ⚠️ Duplicate | Delete from renderers.py |
| `TableSpec` | ✅ Canonical | ⚠️ Duplicate | Delete from renderers.py |

### 3.4 Theme Definition

| Definition | `service.py` | `renderers.py` | Action |
|------------|--------------|----------------|--------|
| `CODEINTEL_THEME` | ✅ Canonical | ⚠️ Duplicate | Delete from renderers.py |

### 3.5 Consumers

| Consumer | Current Import | New Import |
|----------|----------------|------------|
| `execution/executor.py` | `from .renderers import ...` | `from .service import ...` |
| `execution/adapter.py` | `from .renderers import ...` | `from .service import ...` |

---

## 4. Deliverables

### 4.1 Enhanced `service.py`

Add missing capabilities:

```python
# New factory function
def get_renderer(
    output_format: OutputFormat = OutputFormat.TEXT,
    *,
    color: bool | None = None,
) -> UnifiedRenderer:
    """Get renderer for the specified output format."""

# New standalone function for backward compatibility
def render_cli_result[T](
    result: CliResult[T],
    renderer: UnifiedRenderer,
    *,
    table_spec: TableSpec | None = None,
) -> int:
    """Render a CliResult and return exit code."""
```

### 4.2 Moved Table Specs

Move pre-built table specs to `rendering/specs.py` (new file):

```python
# rendering/specs.py
OPERATION_TABLE_SPEC = TableSpec(...)
DATASET_TABLE_SPEC = TableSpec(...)
BUILD_TARGET_TABLE_SPEC = TableSpec(...)
```

### 4.3 Updated Consumers

All rendering consumers use `service.py` imports.

### 4.4 Deleted File

`rendering/renderers.py` removed entirely.

---

## 5. Detailed Tasks

### Task P2-1: Audit Capability Gap

**Duration:** 2 hours

**Steps:**

1. Read both files completely:
   - `src/codeintel/cli/rendering/service.py`
   - `src/codeintel/cli/rendering/renderers.py`

2. Create capability matrix:

   ```markdown
   # Rendering Capability Audit
   
   ## service.py (UnifiedRenderer)
   
   ### Public Methods
   - render_result(result: CliResult[T]) -> int
   - render_table(rows, spec: TableSpec) -> None
   - render_error(error: ProblemDetail) -> None
   - render_message(message, level="info") -> None
   - emit_progress(current, total, message) -> None
   
   ### Private Methods
   - _emit_warning(warning) -> None
   - _render_data(data, metadata) -> None
   - _render_dict(data) -> None
   - _render_rich_table(rows, spec) -> None
   - _render_plain_table(rows, spec) -> None
   - _write_json(obj) -> None
   - _write_jsonl(obj) -> None
   - _serialize(data) -> object
   - _exit_code_for_error(error) -> int
   
   ## renderers.py (RichRenderer/PlainRenderer)
   
   ### Public Methods
   - render_table(rows, spec) -> None
   - render_object(obj) -> None
   - render_error(error) -> None
   - render_success(message) -> None
   - render_warning(message) -> None
   
   ### Module Functions
   - get_renderer(output_format, force_mode) -> OutputRenderer
   - render_cli_result(result, renderer, table_spec) -> int
   
   ## Gaps in service.py
   - [ ] get_renderer() factory function
   - [ ] render_cli_result() standalone function (for backward compatibility)
   - [ ] Pre-built table specs
   ```

3. Document any behavioral differences:
   - How errors are written (stdout vs stderr)
   - JSON formatting differences
   - Table width calculation

---

### Task P2-2: Add Missing Functions to `service.py`

**Duration:** 4 hours

**File:** `src/codeintel/cli/rendering/service.py`

**Add factory function:**

```python
def get_renderer(
    output_format: OutputFormat = OutputFormat.TEXT,
    *,
    color: bool | None = None,
    writer: TextIO | None = None,
    err_writer: TextIO | None = None,
) -> UnifiedRenderer:
    """Get a renderer for the specified output format.

    This is a factory function that creates UnifiedRenderer instances
    with appropriate settings for the output format and environment.

    Parameters
    ----------
    output_format
        Desired output format.
    color
        Override color detection. If None, auto-detect based on TTY.
    writer
        Output stream (defaults to stdout).
    err_writer
        Error stream (defaults to stderr).

    Returns
    -------
    UnifiedRenderer
        Configured renderer instance.

    Examples
    --------
    >>> renderer = get_renderer(OutputFormat.JSON)  # doctest: +SKIP
    >>> renderer.render_message("Hello")  # doctest: +SKIP
    """
    ctx = RenderContext.auto_detect(
        format_override=output_format,
        color_override=color,
        writer=writer,
        err_writer=err_writer,
    )
    return UnifiedRenderer(ctx)
```

**Add standalone render function:**

```python
def render_cli_result[T](
    result: CliResult[T],
    renderer: UnifiedRenderer | None = None,
    *,
    table_spec: TableSpec | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
) -> int:
    """Render a CliResult and return exit code.

    Convenience function that creates a renderer if not provided and
    delegates to render_result(). Supports optional table rendering
    for list data.

    Parameters
    ----------
    result
        CLI result to render.
    renderer
        Optional renderer. If None, creates one based on output_format.
    table_spec
        Optional table spec for rendering list data as tables.
    output_format
        Output format (used if renderer is None).

    Returns
    -------
    int
        Exit code: 0 for success, non-zero for failure.

    Examples
    --------
    >>> result = CliResult.ok({"status": "done"})  # doctest: +SKIP
    >>> exit_code = render_cli_result(result)  # doctest: +SKIP
    """
    if renderer is None:
        renderer = get_renderer(output_format)

    # Handle table spec for list data
    if table_spec is not None and result.success and isinstance(result.data, list):
        renderer.render_table(result.data, table_spec)
        return 0

    return renderer.render_result(result)
```

**Update `__all__`:**

```python
__all__ = [
    "CODEINTEL_THEME",
    "RenderingService",
    "UnifiedRenderer",
    "get_renderer",
    "render_cli_result",
]
```

---

### Task P2-3: Create `rendering/specs.py` for Pre-built Specs

**Duration:** 1 hour

**File:** `src/codeintel/cli/rendering/specs.py`

```python
"""Pre-built table specifications for common CLI outputs.

This module provides reusable TableSpec definitions for standard
output formats across the CLI.
"""

from __future__ import annotations

from codeintel.cli.rendering.table import ColumnSpec, TableSpec

# --- Operation Table ---

OPERATION_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("id", "Operation ID", style="cyan"),
        ColumnSpec("summary", "Summary"),
        ColumnSpec("tags", "Tags", style="muted"),
    ),
    title="Available Operations",
    empty_message="No operations found.",
)

# --- Dataset Table ---

DATASET_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("table_key", "Table Key", style="cyan"),
        ColumnSpec("name", "Name"),
        ColumnSpec("description", "Description", style="muted"),
    ),
    title="Available Datasets",
    empty_message="No datasets found.",
)

# --- Build Target Table ---

BUILD_TARGET_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("name", "Target", style="cyan"),
        ColumnSpec("status", "Status"),
        ColumnSpec("last_run", "Last Run", style="muted"),
    ),
    title="Build Targets",
    empty_message="No build targets found.",
)

# --- Job Table ---

JOB_TABLE_SPEC = TableSpec(
    columns=(
        ColumnSpec("job_id", "Job ID", style="cyan"),
        ColumnSpec("operation_id", "Operation"),
        ColumnSpec("status", "Status"),
        ColumnSpec("created_at", "Created", style="muted"),
    ),
    title="Background Jobs",
    empty_message="No jobs found.",
)

__all__ = [
    "BUILD_TARGET_TABLE_SPEC",
    "DATASET_TABLE_SPEC",
    "JOB_TABLE_SPEC",
    "OPERATION_TABLE_SPEC",
]
```

---

### Task P2-4: Update `execution/executor.py` Imports

**Duration:** 2 hours

**Steps:**

1. Identify all imports from `renderers.py`:
   ```bash
   rg "from.*renderers import" src/codeintel/cli/execution/
   ```

2. Update imports in `execution/executor.py`:

   **Before:**
   ```python
   from codeintel.cli.rendering.renderers import (
       get_renderer,
       render_cli_result,
       OutputRenderer,
   )
   ```

   **After:**
   ```python
   from codeintel.cli.rendering.service import (
       get_renderer,
       render_cli_result,
       UnifiedRenderer,
   )
   ```

3. Update any type hints:
   - `OutputRenderer` → `UnifiedRenderer`
   - `RichRenderer | PlainRenderer` → `UnifiedRenderer`

4. Run tests to verify no breakage:
   ```bash
   uv run pytest tests/cli/execution/ -v
   ```

---

### Task P2-5: Update `execution/adapter.py` Imports

**Duration:** 2 hours

**Steps:**

1. Identify all imports from `renderers.py`:
   ```bash
   rg "from.*renderers import" src/codeintel/cli/execution/adapter.py
   ```

2. Update imports:

   **Before:**
   ```python
   from codeintel.cli.rendering.renderers import (
       get_renderer,
       render_cli_result,
   )
   ```

   **After:**
   ```python
   from codeintel.cli.rendering.service import (
       get_renderer,
       render_cli_result,
   )
   ```

3. Run tests:
   ```bash
   uv run pytest tests/cli/execution/ -v
   ```

---

### Task P2-6: Search for Any Other Consumers

**Duration:** 1 hour

**Steps:**

1. Search entire codebase for `renderers` imports:
   ```bash
   rg "from codeintel.cli.rendering.renderers import" src/
   rg "from codeintel.cli.rendering import.*Renderer" src/
   rg "import.*renderers" src/codeintel/cli/
   ```

2. For each match:
   - Update imports to use `service.py`
   - Update type hints if needed

3. Search tests as well:
   ```bash
   rg "from codeintel.cli.rendering.renderers import" tests/
   ```

4. Document all files changed

---

### Task P2-7: Delete `rendering/renderers.py`

**Duration:** 1 hour

**Pre-deletion verification:**

```bash
# Ensure no imports remain
rg "from codeintel.cli.rendering.renderers" src/ tests/
rg "from codeintel.cli.rendering import.*Rich" src/ tests/
rg "from codeintel.cli.rendering import.*Plain" src/ tests/

# Should return no results
```

**Delete the file:**

```bash
git rm src/codeintel/cli/rendering/renderers.py
```

**Update `rendering/__init__.py`:**

Remove any exports of deleted items:

**Before:**
```python
from codeintel.cli.rendering.renderers import (
    RichRenderer,
    PlainRenderer,
    get_renderer,
    render_cli_result,
)
```

**After:**
```python
from codeintel.cli.rendering.service import (
    UnifiedRenderer,
    get_renderer,
    render_cli_result,
)
from codeintel.cli.rendering.specs import (
    OPERATION_TABLE_SPEC,
    DATASET_TABLE_SPEC,
    BUILD_TARGET_TABLE_SPEC,
    JOB_TABLE_SPEC,
)
```

---

### Task P2-8: Verify All CLI Commands Work

**Duration:** 2 hours

**Manual smoke testing:**

```bash
# Test JSON output
codeintel --help
codeintel jobs list --format json
codeintel health check --format json

# Test text output
codeintel jobs list
codeintel health check

# Test error rendering
codeintel jobs status nonexistent-job-id
```

**Run full CLI test suite:**

```bash
uv run pytest tests/cli/ -v
```

---

### Task P2-9: Run Full Test Suite

**Duration:** 1 hour

```bash
# Full test suite
uv run pytest tests/ -x -q

# CLI-specific tests
uv run pytest tests/cli/ -v --tb=short

# Verify no coverage regression
uv run pytest tests/cli/ \
    --cov=src/codeintel/cli/rendering \
    --cov-report=term-missing
```

---

## 6. File Changes

### 6.1 New Files Created

| File | Type | Purpose |
|------|------|---------|
| `rendering/specs.py` | Python | Pre-built table specifications |

### 6.2 Files Modified

| File | Changes |
|------|---------|
| `rendering/service.py` | Add `get_renderer()`, `render_cli_result()` |
| `rendering/__init__.py` | Update exports |
| `execution/executor.py` | Update imports |
| `execution/adapter.py` | Update imports |

### 6.3 Files Deleted

| File | Reason |
|------|--------|
| `rendering/renderers.py` | Superseded by `service.py` |

---

## 7. Testing Requirements

### 7.1 Unit Tests for New Functions

**File:** `tests/cli/rendering/test_service.py`

Add tests for new functions:

```python
class TestGetRenderer:
    """Tests for get_renderer factory function."""

    def test_returns_unified_renderer(self) -> None:
        """Return UnifiedRenderer instance."""
        renderer = get_renderer()
        assert isinstance(renderer, UnifiedRenderer)

    def test_respects_output_format(self) -> None:
        """Create renderer with specified format."""
        renderer = get_renderer(OutputFormat.JSON)
        assert renderer.context.format == OutputFormat.JSON


class TestRenderCliResult:
    """Tests for render_cli_result function."""

    def test_renders_success(self) -> None:
        """Render successful result."""
        result = CliResult.ok({"key": "value"})
        exit_code = render_cli_result(result)
        assert exit_code == 0

    def test_renders_failure(self) -> None:
        """Render failed result."""
        result = CliResult.fail(ProblemDetail(...))
        exit_code = render_cli_result(result)
        assert exit_code == 1
```

### 7.2 Regression Testing

All existing rendering tests must pass:

```bash
uv run pytest tests/cli/rendering/ -v
```

### 7.3 Integration Testing

Verify rendering works end-to-end:

```bash
uv run pytest tests/cli/ -k "render" -v
```

---

## 8. Verification Checklist

### 8.1 Capability Parity

- [ ] `get_renderer()` function added to service.py
- [ ] `render_cli_result()` function added to service.py
- [ ] Pre-built table specs moved to specs.py
- [ ] All rendering capabilities available via service.py

### 8.2 Import Migration

- [ ] `execution/executor.py` uses service.py imports
- [ ] `execution/adapter.py` uses service.py imports
- [ ] No other files import from renderers.py
- [ ] `rendering/__init__.py` updated

### 8.3 File Cleanup

- [ ] `rendering/renderers.py` deleted
- [ ] No broken imports
- [ ] No duplicate type definitions

### 8.4 Tests Passing

- [ ] All unit tests pass
- [ ] All CLI tests pass
- [ ] Manual smoke test passed

---

## 9. Exit Criteria

Phase 2 is complete when:

| Criterion | Status |
|-----------|--------|
| `get_renderer()` added to service.py | ⬜ |
| `render_cli_result()` added to service.py | ⬜ |
| `specs.py` created with table specs | ⬜ |
| executor.py updated | ⬜ |
| adapter.py updated | ⬜ |
| renderers.py deleted | ⬜ |
| No imports of renderers.py remain | ⬜ |
| All tests pass | ⬜ |
| Manual smoke test passed | ⬜ |

---

## 10. Rollback Procedure

**Risk Level:** Medium (modifies production code)

**To rollback:**

1. Restore deleted file from git:
   ```bash
   git checkout HEAD~1 -- src/codeintel/cli/rendering/renderers.py
   ```

2. Revert import changes:
   ```bash
   git checkout HEAD~1 -- src/codeintel/cli/execution/executor.py
   git checkout HEAD~1 -- src/codeintel/cli/execution/adapter.py
   git checkout HEAD~1 -- src/codeintel/cli/rendering/__init__.py
   ```

3. Remove new files if added:
   ```bash
   rm src/codeintel/cli/rendering/specs.py
   ```

4. Verify tests pass:
   ```bash
   uv run pytest tests/cli/ -x -q
   ```

---

**Previous Phase:** [Phase 1: Foundation Layer](./PHASE_1_FOUNDATION.md)  
**Next Phase:** [Phase 3: Handler Migration](./PHASE_3_HANDLERS.md)
