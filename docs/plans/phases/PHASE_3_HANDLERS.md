# Phase 3: Handler Migration — Detailed Implementation Plan

> **Phase:** 3 of 6  
> **Duration:** 5-7 days  
> **Risk Level:** Medium  
> **Dependencies:** Phase 1 complete, Phase 2 substantially complete  
> **Parallelizable:** Yes (per handler file)  

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Migration Strategy](#3-migration-strategy)
4. [Migration Pattern](#4-migration-pattern)
5. [Handler Migration Schedule](#5-handler-migration-schedule)
6. [Detailed Tasks](#6-detailed-tasks)
7. [File Changes](#7-file-changes)
8. [Testing Requirements](#8-testing-requirements)
9. [Verification Checklist](#9-verification-checklist)
10. [Exit Criteria](#10-exit-criteria)
11. [Rollback Procedure](#11-rollback-procedure)

---

## 1. Objectives

Phase 3 migrates all handlers to the new `HandlerContext`:

1. **Update handler signatures** — All handlers use `HandlerContext` parameter
2. **Remove local param helpers** — Replace with `ctx.param_*()` methods
3. **Standardize context access** — Consistent property usage across handlers
4. **Update commands/context.py** — Create `HandlerContext` instead of `EnhancedHandlerContext`

---

## 2. Prerequisites

### 2.1 Phase Dependencies

- [ ] Phase 1 complete (`handlers/context.py` exists and tested)
- [ ] Phase 2 substantially complete (rendering consolidated)
- [ ] Handler inventory from Phase 0 available

### 2.2 Environment

- [ ] All existing tests passing
- [ ] Clean git working tree

---

## 3. Migration Strategy

### 3.1 Approach

1. **Update `commands/context.py` first** — Make it create `HandlerContext`
2. **Migrate handlers one file at a time** — Each file is independent
3. **Delete local helpers as we go** — Remove after migrating each file
4. **Verify tests after each migration** — Catch issues early

### 3.2 Per-File Process

For each handler file:

1. Update import statement
2. Update handler function signatures
3. Replace `_get_*_param` calls with `ctx.param_*()` calls
4. Replace `_require_*_param` calls with `ctx.require_*()` calls
5. Delete local param helper functions
6. Run tests for that handler
7. Commit changes

### 3.3 Parallelization

Handler files can be migrated in parallel by different contributors:

```
Contributor A: handlers/jobs.py, handlers/health.py
Contributor B: handlers/ops.py, handlers/storage.py
Contributor C: handlers/history.py, handlers/build.py
...
```

---

## 4. Migration Pattern

### 4.1 Import Changes

**Before:**
```python
from codeintel.cli.handlers.protocol import EnhancedHandlerContext
```

**After:**
```python
from codeintel.cli.handlers.context import HandlerContext
```

### 4.2 Signature Changes

**Before:**
```python
def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:
```

**After:**
```python
def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
```

### 4.3 Parameter Access Changes

**Before:**
```python
def _get_str_param(ctx: EnhancedHandlerContext, name: str, default: str | None = None) -> str | None:
    value = ctx.params.get(name)
    if value is None:
        return default
    return str(value)

def _get_int_param(ctx: EnhancedHandlerContext, name: str, default: int = 0) -> int:
    value = ctx.params.get(name)
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value))

def _require_str_param(ctx: EnhancedHandlerContext, name: str) -> str:
    value = ctx.params.get(name)
    if value is None:
        raise ValueError(f"{name} parameter is required")
    return str(value)

def my_handler(ctx: EnhancedHandlerContext) -> CliResult[MyData]:
    name = _get_str_param(ctx, "name")
    limit = _get_int_param(ctx, "limit", 20)
    job_id = _require_str_param(ctx, "job_id")
```

**After:**
```python
def my_handler(ctx: HandlerContext) -> CliResult[MyData]:
    name = ctx.param_str("name")
    limit = ctx.param_int("limit", 20)
    job_id = ctx.require_str("job_id")
```

### 4.4 Param Helper Mapping

| Old Function | New Method |
|--------------|------------|
| `_get_str_param(ctx, key, default)` | `ctx.param_str(key, default)` |
| `_get_int_param(ctx, key, default)` | `ctx.param_int(key, default)` |
| `_get_bool_param(ctx, key, default)` | `ctx.param_bool(key, default=default)` |
| `_get_path_param(ctx, key, default)` | `ctx.param_path(key, default)` |
| `_get_enum_str_param(ctx, key, enum, default)` | `ctx.param_enum(key, enum, default)` |
| `_require_str_param(ctx, key)` | `ctx.require_str(key)` |
| `_require_int_param(ctx, key)` | `ctx.require_int(key)` |
| `_require_path_param(ctx, key)` | `ctx.require_path(key)` |
| `ctx.params.get(key)` | `ctx.param_str(key)` or appropriate type |

---

## 5. Handler Migration Schedule

### 5.1 Migration Order

Handlers are migrated in order of increasing complexity:

| Day | Tier | Files | Complexity | Notes |
|-----|------|-------|------------|-------|
| 1-2 | 1 | `jobs.py`, `health.py` | Low | No runtime/gateway |
| 2-3 | 2 | `ops.py`, `storage.py` | Medium | Uses runtime |
| 3-4 | 3 | `history.py`, `build.py`, `docs.py` | Medium | Multiple handlers |
| 4-5 | 4 | `graphs.py`, `ide.py`, `datasets.py` | Higher | Uses graph_runtime |
| 5-6 | 5 | `plugins.py`, `subsystem.py` | Medium | Plugin integration |

### 5.2 Handler File Details

Based on Phase 0 inventory:

| File | Handlers | Param Helpers | Runtime | Gateway | Graph Runtime |
|------|----------|---------------|---------|---------|---------------|
| `jobs.py` | 5 | `_get_str_param`, `_get_int_param`, `_require_str_param` | No | No | No |
| `health.py` | 2+ | Local helpers | No | No | No |
| `ops.py` | 8+ | `_get_str_param`, `_get_int_param`, `_get_bool_param`, `_require_str_param` | Yes | Yes | No |
| `storage.py` | 3+ | `_get_str_param`, `_get_bool_param` | Yes | Yes | No |
| `history.py` | 1+ | `_get_str_param`, `_get_path_param`, `_get_enum_str_param` | Yes | Yes | No |
| `build.py` | Multiple | Various | Yes | Yes | No |
| `docs.py` | Multiple | Various | Yes | Yes | No |
| `graphs.py` | Multiple | Various | Yes | Yes | Yes |
| `ide.py` | Multiple | Various | Yes | Yes | No |
| `datasets.py` | Multiple | Various | Yes | Yes | No |
| `plugins.py` | Multiple | Various | Yes | Maybe | No |
| `subsystem.py` | Multiple | Various | Yes | Yes | No |

---

## 6. Detailed Tasks

### Task P3-1: Update `commands/context.py`

**Duration:** 2 hours

**Priority:** FIRST — Must be done before any handler migration

**File:** `src/codeintel/cli/commands/context.py`

**Changes:**

1. Update import:
   ```python
   # Before
   from codeintel.cli.handlers.protocol import EnhancedHandlerContext
   
   # After
   from codeintel.cli.handlers.context import HandlerContext
   ```

2. Update `command_context()` yield type:
   ```python
   # Before
   def command_context(...) -> Iterator[tuple[EnhancedHandlerContext, UnifiedRenderer]]:
   
   # After
   def command_context(...) -> Iterator[tuple[HandlerContext, UnifiedRenderer]]:
   ```

3. Update context creation:
   ```python
   # Before
   ctx = EnhancedHandlerContext(
       config=config,
       runtime=runtime,
       params=combined_params,
       verbosity=verbosity,
       _operation_name=operation_id,
   )
   
   # After
   ctx = HandlerContext(
       config=config,
       operation_id=operation_id,
       output_format=render_format,
       verbosity=verbosity,
       project_root=runtime.root if runtime else None,
       database_path=runtime.db_path if runtime else None,
       _params=combined_params,
       _runtime=runtime,  # Pre-populate if available
   )
   ```

4. Update close handling (already compatible since HandlerContext has `close()`)

**Verification:**
```bash
uv run pytest tests/cli/commands/ -v -k "context"
```

---

### Task P3-2: Migrate `handlers/jobs.py`

**Duration:** 2 hours

**File:** `src/codeintel/cli/handlers/jobs.py`

**Current State:**
- 5 handlers: `jobs_list_handler`, `jobs_status_handler`, `jobs_output_handler`, `jobs_cancel_handler`, `jobs_cleanup_handler`
- Local helpers: `_get_str_param`, `_require_str_param`, `_get_int_param`
- No runtime/gateway access needed

**Migration Steps:**

1. **Update import:**
   ```python
   # Before
   from codeintel.cli.handlers.protocol import EnhancedHandlerContext
   
   # After
   from codeintel.cli.handlers.context import HandlerContext
   ```

2. **Update all handler signatures:**
   ```python
   # Before
   def jobs_list_handler(ctx: EnhancedHandlerContext) -> CliResult[JobsListResult]:
   
   # After
   def jobs_list_handler(ctx: HandlerContext) -> CliResult[JobsListResult]:
   ```

3. **Replace param helper calls:**
   ```python
   # jobs_list_handler
   # Before
   status_str = _get_str_param(ctx, "status")
   limit = _get_int_param(ctx, "limit", 20)
   
   # After
   status_str = ctx.param_str("status")
   limit = ctx.param_int("limit", 20)
   
   # jobs_status_handler
   # Before
   job_id = _require_str_param(ctx, "job_id")
   
   # After
   job_id = ctx.require_str("job_id")
   ```

4. **Delete local helper functions:**
   Delete these functions entirely:
   - `_get_str_param`
   - `_require_str_param`
   - `_get_int_param`

5. **Update TYPE_CHECKING import block:**
   ```python
   if TYPE_CHECKING:
       pass  # Remove EnhancedHandlerContext reference if present
   ```

**After migration, the file should NOT contain:**
- Any `EnhancedHandlerContext` references
- Any `_get_*_param` or `_require_*_param` functions
- Any `ctx.params.get()` calls

**Verification:**
```bash
uv run pytest tests/cli/handlers/test_jobs.py -v
uv run pytest tests/cli/commands/test_jobs.py -v
```

---

### Task P3-3: Migrate `handlers/health.py`

**Duration:** 2 hours

Follow same pattern as P3-2.

**Verification:**
```bash
uv run pytest tests/cli/ -k "health" -v
```

---

### Task P3-4: Migrate `handlers/ops.py`

**Duration:** 3 hours

**Additional Complexity:** Uses runtime and gateway

**File:** `src/codeintel/cli/handlers/ops.py`

**Migration Steps:**

1. Update import (as in P3-2)
2. Update all handler signatures
3. Replace param helper calls
4. Verify gateway access patterns:
   ```python
   # Access via property
   gateway = ctx.gateway  # Lazy loaded
   ```
5. Delete local helpers
6. Test with runtime access

**Verification:**
```bash
uv run pytest tests/cli/handlers/test_ops.py -v
uv run pytest tests/cli/commands/test_ops.py -v
```

---

### Task P3-5: Migrate `handlers/storage.py`

**Duration:** 2 hours

Follow same pattern, verify gateway access.

**Verification:**
```bash
uv run pytest tests/cli/ -k "storage" -v
```

---

### Task P3-6: Migrate `handlers/history.py`

**Duration:** 2 hours

**Special handling:** May have `_get_path_param` and `_get_enum_str_param`

Replace with:
```python
# Path
path = ctx.param_path("path")

# Enum
from mymodule import MyEnum
choice = ctx.param_enum("choice", MyEnum)
```

**Verification:**
```bash
uv run pytest tests/cli/ -k "history" -v
```

---

### Task P3-7: Migrate `handlers/build.py`

**Duration:** 3 hours

**Verification:**
```bash
uv run pytest tests/cli/ -k "build" -v
```

---

### Task P3-8: Migrate `handlers/docs.py`

**Duration:** 3 hours

**Verification:**
```bash
uv run pytest tests/cli/ -k "docs" -v
```

---

### Task P3-9: Migrate `handlers/graphs.py`

**Duration:** 4 hours

**Additional Complexity:** Uses `graph_runtime` property

**Verify graph_runtime access:**
```python
# Before
graph_rt = ctx.graph_runtime

# After (same - property name unchanged)
graph_rt = ctx.graph_runtime
```

**Verification:**
```bash
uv run pytest tests/cli/ -k "graph" -v
```

---

### Task P3-10: Migrate `handlers/ide.py`

**Duration:** 3 hours

**Verification:**
```bash
uv run pytest tests/cli/ -k "ide" -v
```

---

### Task P3-11: Migrate `handlers/datasets.py`

**Duration:** 3 hours

**Verification:**
```bash
uv run pytest tests/cli/ -k "dataset" -v
```

---

### Task P3-12: Migrate `handlers/plugins.py`

**Duration:** 2 hours

**Verification:**
```bash
uv run pytest tests/cli/ -k "plugin" -v
```

---

### Task P3-13: Migrate `handlers/subsystem.py`

**Duration:** 2 hours

**Verification:**
```bash
uv run pytest tests/cli/ -k "subsystem" -v
```

---

### Task P3-14: Full Test Suite Validation

**Duration:** 2 hours

After all handlers migrated:

```bash
# Run full CLI test suite
uv run pytest tests/cli/ -v --tb=short

# Verify no EnhancedHandlerContext references in handlers
rg "EnhancedHandlerContext" src/codeintel/cli/handlers/

# Verify no local param helpers remain
rg "_get_str_param|_get_int_param|_get_bool_param|_require_str_param" src/codeintel/cli/handlers/
```

---

### Task P3-15: Code Review and Cleanup

**Duration:** 2 hours

1. Review all changed files
2. Ensure consistent style
3. Update any missed references
4. Run quality checks:
   ```bash
   uv run ruff check --fix src/codeintel/cli/handlers/
   uv run pyright src/codeintel/cli/handlers/
   ```

---

## 7. File Changes

### 7.1 Files Modified

| File | Changes |
|------|---------|
| `commands/context.py` | Create `HandlerContext` instead of `EnhancedHandlerContext` |
| `handlers/jobs.py` | Update signatures, replace param helpers |
| `handlers/health.py` | Update signatures, replace param helpers |
| `handlers/ops.py` | Update signatures, replace param helpers |
| `handlers/storage.py` | Update signatures, replace param helpers |
| `handlers/history.py` | Update signatures, replace param helpers |
| `handlers/build.py` | Update signatures, replace param helpers |
| `handlers/docs.py` | Update signatures, replace param helpers |
| `handlers/graphs.py` | Update signatures, replace param helpers |
| `handlers/ide.py` | Update signatures, replace param helpers |
| `handlers/datasets.py` | Update signatures, replace param helpers |
| `handlers/plugins.py` | Update signatures, replace param helpers |
| `handlers/subsystem.py` | Update signatures, replace param helpers |

### 7.2 Code Removed

From each handler file:
- `_get_str_param()` function
- `_get_int_param()` function
- `_get_bool_param()` function
- `_get_path_param()` function (if present)
- `_get_enum_str_param()` function (if present)
- `_require_str_param()` function
- `_require_int_param()` function (if present)

---

## 8. Testing Requirements

### 8.1 Per-Handler Testing

After migrating each handler file:

```bash
uv run pytest tests/cli/handlers/test_{handler}.py -v
uv run pytest tests/cli/commands/test_{handler}.py -v
```

### 8.2 Integration Testing

Test handler-command integration:

```bash
# Test actual CLI commands
codeintel jobs list
codeintel health check
codeintel ops list
# ... etc
```

### 8.3 Regression Testing

Full test suite:

```bash
uv run pytest tests/cli/ -v
```

---

## 9. Verification Checklist

### 9.1 Per-Handler Checklist

For each handler file:

- [ ] Import updated to `HandlerContext`
- [ ] All handler signatures updated
- [ ] All `_get_*_param` calls replaced
- [ ] All `_require_*_param` calls replaced
- [ ] All local param helper functions deleted
- [ ] No `ctx.params.get()` calls remain
- [ ] Tests pass
- [ ] Lint checks pass

### 9.2 Global Checklist

- [ ] `commands/context.py` creates `HandlerContext`
- [ ] No `EnhancedHandlerContext` imports in handlers (except TYPE_CHECKING if needed)
- [ ] No local param helpers remain in any handler
- [ ] All handler tests pass
- [ ] All command tests pass
- [ ] Full CLI test suite passes

---

## 10. Exit Criteria

Phase 3 is complete when:

| Criterion | Status |
|-----------|--------|
| `commands/context.py` updated | ⬜ |
| `handlers/jobs.py` migrated | ⬜ |
| `handlers/health.py` migrated | ⬜ |
| `handlers/ops.py` migrated | ⬜ |
| `handlers/storage.py` migrated | ⬜ |
| `handlers/history.py` migrated | ⬜ |
| `handlers/build.py` migrated | ⬜ |
| `handlers/docs.py` migrated | ⬜ |
| `handlers/graphs.py` migrated | ⬜ |
| `handlers/ide.py` migrated | ⬜ |
| `handlers/datasets.py` migrated | ⬜ |
| `handlers/plugins.py` migrated | ⬜ |
| `handlers/subsystem.py` migrated | ⬜ |
| No local param helpers remain | ⬜ |
| All tests pass | ⬜ |
| Quality checks pass | ⬜ |

---

## 11. Rollback Procedure

**Per-file rollback:**

If a single handler migration fails:

```bash
git checkout HEAD~1 -- src/codeintel/cli/handlers/{handler}.py
```

**Full phase rollback:**

If significant issues discovered:

1. Revert `commands/context.py`:
   ```bash
   git checkout HEAD~N -- src/codeintel/cli/commands/context.py
   ```

2. Revert all handler changes:
   ```bash
   git checkout HEAD~N -- src/codeintel/cli/handlers/
   ```

3. Verify tests pass with old code

**Note:** Handler files are independent, so individual rollbacks are safe.

---

## Appendix A: Handler Migration Script

```python
#!/usr/bin/env python3
"""Helper script to verify handler migration status.

Usage: python scripts/check_handler_migration.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


HANDLERS_DIR = Path("src/codeintel/cli/handlers")

# Patterns to check
OLD_IMPORT = re.compile(r"from codeintel\.cli\.handlers\.protocol import EnhancedHandlerContext")
OLD_PARAM_HELPER = re.compile(r"def _(?:get|require)_\w+_param\(")
OLD_PARAM_ACCESS = re.compile(r"ctx\.params\.get\(")
NEW_IMPORT = re.compile(r"from codeintel\.cli\.handlers\.context import HandlerContext")


def check_handler_file(path: Path) -> dict:
    """Check migration status of a handler file."""
    content = path.read_text()
    
    return {
        "file": path.name,
        "has_old_import": bool(OLD_IMPORT.search(content)),
        "has_new_import": bool(NEW_IMPORT.search(content)),
        "has_old_param_helpers": bool(OLD_PARAM_HELPER.search(content)),
        "has_old_param_access": bool(OLD_PARAM_ACCESS.search(content)),
    }


def main() -> int:
    """Check all handler files."""
    issues = []
    
    for path in sorted(HANDLERS_DIR.glob("*.py")):
        if path.name.startswith("_") or path.name in ("__init__.py", "base.py", "protocol.py", "context.py"):
            continue
        
        status = check_handler_file(path)
        
        if status["has_old_import"] or status["has_old_param_helpers"] or status["has_old_param_access"]:
            issues.append(status)
            print(f"❌ {status['file']}: Not fully migrated")
        elif status["has_new_import"]:
            print(f"✅ {status['file']}: Migrated")
        else:
            print(f"⚠️ {status['file']}: Unknown state")
    
    if issues:
        print(f"\n{len(issues)} handler(s) need migration")
        return 1
    
    print("\n✅ All handlers migrated!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

**Previous Phase:** [Phase 2: Rendering Consolidation](./PHASE_2_RENDERING.md)  
**Next Phase:** [Phase 4: Registry Unification](./PHASE_4_REGISTRY.md)
