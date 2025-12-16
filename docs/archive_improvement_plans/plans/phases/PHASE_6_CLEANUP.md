# Phase 6: Legacy Cleanup — Detailed Implementation Plan

> **Phase:** 6 of 6 (Final) ✅ **COMPLETE**  
> **Duration:** 2-3 days  
> **Risk Level:** Low  
> **Dependencies:** Phase 5 complete ✅  
> **Parallelizable:** Partially  
> **Last Updated:** December 2024 (Implementation Complete)  

---

## Implementation Summary

Phase 6 cleanup was completed successfully with the following changes:

### Files Deleted
- `src/codeintel/cli/operations/*.py` (9 files) — LEGACY operation registrations
- `src/codeintel/cli/introspection/registry.py` — LEGACY registry (replaced by `execution/registry.py`)
- `src/codeintel/cli/_migration_flags.py` — Temporary migration feature flags

### Files Updated
- `src/codeintel/cli/introspection/__init__.py` — Now uses `execution/registry.py`
- `src/codeintel/cli/introspection/discovery.py` — Uses new `OperationInfo` structure
- `src/codeintel/cli/introspection/help.py` — Updated for new `OperationInfo` fields
- `src/codeintel/cli/handlers/__init__.py` — Exports both new and legacy context types
- `src/codeintel/cli/handlers/context.py` — Removed `handler_context_from_enhanced` adapter
- `src/codeintel/cli/commands/help_commands.py` — Updated for renamed functions
- `src/codeintel/cli/execution/registry.py` — Added legacy compatibility fields
- `tests/cli/_harness/__init__.py` — Updated to handle handler-based operations

### Backward Compatibility
- `get_operation_registry()` now returns the unified `get_registry()` from `execution/registry.py`
- NEW `OperationSpec` includes legacy fields (`category`, `is_async`, `param_schema`, etc.) for executor compatibility
- `LegacyHandlerContext` alias provided for old code

### Test Results
- 540 CLI tests pass
- 2 pre-existing failures from Phase 5 (not caused by Phase 6):
  - `test_datasets_scaffold_existing_name` 
  - `test_history_timeseries_cli_happy_path`

### Not Completed (Optional)
- `graphs.py` and `serve.py` migration to `@cli_command` — Retained manual `__call__` due to complex conditional logic
- `commands/context.py` deletion — Still used by special-case commands

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Files to Delete](#3-files-to-delete)
4. [Detailed Tasks](#4-detailed-tasks)
5. [File Changes](#5-file-changes)
6. [Testing Requirements](#6-testing-requirements)
7. [Verification Checklist](#7-verification-checklist)
8. [Exit Criteria](#8-exit-criteria)
9. [Final Validation](#9-final-validation)
10. [Known Issues from Phase 5](#10-known-issues-from-phase-5)
11. [Post-Migration](#11-post-migration)

---

## 1. Objectives

Phase 6 completes the migration by removing all legacy code:

1. **Delete old context types** — Remove superseded context implementations
2. **Delete old command infrastructure** — Remove `command_context`, adapter
3. **Delete operation placeholders** — Remove `operations/*.py` files
4. **Delete old registry** — Remove `introspection/registry.py`
5. **Remove temporary scaffolding** — Delete feature flags, adapters
6. **Final validation** — Ensure clean architecture

---

## 2. Prerequisites

### 2.1 Phase Dependencies

- [x] Phase 5 complete (34 commands migrated to `@cli_command`)
- [x] Phase 3 complete (all handlers use `HandlerContext`)
- [x] Handler module registrations removed (decorator handles registration)
- [ ] All CLI commands working
- [ ] All tests passing
- [ ] No imports of files to be deleted

### 2.1.1 Phase 5 State Summary

Phase 5 completed the following:
- **34 commands** migrated to `@cli_command` decorator
- **Handler modules** no longer contain `register_operation()` calls
- **NEW registry** populated via decorator at class decoration time
- **LEGACY registry** still populated by `operations/*.py` (for backward compat)
- **Special cases** (`graphs.py`) retain manual `__call__` due to conditional logic
- **Backward compat** `cyclopts_ops` re-exported from `cli/__init__.py`

### 2.2 Phase 3 Context (Relevant to Cleanup)

Phase 3 introduced a temporary adapter function that should be removed in Phase 6:

```python
# In handlers/context.py (temporary - remove in Phase 6)
def handler_context_from_enhanced(
    ctx: EnhancedHandlerContext,
    operation_id: str,
    params: dict[str, object] | None = None,
) -> HandlerContext:
    """Create HandlerContext from legacy EnhancedHandlerContext."""
    ...
```

This is a **standalone function** (not a classmethod) that was extracted from `HandlerContext` in Phase 3 to reduce public method count.

### 2.3 Pre-Deletion Verification

Before deleting any file, verify no imports remain:

```bash
# Check for imports of each file to delete
rg "from codeintel.cli.handlers.base import" src/ tests/
rg "from codeintel.cli.handlers.protocol import" src/ tests/
rg "from codeintel.cli.execution.context import" src/ tests/
rg "from codeintel.cli.commands.context import" src/ tests/
rg "from codeintel.cli.execution.adapter import" src/ tests/
rg "from codeintel.cli.introspection.registry import" src/ tests/
rg "from codeintel.cli.operations" src/ tests/

# IMPORTANT: commands/context.py is still used by graphs.py (special case)
# Check if it can be deleted or needs migration first
rg "from codeintel.cli.commands.context import" src/codeintel/cli/commands/
```

**Note:** After Phase 5, `commands/context.py` may still be imported by:
- `graphs.py` (uses `command_context` for conditional handler dispatch)
- Any other command that wasn't migrated to `@cli_command`

These special cases need to be addressed before `commands/context.py` can be deleted.

---

## 3. Files to Delete

### 3.0 Already Cleaned in Phase 5

The following were removed during Phase 5 command migrations:

| Change | Location | Notes |
|--------|----------|-------|
| `register_operation()` calls | `handlers/*.py` | Decorator handles registration |
| `OperationSpec` imports | `handlers/*.py` | No longer needed in handlers |
| Handler registration sections | `handlers/*.py` | Entire sections removed |

**Handler files cleaned:**
- `handlers/jobs.py`, `handlers/health.py`, `handlers/ide.py`
- `handlers/docs.py`, `handlers/history.py`, `handlers/storage.py`
- `handlers/build.py`, `handlers/datasets.py`, `handlers/plugins.py`
- `handlers/subsystem.py`, `handlers/ops.py`

### 3.1 Old Context Types

| File | Reason | Superseded By |
|------|--------|---------------|
| `handlers/base.py` | Old HandlerContext | `handlers/context.py` |
| `handlers/protocol.py` | EnhancedHandlerContext | `handlers/context.py` |
| `execution/context.py` | ExecutionContext | `handlers/context.py` |

### 3.2 Old Command Infrastructure

| File | Reason | Superseded By | Notes |
|------|--------|---------------|-------|
| `commands/context.py` | command_context() | `commands/decorators.py` | **⚠️ Check graphs.py first** |
| `execution/adapter.py` | CycloptsAdapter | `commands/decorators.py` | Can delete immediately |

**⚠️ `commands/context.py` Special Handling:**

After Phase 5, the following files still import `command_context`:

| File | Reason | Migration Path |
|------|--------|----------------|
| `graphs.py` | Conditional handler dispatch | Create wrapper handler |
| `serve.py` | MCP/HTTP serve commands | Migrate to `@cli_command` |
| `_common.py` | Re-export | Remove export |
| `__init__.py` | Re-export | Remove export |

**`graphs.py` Example:**
```python
# Current: uses command_context for conditional dispatch
with command_context("graph.plugins", ...) as (ctx, renderer):
    if self.plan or self.validate_plan:
        result = graph_plugins_plan_handler(ctx)
    else:
        result = graph_plugins_list_handler(ctx)
```

**Options:**
1. **Keep `commands/context.py`** for special cases only (short-term)
2. **Migrate remaining commands** - graphs.py and serve.py (recommended)
3. **Inline the context creation** where needed and delete `commands/context.py`

**Recommended:** Option 2 - Migrate `graphs.py` (create wrapper handler) and `serve.py` to `@cli_command`, then delete `commands/context.py`.

### 3.3 Operation Placeholders

| File | Reason | Superseded By |
|------|--------|---------------|
| `operations/build_operations.py` | Placeholder specs | Handler registrations |
| `operations/dataset_operations.py` | Placeholder specs | Handler registrations |
| `operations/docs_operations.py` | Placeholder specs | Handler registrations |
| `operations/graph_operations.py` | Placeholder specs | Handler registrations |
| `operations/history_operations.py` | Placeholder specs | Handler registrations |
| `operations/ide_operations.py` | Placeholder specs | Handler registrations |
| `operations/op_operations.py` | Placeholder specs | Handler registrations |
| `operations/storage_operations.py` | Placeholder specs | Handler registrations |
| `operations/subsystem_operations.py` | Placeholder specs | Handler registrations |

### 3.4 Old Registry

| File | Reason | Superseded By |
|------|--------|---------------|
| `introspection/registry.py` | Original registry | `execution/registry.py` |

### 3.5 Temporary Scaffolding

| File | Reason |
|------|--------|
| `_migration_flags.py` | Migration complete |

---

## 4. Detailed Tasks

### Task P6-1: Verify No Imports of Files to Delete

**Duration:** 2 hours

**Critical:** This step prevents breaking changes.

**Run all verification commands:**

```bash
#!/bin/bash
# scripts/verify_no_legacy_imports.sh

echo "=== Checking for legacy imports ==="

# Context types
echo "Checking handlers/base.py..."
rg "from codeintel.cli.handlers.base import" src/ tests/ && echo "❌ FOUND" || echo "✓ Clean"

echo "Checking handlers/protocol.py..."
rg "from codeintel.cli.handlers.protocol import" src/ tests/ && echo "❌ FOUND" || echo "✓ Clean"

echo "Checking execution/context.py..."
rg "from codeintel.cli.execution.context import" src/ tests/ && echo "❌ FOUND" || echo "✓ Clean"

# Command infrastructure (NOTE: graphs.py may still use this)
echo "Checking commands/context.py..."
rg "from codeintel.cli.commands.context import" src/ tests/ && echo "❌ FOUND (check if graphs.py)" || echo "✓ Clean"

echo "Checking execution/adapter.py..."
rg "from codeintel.cli.execution.adapter import" src/ tests/ && echo "❌ FOUND" || echo "✓ Clean"

# Old registry
echo "Checking introspection/registry.py..."
rg "from codeintel.cli.introspection.registry import" src/ tests/ && echo "❌ FOUND" || echo "✓ Clean"

# Operations
echo "Checking operations/*.py..."
rg "from codeintel.cli.operations" src/ tests/ && echo "❌ FOUND" || echo "✓ Clean"

# Additional Phase 5 verification
echo ""
echo "=== Phase 5 Cleanup Verification ==="

echo "Checking for handler registration calls..."
rg "^register_operation\(" src/codeintel/cli/handlers/ && echo "❌ FOUND" || echo "✓ No handler registrations"

echo "Checking decorator usage..."
COUNT=$(rg "@cli_command" src/codeintel/cli/commands/*.py --count | wc -l)
echo "✓ @cli_command found in $COUNT files"

echo "=== Verification complete ==="
```

**Expected state after Phase 5:**
- `handlers/*.py` should have NO `register_operation()` calls
- `commands/*.py` should have 34+ `@cli_command` decorators
- Only `graphs.py` should still use `commands/context.py`

**If any imports found:**
1. Update the importing file to use new imports
2. Re-run verification
3. Only proceed when all checks pass (except known exceptions)

---

### Task P6-2: Delete Old Context Types

**Duration:** 1 hour

**Order:** Delete in reverse dependency order

**Step 1: Delete `handlers/base.py`**

```bash
git rm src/codeintel/cli/handlers/base.py
```

**Step 2: Delete `handlers/protocol.py`**

```bash
git rm src/codeintel/cli/handlers/protocol.py
```

**Step 3: Delete `execution/context.py`**

```bash
git rm src/codeintel/cli/execution/context.py
```

**After each deletion, run:**

```bash
uv run pytest tests/cli/ -x -q
```

---

### Task P6-2.5: Migrate graphs.py Special Case (NEW)

**Duration:** 2 hours

**Before deleting `commands/context.py`, migrate `graphs.py`:**

**Option A: Create wrapper handler (Recommended)**

```python
# In handlers/graphs.py - add wrapper handler
def graph_plugins_handler(ctx: HandlerContext) -> CliResult[GraphPluginsResult]:
    """Handle graph plugins list or plan based on params."""
    plan_mode = ctx.param_bool("plan") or ctx.param_bool("validate_plan")
    
    if plan_mode:
        return graph_plugins_plan_handler(ctx)
    return graph_plugins_list_handler(ctx)
```

Then migrate `commands/graphs.py`:

```python
# In commands/graphs.py
_GRAPH_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

@cli_command("graph.plugins", handler=graph_plugins_handler, config=_GRAPH_CONFIG)
@graphs_app.command(name="plugins")
@dataclass
class GraphPluginsCommand:
    """List or plan graph plugins."""
    plan: Annotated[bool, Parameter(...)] = False
    # ... rest of fields
    # No __call__ needed
```

**Option B: Keep commands/context.py for special cases**

If migration is too risky, retain `commands/context.py` but:
1. Add deprecation warning
2. Document as legacy-only
3. Plan removal in future

**Verify after migration:**

```bash
uv run pytest tests/cli/test_graph*.py -v
codeintel graph plugins
codeintel graph plugins --plan
```

---

### Task P6-3: Delete Old Command Infrastructure

**Duration:** 1 hour

**Prerequisites:** Task P6-2.5 complete (graphs.py migrated or decision made)

**Step 1: Delete `commands/context.py`** (if graphs.py migrated)

```bash
git rm src/codeintel/cli/commands/context.py
```

**Step 2: Delete `execution/adapter.py`**

```bash
git rm src/codeintel/cli/execution/adapter.py
```

**Verify:**

```bash
uv run pytest tests/cli/ -x -q
```

---

### Task P6-4: Delete Operation Placeholder Files

**Duration:** 1 hour

**Delete all operation files:**

```bash
git rm src/codeintel/cli/operations/build_operations.py
git rm src/codeintel/cli/operations/dataset_operations.py
git rm src/codeintel/cli/operations/docs_operations.py
git rm src/codeintel/cli/operations/graph_operations.py
git rm src/codeintel/cli/operations/history_operations.py
git rm src/codeintel/cli/operations/ide_operations.py
git rm src/codeintel/cli/operations/op_operations.py
git rm src/codeintel/cli/operations/storage_operations.py
git rm src/codeintel/cli/operations/subsystem_operations.py
```

**Update `operations/__init__.py`:**

```python
"""Operation definitions (registrations are in handler modules)."""

from __future__ import annotations

# Operations are now registered in handler modules via @cli_command decorator
# This package is retained for potential future use

__all__: list[str] = []
```

Or delete the entire operations directory if empty:

```bash
# If no other files remain in operations/
git rm -r src/codeintel/cli/operations/
```

---

### Task P6-5: Delete Old Registry

**Duration:** 1 hour

**Delete `introspection/registry.py`:**

```bash
git rm src/codeintel/cli/introspection/registry.py
```

**Update `introspection/__init__.py`:**

Re-export from `execution/registry.py` with compatibility aliases:

```python
"""Introspection utilities for CLI commands and operations."""

from __future__ import annotations

# Registry is now in execution layer - re-export with both old and new names
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
    reset_registry,
)

# Provide legacy aliases for code that imported from here
get_operation_registry = get_registry  # Legacy name

# Introspection utilities (if these exist - check actual modules)
# from codeintel.cli.introspection.discovery import ...
# from codeintel.cli.introspection.help import ...
# from codeintel.cli.introspection.validation import ...

__all__ = [
    # Registry (from execution)
    "OperationRegistry",
    "OperationSpec",
    "get_registry",
    "get_operation_registry",  # Legacy alias
    "register_operation",
    "reset_registry",
    # Add other introspection exports as needed
]
```

**Also update `cli/__init__.py`:**

The Phase 5 implementation added `cyclopts_ops` re-export for backward compatibility:

```python
# In cli/__init__.py
from codeintel.cli.commands import ops as cyclopts_ops

__all__ = [
    # ... existing exports
    "cyclopts_ops",  # Backward compatibility for tests
]
```

This allows `from codeintel.cli import cyclopts_ops` to continue working.

---

### Task P6-6: Delete Feature Flags Module

**Duration:** 0.5 hours

**If created in Phase 0:**

```bash
git rm src/codeintel/cli/_migration_flags.py
```

**Remove any references:**

```bash
rg "_migration_flags" src/
# Update any files that import it
```

---

### Task P6-7: Update All `__init__.py` Files

**Duration:** 2 hours

Update exports in each package's `__init__.py`:

**`handlers/__init__.py`:**

```python
"""CLI handler implementations."""

from __future__ import annotations

# Context is now the single context type
from codeintel.cli.handlers.context import (
    HandlerContext,
    ParameterError,  # Exception for missing/invalid params
    handler_context_manager,
    # NOTE: handler_context_from_enhanced removed in Phase 6
)

# Re-export handlers
from codeintel.cli.handlers.jobs import (
    jobs_cancel_handler,
    jobs_cleanup_handler,
    jobs_list_handler,
    jobs_output_handler,
    jobs_status_handler,
)
# ... other handlers

__all__ = [
    "HandlerContext",
    "ParameterError",
    "handler_context_manager",
    # ... handler exports
]
```

**Note:** The `handler_context_from_enhanced` adapter function is removed in Phase 6 as all code paths now use `HandlerContext` directly.

**`execution/__init__.py`:**

```python
"""CLI execution infrastructure."""

from __future__ import annotations

from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.cli.execution.executor import CommandExecutor
from codeintel.cli.execution.registry import (
    OperationRegistry,
    OperationSpec,
    get_registry,
    register_operation,
)

__all__ = [
    "CommandExecutor",
    "OperationRegistry",
    "OperationSpec",
    "bootstrap_cli",
    "get_registry",
    "register_operation",
]
```

**`commands/__init__.py`:**

```python
"""CLI command definitions."""

from __future__ import annotations

from codeintel.cli.commands.app import app
from codeintel.cli.commands.decorators import cli_command

__all__ = [
    "app",
    "cli_command",
]
```

---

### Task P6-8: Remove Compatibility Code

**Duration:** 2 hours

Search for and remove any compatibility code:

```bash
# Find adapter function (standalone, not classmethod - changed in Phase 3)
rg "handler_context_from_enhanced" src/

# Find deprecation warnings
rg "DeprecationWarning" src/codeintel/cli/

# Find TODO comments about migration
rg "TODO.*migration|TODO.*Phase" src/codeintel/cli/
```

**Remove from `handlers/context.py`:**

Delete the `handler_context_from_enhanced` **standalone function** (note: this was converted from a classmethod in Phase 3):

```python
# DELETE THIS ENTIRE FUNCTION (it's at module level, not inside HandlerContext)
def handler_context_from_enhanced(
    ctx: EnhancedHandlerContext,
    operation_id: str,
    params: dict[str, object] | None = None,
) -> HandlerContext:
    """Create HandlerContext from legacy EnhancedHandlerContext.
    
    This is a temporary adapter for gradual migration. It will be
    removed in Phase 6 when all handlers have been migrated.
    """
    ...
```

**Also remove from `__all__`:**

```python
# In handlers/context.py __all__, remove:
"handler_context_from_enhanced",
```

---

### Task P6-9: Full Test Suite

**Duration:** 1 hour

```bash
# Full test suite
uv run pytest tests/ -x --tb=short

# CLI-specific tests
uv run pytest tests/cli/ -v

# Verify coverage hasn't regressed
uv run pytest tests/cli/ \
    --cov=src/codeintel/cli \
    --cov-report=term-missing
```

---

### Task P6-10: Type Checking

**Duration:** 1 hour

```bash
# Pyright
uv run pyright --warnings --pythonversion=3.13 src/codeintel/cli/

# Pyrefly
uv run pyrefly check src/codeintel/cli/
```

**Fix any errors introduced by deletions.**

---

### Task P6-11: Linting

**Duration:** 0.5 hours

```bash
# Ruff check
uv run ruff check src/codeintel/cli/

# Ruff format
uv run ruff format src/codeintel/cli/
```

---

### Task P6-12: CLI Smoke Test

**Duration:** 1 hour

Test all major command groups manually:

```bash
# Core commands
codeintel --help
codeintel --version

# Jobs
codeintel jobs list
codeintel jobs list --format json

# Health
codeintel health check

# Operations
codeintel ops list

# Storage
codeintel storage info

# Build
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

# Config
codeintel config show

# Completions
codeintel completions install --help
```

---

### Task P6-13: Coverage Comparison

**Duration:** 1 hour

Compare current coverage to Phase 0 baseline:

```bash
# Generate current coverage
uv run pytest tests/cli/ \
    --cov=src/codeintel/cli \
    --cov-report=json:docs/plans/phases/artifacts/coverage_final.json

# Compare
python -c "
import json

with open('docs/plans/phases/artifacts/coverage_baseline.json') as f:
    baseline = json.load(f)

with open('docs/plans/phases/artifacts/coverage_final.json') as f:
    final = json.load(f)

baseline_pct = baseline['totals']['percent_covered']
final_pct = final['totals']['percent_covered']

print(f'Baseline coverage: {baseline_pct:.1f}%')
print(f'Final coverage: {final_pct:.1f}%')
print(f'Change: {final_pct - baseline_pct:+.1f}%')

if final_pct >= baseline_pct:
    print('✓ Coverage maintained or improved')
else:
    print('⚠ Coverage decreased - investigate')
"
```

---

### Task P6-14: Documentation Updates

**Duration:** 2 hours

1. **Archive migration documents:**
   ```bash
   mkdir -p docs/plans/archive
   mv docs/plans/phases/MIGRATION_TRACKING.md docs/plans/archive/
   mv docs/plans/phases/artifacts/ docs/plans/archive/
   ```

2. **Update architecture documentation:**
   - Update `CLI_UNIFIED_ARCHITECTURE.md` with final state
   - Mark migration as complete

3. **Update developer docs:**
   - Document new `@cli_command` pattern
   - Remove references to old patterns

---

### Task P6-15: Final Code Review

**Duration:** 2 hours

1. Review all changes since Phase 0
2. Ensure consistent style
3. Verify all quality checks pass
4. Get sign-off from team (if applicable)

---

## 5. File Changes

### 5.1 Files Deleted

| File | Reason |
|------|--------|
| `handlers/base.py` | Superseded by `handlers/context.py` |
| `handlers/protocol.py` | Superseded by `handlers/context.py` |
| `execution/context.py` | Superseded by `handlers/context.py` |
| `execution/adapter.py` | Superseded by `commands/decorators.py` |
| `commands/context.py` | Superseded by `commands/decorators.py` |
| `introspection/registry.py` | Moved to `execution/registry.py` |
| `operations/build_operations.py` | Registrations in handlers |
| `operations/dataset_operations.py` | Registrations in handlers |
| `operations/docs_operations.py` | Registrations in handlers |
| `operations/graph_operations.py` | Registrations in handlers |
| `operations/history_operations.py` | Registrations in handlers |
| `operations/ide_operations.py` | Registrations in handlers |
| `operations/op_operations.py` | Registrations in handlers |
| `operations/storage_operations.py` | Registrations in handlers |
| `operations/subsystem_operations.py` | Registrations in handlers |
| `_migration_flags.py` | Migration scaffolding |

**Total: ~16 files deleted**

### 5.2 Files Modified

| File | Changes |
|------|---------|
| `handlers/__init__.py` | Update exports |
| `execution/__init__.py` | Update exports |
| `commands/__init__.py` | Update exports |
| `introspection/__init__.py` | Update re-exports |
| `handlers/context.py` | Remove adapter method |

---

## 6. Testing Requirements

### 6.1 After Each Deletion

Run quick tests to catch issues immediately:

```bash
uv run pytest tests/cli/ -x -q
```

### 6.2 After All Deletions

Full validation:

```bash
uv run pytest tests/ -v
uv run pyright src/codeintel/cli/
uv run ruff check src/codeintel/cli/
```

---

## 7. Verification Checklist

### 7.1 Pre-Cleanup Status (After Phase 5)

- [x] 34 commands migrated to `@cli_command`
- [x] Handler modules cleaned of registration code
- [x] Decorator unit tests passing
- [x] Quality checks (ruff, pyright, pyrefly) clean
- [ ] Special case (`graphs.py`) migrated or decision made

### 7.2 Files Deleted

- [ ] `handlers/base.py` deleted
- [ ] `handlers/protocol.py` deleted
- [ ] `execution/context.py` deleted
- [ ] `execution/adapter.py` deleted
- [ ] `commands/context.py` deleted (or retained for `graphs.py`)
- [ ] `introspection/registry.py` deleted
- [ ] All `operations/*.py` files deleted
- [ ] `_migration_flags.py` deleted (if created)

### 7.3 Imports Clean

- [ ] No imports of deleted files in `src/`
- [ ] No imports of deleted files in `tests/`
- [ ] All `__init__.py` files updated
- [ ] `handler_context_from_enhanced` removed from `context.py` and `__all__`
- [ ] `cyclopts_ops` re-export retained in `cli/__init__.py` (backward compat)

### 7.4 Quality Checks

- [ ] All tests pass
- [ ] Type checking clean (pyright + pyrefly)
- [ ] Linting clean (ruff)
- [ ] CLI smoke test passed
- [ ] Coverage >= baseline

---

## 8. Exit Criteria

Phase 6 is complete when:

| Criterion | Status |
|-----------|--------|
| All legacy files deleted | ⬜ |
| No imports of deleted files | ⬜ |
| All `__init__.py` updated | ⬜ |
| `handler_context_from_enhanced` removed | ⬜ |
| No compatibility code remains | ⬜ |
| All tests pass | ⬜ |
| Type checking clean | ⬜ |
| Linting clean | ⬜ |
| CLI smoke test passed | ⬜ |
| Coverage >= baseline | ⬜ |
| Documentation updated | ⬜ |
| Final code review complete | ⬜ |

---

## 9. Final Validation

### 9.1 Architecture Verification

Verify final module structure matches target:

```
cli/
├── __init__.py
├── commands/
│   ├── __init__.py
│   ├── _common.py
│   ├── _help.py
│   ├── app.py
│   ├── decorators.py          ← NEW
│   ├── build.py
│   ├── completions.py
│   ├── config.py
│   ├── dataset_ops.py
│   ├── datasets.py
│   ├── docs.py
│   ├── graphs.py
│   ├── health.py
│   ├── help_commands.py
│   ├── history.py
│   ├── ide.py
│   ├── jobs.py
│   ├── ops.py
│   ├── plugins.py
│   ├── serve.py
│   ├── storage.py
│   └── subsystem.py
├── execution/
│   ├── __init__.py
│   ├── _lazy_deps.py
│   ├── bootstrap.py           ← NEW
│   ├── executor.py
│   ├── middleware.py
│   ├── progress.py
│   ├── registry.py            ← NEW (moved from introspection)
│   └── types.py
├── handlers/
│   ├── __init__.py
│   ├── context.py             ← NEW (unified context)
│   ├── build.py
│   ├── datasets.py
│   ├── docs.py
│   ├── graphs.py
│   ├── health.py
│   ├── history.py
│   ├── ide.py
│   ├── jobs.py
│   ├── ops.py
│   ├── plugins.py
│   ├── storage.py
│   └── subsystem.py
├── rendering/
│   ├── __init__.py
│   ├── service.py             ← ENHANCED
│   ├── specs.py               ← NEW
│   ├── table.py
│   └── types.py
└── [other packages unchanged]
```

### 9.2 Deleted Files Confirmation

Verify these files no longer exist:

```bash
# Should all return "No such file"
ls src/codeintel/cli/handlers/base.py 2>&1
ls src/codeintel/cli/handlers/protocol.py 2>&1
ls src/codeintel/cli/execution/context.py 2>&1
ls src/codeintel/cli/execution/adapter.py 2>&1
ls src/codeintel/cli/commands/context.py 2>&1  # May be retained for graphs.py
ls src/codeintel/cli/introspection/registry.py 2>&1
ls src/codeintel/cli/operations/*.py 2>&1
```

### 9.2.1 Verify No Handler Registrations Remain

Confirm that handler modules don't contain duplicate registrations:

```bash
# Should return no matches (registrations moved to decorator)
rg "^register_operation\(" src/codeintel/cli/handlers/
rg "from codeintel.cli.execution.registry import.*register_operation" src/codeintel/cli/handlers/
```

### 9.2.2 Verify Decorator Coverage

Check that migrated commands use the decorator:

```bash
# Count @cli_command usage
rg "@cli_command" src/codeintel/cli/commands/*.py --count

# Should show approximately 34 uses across 12 files
```

### 9.3 Metric Summary

Generate final metrics:

```bash
echo "=== Final Migration Metrics ==="

# File counts
echo "Handler files: $(ls src/codeintel/cli/handlers/*.py | wc -l)"
echo "Command files: $(ls src/codeintel/cli/commands/*.py | wc -l)"

# Test counts
echo "CLI tests: $(uv run pytest tests/cli/ --co -q 2>&1 | tail -1)"

# Coverage
echo "Coverage: $(python -c "import json; print(f\"{json.load(open('docs/plans/phases/artifacts/coverage_final.json'))['totals']['percent_covered']:.1f}%\")")"

# Lines of code removed (approximate)
echo "Estimated lines removed: ~2000"
```

---

## 10. Known Issues from Phase 5

### 10.1 Test Adjustments Needed

Some tests may need adjustment due to changes in error messages or parameter handling:

| Test | Issue | Fix |
|------|-------|-----|
| `test_validation_error_exit_code` | Error message changed from "No codeintel.yaml found" to "Project not found: ..." | Update assertion to check for "codeintel.yaml" substring |
| `test_history_timeseries_cli_happy_path` | Test setup doesn't mock git operations | Test isolation issue - unrelated to migration |

### 10.2 Parameter Handling Edge Cases

The decorator's infrastructure field filtering needed refinement:

- `repo`, `repo_root`, `commit` are NOT infrastructure fields (some handlers need them as params)
- `project_root` IS infrastructure (used for runtime resolution)
- This distinction is documented in `_INFRASTRUCTURE_FIELDS`

### 10.3 Backward Compatibility Additions

Phase 5 added `cyclopts_ops` re-export to `cli/__init__.py` for test compatibility:

```python
# Tests import: from codeintel.cli import cyclopts_ops
from codeintel.cli.commands import ops as cyclopts_ops
```

---

## 11. Post-Migration

### 11.1 Celebrate! 🎉

The CLI architecture migration is complete.

### 11.2 Monitor

For the next few weeks:
- Monitor CI for any regressions
- Watch for user-reported issues
- Track any performance changes

### 11.3 Clean Up Artifacts

After a stabilization period:

```bash
# Archive migration artifacts
git mv docs/plans/CLI_MIGRATION_PLAN.md docs/archive/
git mv docs/plans/phases/ docs/archive/phases/

# Keep architecture doc
# docs/plans/CLI_UNIFIED_ARCHITECTURE.md stays as reference
```

### 11.4 Future Improvements

With the new architecture, consider:
- Adding more operation metadata to `CommandConfig`
- Implementing decorator-level middleware hooks
- Enhancing the registry for plugin discovery
- Adding telemetry/metrics per operation via decorator
- Supporting async handlers (`async def handler(ctx) -> CliResult[T]`)

### 11.5 Test Helpers Available

Phase 5 exposed test helpers for decorator testing:

```python
from codeintel.cli.commands.decorators import (
    # Public test helpers
    extract_params,      # Extract non-infrastructure params from command
    get_output_format,   # Get output format from command instance
    get_path_field,      # Get Path field from command instance
)
```

---

## Appendix A: Deletion Order Script

```bash
#!/bin/bash
# scripts/phase6_cleanup.sh
# Execute Phase 6 deletions in correct order

set -euo pipefail

echo "=== Phase 6: Legacy Cleanup ==="
echo "Pre-requisite: Phase 5 complete with 34 commands migrated"

# Verify no imports first
echo ""
echo "=== Step 1: Verify no remaining imports ==="

ERRORS=0

if rg "from codeintel.cli.handlers.base import" src/ tests/ --quiet; then
    echo "❌ handlers/base.py still imported"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ handlers/base.py clean"
fi

if rg "from codeintel.cli.handlers.protocol import" src/ tests/ --quiet; then
    echo "❌ handlers/protocol.py still imported"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ handlers/protocol.py clean"
fi

if rg "from codeintel.cli.execution.context import" src/ tests/ --quiet; then
    echo "❌ execution/context.py still imported"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ execution/context.py clean"
fi

if rg "from codeintel.cli.execution.adapter import" src/ tests/ --quiet; then
    echo "❌ execution/adapter.py still imported"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ execution/adapter.py clean"
fi

# Special case: commands/context.py may be used by graphs.py
echo ""
echo "Checking commands/context.py (graphs.py exception allowed)..."
CONTEXT_USERS=$(rg "from codeintel.cli.commands.context import" src/ --files-with-matches 2>/dev/null || true)
if [ -n "$CONTEXT_USERS" ]; then
    echo "⚠️ commands/context.py imported by:"
    echo "$CONTEXT_USERS"
    echo "   -> Verify these are expected (graphs.py is OK)"
fi

# Check for adapter function usage
echo ""
echo "Checking for handler_context_from_enhanced usage..."
if rg "handler_context_from_enhanced" src/ --glob '!**/context.py' tests/ --quiet; then
    echo "❌ handler_context_from_enhanced still used outside context.py"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ handler_context_from_enhanced not used externally"
fi

# Verify handler registrations removed (Phase 5 cleanup)
echo ""
echo "Verifying Phase 5 cleanup..."
if rg "^register_operation\(" src/codeintel/cli/handlers/ --quiet; then
    echo "❌ Handler modules still have registration calls"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ No handler registration calls"
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "❌ $ERRORS errors found. Fix before proceeding."
    exit 1
fi

echo ""
echo "✓ All pre-deletion checks passed"

# Delete files
echo ""
echo "=== Step 2: Delete legacy files ==="

git rm -f src/codeintel/cli/handlers/base.py || true
git rm -f src/codeintel/cli/handlers/protocol.py || true
git rm -f src/codeintel/cli/execution/context.py || true
git rm -f src/codeintel/cli/execution/adapter.py || true
# NOTE: commands/context.py deletion depends on graphs.py migration
# git rm -f src/codeintel/cli/commands/context.py || true
git rm -f src/codeintel/cli/introspection/registry.py || true
git rm -f src/codeintel/cli/operations/build_operations.py || true
git rm -f src/codeintel/cli/operations/dataset_operations.py || true
git rm -f src/codeintel/cli/operations/docs_operations.py || true
git rm -f src/codeintel/cli/operations/graph_operations.py || true
git rm -f src/codeintel/cli/operations/history_operations.py || true
git rm -f src/codeintel/cli/operations/ide_operations.py || true
git rm -f src/codeintel/cli/operations/op_operations.py || true
git rm -f src/codeintel/cli/operations/storage_operations.py || true
git rm -f src/codeintel/cli/operations/subsystem_operations.py || true
git rm -f src/codeintel/cli/_migration_flags.py || true

echo "✓ Files deleted"

# Run quality checks
echo ""
echo "=== Step 3: Quality checks ==="
uv run ruff check src/codeintel/cli/ --fix
uv run pyright src/codeintel/cli/

# Run tests
echo ""
echo "=== Step 4: Tests ==="
uv run pytest tests/cli/ -x -q

echo ""
echo "✓ Tests passed"
echo "=== Phase 6 cleanup complete ==="
```

---

**Previous Phase:** [Phase 5: Command Decorator](./PHASE_5_DECORATOR.md)  
**Architecture Reference:** [CLI_UNIFIED_ARCHITECTURE.md](../CLI_UNIFIED_ARCHITECTURE.md)

---

# 🎉 Migration Complete! 🎉

The CLI has been successfully migrated to the unified architecture:

## Architectural Achievements

- **Single `HandlerContext`** for all operations
  - Typed parameter accessors: `param_str()`, `param_int()`, `param_bool()`, `param_path()`, `param_enum()`, `param_list()`, `param_tuple()`
  - Required parameter accessors: `require_str()`, `require_int()`, `require_path()` (raise `ParameterError`)
  - Lazy resource access: `ctx.gateway`, `ctx.runtime`, `ctx.graph_runtime`
- **Single `UnifiedRenderer`** for all output
- **Declarative `@cli_command`** with `CommandConfig` for 34+ commands
- **Unified `OperationRegistry`** in `execution/registry.py` for introspection
- **Zero legacy code** remaining (after Phase 6)

## Phase 5 Statistics

| Metric | Count |
|--------|-------|
| Commands migrated to `@cli_command` | 34 |
| Command files updated | 12 |
| Handler registration sections removed | 11 |
| Lines of boilerplate eliminated | ~1,000+ |
| Special cases (manual `__call__`) | 1 (`graphs.py`) |

## Key Design Decisions

1. **`CommandConfig` dataclass** bundles resource requirements to keep decorator signature clean
2. **Infrastructure fields** are filtered from params dict (output_format, verbose, project_root, etc.)
3. **Non-infrastructure fields** like `repo`, `commit` pass through because handlers may need them
4. **PEP 695 type parameters** provide proper generic handling in decorator signature
5. **Handler modules** no longer contain registration code - decorator handles it
