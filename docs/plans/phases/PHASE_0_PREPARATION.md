# Phase 0: Preparation — Detailed Implementation Plan

> **Phase:** 0 of 6  
> **Duration:** 1-2 days  
> **Risk Level:** Low  
> **Dependencies:** None  
> **Parallelizable:** No  

---

## Table of Contents

1. [Objectives](#1-objectives)
2. [Prerequisites](#2-prerequisites)
3. [Deliverables](#3-deliverables)
4. [Detailed Tasks](#4-detailed-tasks)
5. [File Changes](#5-file-changes)
6. [Testing Requirements](#6-testing-requirements)
7. [Verification Checklist](#7-verification-checklist)
8. [Exit Criteria](#8-exit-criteria)
9. [Rollback Procedure](#9-rollback-procedure)

---

## 1. Objectives

Phase 0 establishes the foundation for a safe migration by:

1. **Creating a regression baseline** — Capture current test state and coverage
2. **Documenting current state** — Inventory all handlers, commands, and their dependencies
3. **Setting up migration infrastructure** — Optional feature flags for gradual rollout
4. **Validating starting point** — Ensure all tests pass before any changes

This phase involves **no production code changes** — only documentation, tooling, and baseline capture.

---

## 2. Prerequisites

### 2.1 Environment

- [ ] Clean git working tree (`git status` shows no uncommitted changes)
- [ ] All CI checks passing on main branch
- [ ] Development environment bootstrapped (`scripts/bootstrap.sh` completed)
- [ ] `uv sync` completed successfully

### 2.2 Access

- [ ] Write access to `docs/plans/` directory
- [ ] Ability to run full test suite locally

---

## 3. Deliverables

### 3.1 Test Baseline Report

**File:** `docs/plans/phases/artifacts/test_baseline_report.md`

Contents:
- Total test count for `tests/cli/`
- Test pass/fail/skip counts
- Coverage percentage for `src/codeintel/cli/`
- List of any currently failing or skipped tests
- Timestamp of baseline capture

### 3.2 Handler Inventory

**File:** `docs/plans/phases/artifacts/handler_inventory.md`

A comprehensive table documenting every handler file:

| Handler File | Handler Functions | Context Type | Param Helpers | Runtime Req | Gateway Req | Graph Runtime Req |
|--------------|-------------------|--------------|---------------|-------------|-------------|-------------------|
| Example | List of functions | Type used | Local/None | Yes/No | Yes/No | Yes/No |

### 3.3 Command Inventory

**File:** `docs/plans/phases/artifacts/command_inventory.md`

A comprehensive table documenting every command file:

| Command File | Command Classes | Has `__call__` | Uses `command_context` | Runtime Req |
|--------------|-----------------|----------------|------------------------|-------------|
| Example | List of classes | Yes/No | Yes/No | Yes/No |

### 3.4 Feature Flag Module (Optional)

**File:** `src/codeintel/cli/_migration_flags.py`

Temporary module for controlling migration rollout. **Will be deleted in Phase 6.**

---

## 4. Detailed Tasks

### Task P0-1: Run Full CLI Test Suite and Capture Baseline

**Duration:** 1 hour

**Steps:**

1. Navigate to project root
2. Run the full CLI test suite with verbose output:
   ```bash
   uv run pytest tests/cli/ -v --tb=short 2>&1 | tee docs/plans/phases/artifacts/test_baseline_output.txt
   ```
3. Capture summary statistics:
   ```bash
   uv run pytest tests/cli/ --co -q 2>&1 | tail -5 > docs/plans/phases/artifacts/test_count.txt
   ```

**Output:**
- `docs/plans/phases/artifacts/test_baseline_output.txt` — Full test output
- `docs/plans/phases/artifacts/test_count.txt` — Test count summary

---

### Task P0-2: Generate Coverage Report for CLI

**Duration:** 1 hour

**Steps:**

1. Run pytest with coverage:
   ```bash
   uv run pytest tests/cli/ \
     --cov=src/codeintel/cli \
     --cov-report=html:docs/plans/phases/artifacts/htmlcov_baseline \
     --cov-report=json:docs/plans/phases/artifacts/coverage_baseline.json \
     --cov-report=term-missing
   ```

2. Extract key metrics from JSON report:
   ```python
   import json
   with open("docs/plans/phases/artifacts/coverage_baseline.json") as f:
       data = json.load(f)
   print(f"Total coverage: {data['totals']['percent_covered']:.1f}%")
   ```

**Output:**
- `docs/plans/phases/artifacts/htmlcov_baseline/` — HTML coverage report
- `docs/plans/phases/artifacts/coverage_baseline.json` — JSON coverage data

---

### Task P0-3: Create Handler Inventory

**Duration:** 2 hours

**Steps:**

1. List all handler files:
   ```bash
   ls -la src/codeintel/cli/handlers/*.py
   ```

2. For each handler file, document:
   - All public handler functions (functions ending in `_handler`)
   - Context type used (`EnhancedHandlerContext`, `HandlerContext`, etc.)
   - Local param helper functions (`_get_str_param`, `_get_int_param`, etc.)
   - Whether handler requires runtime (`ctx.runtime` access)
   - Whether handler requires gateway (`ctx.gateway` access)
   - Whether handler requires graph_runtime (`ctx.graph_runtime` access)

3. Create the inventory document:

**File:** `docs/plans/phases/artifacts/handler_inventory.md`

```markdown
# Handler Inventory

> Generated: [DATE]
> Purpose: Track handler migration status for Phase 3

## Summary

| Metric | Count |
|--------|-------|
| Total handler files | X |
| Total handler functions | X |
| Files with local param helpers | X |
| Files requiring runtime | X |
| Files requiring gateway | X |
| Files requiring graph_runtime | X |

## Handler Details

### handlers/jobs.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `jobs_list_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_int_param` | No | No | No |
| `jobs_status_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| ... | ... | ... | ... | ... | ... |

### handlers/health.py
...

### handlers/ops.py
...

[Continue for all handler files]
```

**Specific handler files to inventory:**
- `handlers/jobs.py`
- `handlers/health.py`
- `handlers/ops.py`
- `handlers/storage.py`
- `handlers/history.py`
- `handlers/build.py`
- `handlers/docs.py`
- `handlers/graphs.py`
- `handlers/ide.py`
- `handlers/datasets.py`
- `handlers/plugins.py`
- `handlers/subsystem.py`

---

### Task P0-4: Create Command Inventory

**Duration:** 2 hours

**Steps:**

1. List all command files:
   ```bash
   ls -la src/codeintel/cli/commands/*.py
   ```

2. For each command file (excluding `__init__.py`, `_common.py`, `_help.py`, `app.py`), document:
   - All command classes (dataclasses with `@app.command` decorator)
   - Whether each has a `__call__` method
   - Whether it uses `command_context`
   - What params are extracted manually
   - Whether runtime is required (`require_runtime` parameter)

3. Create the inventory document:

**File:** `docs/plans/phases/artifacts/command_inventory.md`

```markdown
# Command Inventory

> Generated: [DATE]
> Purpose: Track command migration status for Phase 5

## Summary

| Metric | Count |
|--------|-------|
| Total command files | X |
| Total command classes | X |
| Commands with `__call__` | X |
| Commands using `command_context` | X |
| Commands requiring runtime | X |

## Command Details

### commands/jobs.py

| Command Class | Has `__call__` | Uses `command_context` | Runtime Required | Params |
|---------------|----------------|------------------------|------------------|--------|
| `JobsListCommand` | Yes | Yes | No | status, limit |
| `JobsStatusCommand` | Yes | Yes | No | job_id |
| ... | ... | ... | ... | ... |

### commands/health.py
...

[Continue for all command files]
```

**Specific command files to inventory:**
- `commands/jobs.py`
- `commands/health.py`
- `commands/ops.py`
- `commands/storage.py`
- `commands/history.py`
- `commands/build.py`
- `commands/docs.py`
- `commands/graphs.py`
- `commands/ide.py`
- `commands/datasets.py`
- `commands/dataset_ops.py`
- `commands/plugins.py`
- `commands/subsystem.py`
- `commands/serve.py`
- `commands/config.py`
- `commands/completions.py`

---

### Task P0-5: Create Feature Flag Module (Optional)

**Duration:** 1 hour

This task is optional but recommended for large teams or when gradual rollout is desired.

**File:** `src/codeintel/cli/_migration_flags.py`

```python
"""Temporary feature flags for CLI migration.

WARNING: This module is temporary scaffolding for the CLI architecture
migration. It will be DELETED in Phase 6 of the migration.

Do not add permanent feature flags here. This module exists solely to
enable gradual rollout of new code paths during the migration.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# Migration Feature Flags
# -----------------------------------------------------------------------------
# These flags control which code paths are used during the migration.
# Set via environment variables for testing/rollout.
# -----------------------------------------------------------------------------

# Phase 3: Use new HandlerContext in handlers
# When True, handlers expect the new unified HandlerContext
# When False (default), handlers use EnhancedHandlerContext
USE_NEW_HANDLER_CONTEXT: bool = os.environ.get(
    "CODEINTEL_CLI_NEW_CONTEXT", "0"
) == "1"

# Phase 2: Use UnifiedRenderer everywhere
# When True, executor uses UnifiedRenderer from service.py
# When False (default), executor uses renderers from renderers.py
USE_UNIFIED_RENDERER: bool = os.environ.get(
    "CODEINTEL_CLI_UNIFIED_RENDERER", "0"
) == "1"

# Phase 5: Use @cli_command decorator
# When True, new decorator-based commands are active
# When False (default), traditional __call__ commands are used
USE_CLI_COMMAND_DECORATOR: bool = os.environ.get(
    "CODEINTEL_CLI_DECORATOR", "0"
) == "1"


def log_migration_flags() -> None:
    """Log current migration flag states.
    
    Call at CLI startup to log which migration features are enabled.
    Useful for debugging and rollout verification.
    """
    import logging
    
    log = logging.getLogger(__name__)
    log.debug("Migration flags: context=%s, renderer=%s, decorator=%s",
              USE_NEW_HANDLER_CONTEXT,
              USE_UNIFIED_RENDERER,
              USE_CLI_COMMAND_DECORATOR)


__all__ = [
    "USE_NEW_HANDLER_CONTEXT",
    "USE_UNIFIED_RENDERER", 
    "USE_CLI_COMMAND_DECORATOR",
    "log_migration_flags",
]
```

---

### Task P0-6: Document Any Currently Failing Tests

**Duration:** 1 hour

**Steps:**

1. Review test baseline output for failures:
   ```bash
   grep -E "^(FAILED|ERROR)" docs/plans/phases/artifacts/test_baseline_output.txt
   ```

2. For each failing test, document:
   - Test file and function name
   - Error type
   - Whether it's a known issue
   - Whether it blocks migration

3. Create documentation:

**File:** `docs/plans/phases/artifacts/known_test_issues.md`

```markdown
# Known Test Issues at Migration Start

> Generated: [DATE]
> Purpose: Document pre-existing test failures to avoid confusion during migration

## Summary

| Category | Count |
|----------|-------|
| Failing tests | X |
| Skipped tests | X |
| Known issues (not migration-related) | X |
| Blocking issues | X |

## Failing Tests

### test_cli/test_example.py::test_function

- **Error:** [Error message]
- **Known issue:** Yes/No
- **Blocks migration:** Yes/No
- **Notes:** [Any relevant context]

## Skipped Tests

[List any tests marked with @pytest.mark.skip]

## Action Items

- [ ] [Any issues that should be fixed before starting Phase 1]
```

---

### Task P0-7: Create Migration Tracking Document

**Duration:** 1 hour

**File:** `docs/plans/phases/MIGRATION_TRACKING.md`

```markdown
# CLI Migration Tracking

> **Started:** [DATE]
> **Target Completion:** [DATE + 6 weeks]
> **Current Phase:** 0 (Preparation)

## Phase Status

| Phase | Name | Status | Started | Completed | Notes |
|-------|------|--------|---------|-----------|-------|
| 0 | Preparation | 🟡 In Progress | [DATE] | | |
| 1 | Foundation Layer | ⚪ Not Started | | | |
| 2 | Rendering Consolidation | ⚪ Not Started | | | |
| 3 | Handler Migration | ⚪ Not Started | | | |
| 4 | Registry Unification | ⚪ Not Started | | | |
| 5 | Command Decorator | ⚪ Not Started | | | |
| 6 | Legacy Cleanup | ⚪ Not Started | | | |

## Key Metrics

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Test count | X | X | >= X |
| Test pass rate | X% | X% | 100% |
| CLI coverage | X% | X% | >= X% |
| Handler files migrated | 0/12 | 0/12 | 12/12 |
| Command files migrated | 0/16 | 0/16 | 16/16 |
| Legacy files deleted | 0/15 | 0/15 | 15/15 |

## Daily Log

### [DATE]
- Started Phase 0
- Created baseline test report
- ...

## Blockers

[List any blockers with owner and ETA]

## Decisions

[Log any architectural decisions made during migration]
```

---

## 5. File Changes

### 5.1 New Files Created

| File | Type | Purpose |
|------|------|---------|
| `docs/plans/phases/artifacts/` | Directory | Artifacts storage |
| `docs/plans/phases/artifacts/test_baseline_report.md` | Documentation | Test baseline |
| `docs/plans/phases/artifacts/test_baseline_output.txt` | Log | Raw test output |
| `docs/plans/phases/artifacts/test_count.txt` | Log | Test count summary |
| `docs/plans/phases/artifacts/coverage_baseline.json` | JSON | Coverage data |
| `docs/plans/phases/artifacts/htmlcov_baseline/` | HTML | Coverage report |
| `docs/plans/phases/artifacts/handler_inventory.md` | Documentation | Handler catalog |
| `docs/plans/phases/artifacts/command_inventory.md` | Documentation | Command catalog |
| `docs/plans/phases/artifacts/known_test_issues.md` | Documentation | Pre-existing failures |
| `docs/plans/phases/MIGRATION_TRACKING.md` | Documentation | Progress tracking |
| `src/codeintel/cli/_migration_flags.py` | Python | Feature flags (optional) |

### 5.2 Files Modified

None — Phase 0 makes no changes to production code.

### 5.3 Files Deleted

None.

---

## 6. Testing Requirements

### 6.1 Pre-Phase Validation

Before starting Phase 0 tasks:

```bash
# Verify environment is clean
git status

# Verify tests pass
uv run pytest tests/cli/ -x -q
```

### 6.2 Post-Phase Validation

After completing Phase 0:

```bash
# Verify no production code was changed
git diff src/codeintel/cli/ --stat

# Should show only _migration_flags.py if created
# All other changes should be in docs/

# Verify tests still pass (unchanged)
uv run pytest tests/cli/ -x -q
```

---

## 7. Verification Checklist

### 7.1 Artifacts Complete

- [ ] `docs/plans/phases/artifacts/` directory created
- [ ] Test baseline report exists and is accurate
- [ ] Coverage baseline JSON exists
- [ ] Coverage HTML report generated
- [ ] Handler inventory is complete (all 12 handler files documented)
- [ ] Command inventory is complete (all 16 command files documented)
- [ ] Known test issues documented
- [ ] Migration tracking document created

### 7.2 Inventories Validated

- [ ] Handler inventory reviewed for accuracy
- [ ] Command inventory reviewed for accuracy
- [ ] Param helper functions identified in each handler
- [ ] Runtime/Gateway requirements correctly identified

### 7.3 No Regressions

- [ ] All tests that passed before still pass
- [ ] No production code changes (except optional feature flags)
- [ ] Git status clean for `src/` (except `_migration_flags.py`)

---

## 8. Exit Criteria

Phase 0 is complete when:

| Criterion | Status |
|-----------|--------|
| Test baseline documented | ⬜ |
| Coverage baseline captured | ⬜ |
| Handler inventory complete (12 files) | ⬜ |
| Command inventory complete (16 files) | ⬜ |
| Known issues documented | ⬜ |
| Migration tracking initialized | ⬜ |
| Feature flag module created (if using) | ⬜ |
| All existing tests still pass | ⬜ |
| No unplanned code changes | ⬜ |

---

## 9. Rollback Procedure

Phase 0 requires no rollback because no production code is changed.

If issues are discovered:
1. Delete generated artifacts in `docs/plans/phases/artifacts/`
2. Delete `src/codeintel/cli/_migration_flags.py` if created
3. Re-run Phase 0 from the beginning

---

## Appendix A: Automation Scripts

### A.1 Generate Handler Inventory Script

```python
#!/usr/bin/env python3
"""Generate handler inventory for Phase 0.

Usage: python scripts/generate_handler_inventory.py > handler_inventory.md
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def analyze_handler_file(path: Path) -> dict:
    """Analyze a handler file for functions and helpers."""
    source = path.read_text()
    tree = ast.parse(source)
    
    handlers = []
    param_helpers = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.endswith("_handler"):
                handlers.append(node.name)
            elif node.name.startswith("_get_") or node.name.startswith("_require_"):
                param_helpers.append(node.name)
    
    return {
        "file": path.name,
        "handlers": handlers,
        "param_helpers": param_helpers,
    }


def main() -> int:
    """Generate the handler inventory."""
    handlers_dir = Path("src/codeintel/cli/handlers")
    
    print("# Handler Inventory\n")
    print(f"> Generated: {__import__('datetime').datetime.now().isoformat()}\n")
    
    for path in sorted(handlers_dir.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        
        info = analyze_handler_file(path)
        print(f"\n## {info['file']}\n")
        print(f"- Handlers: {', '.join(info['handlers']) or 'None'}")
        print(f"- Param helpers: {', '.join(info['param_helpers']) or 'None'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### A.2 Baseline Capture Script

```bash
#!/bin/bash
# scripts/capture_baseline.sh
# Capture test and coverage baseline for Phase 0

set -euo pipefail

ARTIFACTS_DIR="docs/plans/phases/artifacts"
mkdir -p "$ARTIFACTS_DIR"

echo "=== Capturing test baseline ==="
uv run pytest tests/cli/ -v --tb=short 2>&1 | tee "$ARTIFACTS_DIR/test_baseline_output.txt"

echo "=== Capturing test count ==="
uv run pytest tests/cli/ --co -q 2>&1 | tail -5 > "$ARTIFACTS_DIR/test_count.txt"

echo "=== Capturing coverage ==="
uv run pytest tests/cli/ \
    --cov=src/codeintel/cli \
    --cov-report=html:"$ARTIFACTS_DIR/htmlcov_baseline" \
    --cov-report=json:"$ARTIFACTS_DIR/coverage_baseline.json"

echo "=== Baseline capture complete ==="
echo "Artifacts in: $ARTIFACTS_DIR"
```

---

**Next Phase:** [Phase 1: Foundation Layer](./PHASE_1_FOUNDATION.md)
