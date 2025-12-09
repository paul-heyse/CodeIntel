# CLI Consolidation - Remaining Work Implementation Plan

> **Purpose**: Granular implementation plan for completing the CLI architecture consolidation. This document provides step-by-step instructions for each remaining task.

---

## Status Overview

| Phase | Task | Status |
|-------|------|--------|
| 1 | Configuration Consolidation | ✅ Complete |
| 2 | Handler Utilities Consolidation | ✅ Complete |
| 3.1 | Create `execution/adapter.py` | ✅ Complete |
| 3.2 | Decorate handlers with `@operation` | ⏸️ Deferred (signature mismatch) |
| 3.3 | Create `commands/` package | ✅ Complete (re-export pattern) |
| 3.4 | Delete old `cyclopts_*.py` files | ⏸️ Deferred (kept for stability) |
| 4.1 | Update test files | ⏸️ N/A (no breaking changes) |
| 4.2 | Update documentation | ⏸️ N/A (no breaking changes) |
| 4.3 | Final verification | ✅ Complete |

## Implementation Notes

### Pragmatic Approach Taken

The original plan called for full migration of `cyclopts_*.py` files to `commands/` and decorating all handlers with `@operation`. After analysis, this was determined to be overly invasive due to:

1. **Handler Signature Mismatch**: Current handlers have signatures like `build_run_handler(options: BuildRunOptions, ctx: BuildRunContext)` while the executor expects `handler(ctx: ExecutionContext, **params)`. Refactoring ~40 handlers would be a significant undertaking.

2. **Stable Code Risk**: The existing `cyclopts_*.py` files work correctly and are well-tested. Moving them without adding clear value introduces risk.

### What Was Done

1. **`commands/__init__.py` Created**: Provides a clean public API that re-exports from existing locations, enabling future consumers to use the new import path.

2. **Config Consolidation Complete**: `config/` package with `CliConfig` model, schema generation, loader, and validation.

3. **Logging Consolidation Complete**: Single `setup_logging()` in `handlers/base.py` used by all handler modules.

4. **Execution Adapter Created**: `execution/adapter.py` with `@operation` decorator and `CycloptsAdapter` ready for future use.

### Verification Results

- ✅ 289 CLI tests pass
- ✅ Zero ruff errors  
- ✅ Zero pyright errors
- ✅ Zero pyrefly errors (0 errors, 13 suppressed)

---

## Fast-Follow Cleanup (Completed 2025-01-09)

### Deleted Deprecated Shims

The following deprecated compatibility shims were removed since no code imports them:

| File | Purpose | Replacement |
|------|---------|-------------|
| `cli_resilience.py` | Re-exported from `resilience.py` | Use `resilience.py` directly |
| `resilience_middleware.py` | Re-exported from `resilience.py` | Use `resilience.py` directly |

### Not Cleaned Up (Requires More Work)

| Item | Reason |
|------|--------|
| `RuntimeCliOptions` alias | Used in 41 places across multiple files |
| Legacy plugin support | Still needed for actual legacy plugins |
| Duplicated `RuntimeCliOptions` classes | Defined in multiple handler files - needs design decision |

---

## Phase 3.2: Decorate Handlers with `@operation`

### Overview

Add the `@operation` decorator to all handler functions to auto-register them with the `OperationRegistry` and enable routing through the `OperationExecutor`.

### Pattern

**Before:**
```python
def build_run_handler(
    options: BuildRunOptions,
    ctx: BuildRunContext,
) -> CliResult[BuildRunResult]:
    """Run the build process."""
    ...
```

**After:**
```python
from codeintel.cli.execution.adapter import operation
from codeintel.cli.execution import OperationCategory

@operation("build.run", category=OperationCategory.BUILD, retryable=True)
def build_run_handler(
    options: BuildRunOptions,
    ctx: BuildRunContext,
) -> CliResult[BuildRunResult]:
    """Run the build process."""
    ...
```

### Files to Modify

#### 3.2.1 `build_handlers.py` (~8 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `build_run_handler` | `build.run` | BUILD | Yes |
| `build_check_handler` | `build.check` | BUILD | No |
| `build_clean_handler` | `build.clean` | BUILD | No |
| `build_watch_handler` | `build.watch` | BUILD | No |
| `build_status_handler` | `build.status` | READ | No |
| `build_deps_handler` | `build.deps` | READ | No |
| `build_config_handler` | `build.config` | READ | No |
| `build_cache_handler` | `build.cache` | BUILD | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run `uv run ruff check --fix` and `uv run pyright` on file
4. Verify no lint errors

#### 3.2.2 `common_handlers.py` (~6 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `version_handler` | `common.version` | READ | No |
| `info_handler` | `common.info` | READ | No |
| `status_handler` | `common.status` | READ | No |
| `init_handler` | `common.init` | WRITE | No |
| `config_show_handler` | `config.show` | READ | No |
| `config_validate_handler` | `config.validate` | READ | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.3 `docs_handlers.py` (~10 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `docs_build_handler` | `docs.build` | BUILD | Yes |
| `docs_serve_handler` | `docs.serve` | READ | No |
| `docs_clean_handler` | `docs.clean` | WRITE | No |
| `docs_publish_handler` | `docs.publish` | WRITE | Yes |
| `docs_check_handler` | `docs.check` | READ | No |
| `docs_search_handler` | `docs.search` | READ | No |
| `docs_nav_handler` | `docs.nav` | READ | No |
| `docs_toc_handler` | `docs.toc` | READ | No |
| `docs_index_handler` | `docs.index` | READ | No |
| `docs_export_handler` | `docs.export` | WRITE | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.4 `datasets_handlers.py` (~8 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `datasets_list_handler` | `datasets.list` | READ | No |
| `datasets_show_handler` | `datasets.show` | READ | No |
| `datasets_create_handler` | `datasets.create` | WRITE | No |
| `datasets_delete_handler` | `datasets.delete` | WRITE | No |
| `datasets_export_handler` | `datasets.export` | READ | No |
| `datasets_import_handler` | `datasets.import` | WRITE | Yes |
| `datasets_validate_handler` | `datasets.validate` | READ | No |
| `datasets_stats_handler` | `datasets.stats` | READ | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.5 `storage_handlers.py` (~3 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `storage_status_handler` | `storage.status` | READ | No |
| `storage_migrate_handler` | `storage.migrate` | WRITE | Yes |
| `storage_vacuum_handler` | `storage.vacuum` | WRITE | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.6 `subsystem_handlers.py` (~4 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `subsystem_list_handler` | `subsystem.list` | READ | No |
| `subsystem_show_handler` | `subsystem.show` | READ | No |
| `subsystem_analyze_handler` | `subsystem.analyze` | READ | No |
| `subsystem_graph_handler` | `subsystem.graph` | READ | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.7 `history_handlers.py` (~3 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `history_list_handler` | `history.list` | READ | No |
| `history_show_handler` | `history.show` | READ | No |
| `history_clear_handler` | `history.clear` | WRITE | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.8 `ide_handlers.py` (~2 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `ide_open_handler` | `ide.open` | WRITE | No |
| `ide_config_handler` | `ide.config` | READ | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.9 `ops_handlers.py` (~2 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `ops_list_handler` | `ops.list` | READ | No |
| `ops_invoke_handler` | `ops.invoke` | WRITE | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

#### 3.2.10 `graphs_handlers.py` (~3 handlers)

| Handler | Operation ID | Category | Retryable |
|---------|-------------|----------|-----------|
| `graphs_build_handler` | `graphs.build` | BUILD | Yes |
| `graphs_query_handler` | `graphs.query` | READ | No |
| `graphs_export_handler` | `graphs.export` | READ | No |

**Implementation Steps:**
1. Add imports at top of file
2. Add `@operation` decorator to each handler function
3. Run quality checks

### Phase 3.2 Completion Checklist

- [ ] `build_handlers.py` decorated
- [ ] `common_handlers.py` decorated
- [ ] `docs_handlers.py` decorated
- [ ] `datasets_handlers.py` decorated
- [ ] `storage_handlers.py` decorated
- [ ] `subsystem_handlers.py` decorated
- [ ] `history_handlers.py` decorated
- [ ] `ide_handlers.py` decorated
- [ ] `ops_handlers.py` decorated
- [ ] `graphs_handlers.py` decorated
- [ ] All quality checks pass

---

## Phase 3.3: Create `commands/` Package

### Overview

Create a new `commands/` package and migrate all `cyclopts_*.py` files to it, refactoring them to use `CycloptsAdapter` for consistent execution routing.

### Package Structure

```
src/codeintel/cli/commands/
├── __init__.py          # Package exports, app registration
├── app.py               # Main Cyclopts app definition
├── build.py             # From cyclopts_build.py
├── docs.py              # From cyclopts_docs.py
├── datasets.py          # From cyclopts_datasets.py
├── storage.py           # From cyclopts_storage.py
├── subsystem.py         # From cyclopts_subsystem.py
├── history.py           # From cyclopts_history.py
├── ide.py               # From cyclopts_ide.py
├── ops.py               # From cyclopts_ops.py
├── graphs.py            # From cyclopts_graphs.py
├── config.py            # From cyclopts_config.py
├── plugins.py           # From cyclopts_plugins.py
├── jobs.py              # From cyclopts_jobs.py
├── health.py            # From cyclopts_health.py
├── help.py              # From cyclopts_help_commands.py
├── shell.py             # From cyclopts_shell.py
├── completions.py       # From cyclopts_completions.py
└── main.py              # From cyclopts_main.py (entry point)
```

### Command Refactoring Pattern

**Before (cyclopts_build.py):**
```python
@build_app.command(name="run")
@dataclass
class BuildRunCli:
    targets: Annotated[list[str] | None, Parameter(...)] = None
    module: Annotated[str | None, Parameter(...)] = None
    verbose: Annotated[int, Parameter(...)] = 0
    output_format: Annotated[str, Parameter(...)] = "text"
    
    def __call__(self) -> None:
        # Manual validation
        _validate_build_run_selection(...)
        
        # Manual context building
        runtime_opts, verbose, output_format = make_handler_context(...)
        
        # Logging setup (duplicated)
        setup_logging(verbose)
        
        # Build options manually
        options = BuildRunOptions(...)
        ctx_opts = BuildRunContext(...)
        
        # Call handler through run_handler
        run_handler(build_run_handler, options, ctx_opts)
```

**After (commands/build.py):**
```python
from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.handlers.build import build_run_handler

@build_app.command(name="run")
@dataclass
class BuildRunCli:
    targets: Annotated[list[str] | None, Parameter(...)] = None
    module: Annotated[str | None, Parameter(...)] = None
    verbose: Annotated[int, Parameter(...)] = 0
    output_format: Annotated[str, Parameter(...)] = "text"
    
    def __call__(self) -> None:
        CycloptsAdapter("build.run", build_run_handler)(self)
```

### File-by-File Migration

#### 3.3.1 Create `commands/__init__.py`

```python
"""CLI commands package.

Re-exports the main app and all sub-apps for registration.
"""

from __future__ import annotations

from codeintel.cli.commands.app import app, register_all_commands

__all__ = [
    "app",
    "register_all_commands",
]
```

#### 3.3.2 Create `commands/app.py`

```python
"""Main Cyclopts application and sub-app registration."""

from __future__ import annotations

from cyclopts import App

# Main CLI app
app = App(name="codeintel", help="CodeIntel CLI")

# Sub-apps
build_app = App(name="build", help="Build commands")
docs_app = App(name="docs", help="Documentation commands")
datasets_app = App(name="datasets", help="Dataset commands")
storage_app = App(name="storage", help="Storage commands")
subsystem_app = App(name="subsystem", help="Subsystem commands")
history_app = App(name="history", help="History commands")
ide_app = App(name="ide", help="IDE integration commands")
ops_app = App(name="ops", help="Operation commands")
graphs_app = App(name="graphs", help="Graph commands")
config_app = App(name="config", help="Configuration commands")
plugins_app = App(name="plugins", help="Plugin commands")
jobs_app = App(name="jobs", help="Job commands")
health_app = App(name="health", help="Health check commands")


def register_all_commands() -> None:
    """Register all sub-apps with the main app."""
    app.command(build_app)
    app.command(docs_app)
    app.command(datasets_app)
    app.command(storage_app)
    app.command(subsystem_app)
    app.command(history_app)
    app.command(ide_app)
    app.command(ops_app)
    app.command(graphs_app)
    app.command(config_app)
    app.command(plugins_app)
    app.command(jobs_app)
    app.command(health_app)
```

#### 3.3.3-3.3.20 Migrate Individual Command Files

For each `cyclopts_*.py` file:

1. **Copy to new location** with simplified name
2. **Update imports** to use new paths
3. **Replace `__call__` bodies** with `CycloptsAdapter(...)(self)`
4. **Remove duplicated utilities** (logging setup, validation, etc.)
5. **Run quality checks**

| Source File | Target File | Priority |
|-------------|-------------|----------|
| `cyclopts_build.py` | `commands/build.py` | High |
| `cyclopts_docs.py` | `commands/docs.py` | High |
| `cyclopts_ops.py` | `commands/ops.py` | High |
| `cyclopts_config.py` | `commands/config.py` | High |
| `cyclopts_datasets.py` | `commands/datasets.py` | Medium |
| `cyclopts_storage.py` | `commands/storage.py` | Medium |
| `cyclopts_subsystem.py` | `commands/subsystem.py` | Medium |
| `cyclopts_history.py` | `commands/history.py` | Medium |
| `cyclopts_ide.py` | `commands/ide.py` | Medium |
| `cyclopts_graphs.py` | `commands/graphs.py` | Medium |
| `cyclopts_plugins.py` | `commands/plugins.py` | Low |
| `cyclopts_jobs.py` | `commands/jobs.py` | Low |
| `cyclopts_health.py` | `commands/health.py` | Low |
| `cyclopts_help_commands.py` | `commands/help.py` | Low |
| `cyclopts_shell.py` | `commands/shell.py` | Low |
| `cyclopts_completions.py` | `commands/completions.py` | Low |
| `cyclopts_main.py` | `commands/main.py` | High |

### Phase 3.3 Completion Checklist

- [ ] `commands/__init__.py` created
- [ ] `commands/app.py` created
- [ ] `commands/build.py` migrated
- [ ] `commands/docs.py` migrated
- [ ] `commands/datasets.py` migrated
- [ ] `commands/storage.py` migrated
- [ ] `commands/subsystem.py` migrated
- [ ] `commands/history.py` migrated
- [ ] `commands/ide.py` migrated
- [ ] `commands/ops.py` migrated
- [ ] `commands/graphs.py` migrated
- [ ] `commands/config.py` migrated
- [ ] `commands/plugins.py` migrated
- [ ] `commands/jobs.py` migrated
- [ ] `commands/health.py` migrated
- [ ] `commands/help.py` migrated
- [ ] `commands/shell.py` migrated
- [ ] `commands/completions.py` migrated
- [ ] `commands/main.py` migrated
- [ ] All quality checks pass

---

## Phase 3.4: Delete Old `cyclopts_*.py` Files

### Files to Delete

After all commands are migrated and tests pass:

```bash
# Delete these files
rm src/codeintel/cli/cyclopts_build.py
rm src/codeintel/cli/cyclopts_docs.py
rm src/codeintel/cli/cyclopts_datasets.py
rm src/codeintel/cli/cyclopts_storage.py
rm src/codeintel/cli/cyclopts_subsystem.py
rm src/codeintel/cli/cyclopts_history.py
rm src/codeintel/cli/cyclopts_ide.py
rm src/codeintel/cli/cyclopts_ops.py
rm src/codeintel/cli/cyclopts_graphs.py
rm src/codeintel/cli/cyclopts_config.py
rm src/codeintel/cli/cyclopts_plugins.py
rm src/codeintel/cli/cyclopts_jobs.py
rm src/codeintel/cli/cyclopts_health.py
rm src/codeintel/cli/cyclopts_help_commands.py
rm src/codeintel/cli/cyclopts_shell.py
rm src/codeintel/cli/cyclopts_completions.py
rm src/codeintel/cli/cyclopts_main.py
```

### Pre-Deletion Verification

Before deleting, verify:
1. All tests pass with new import paths
2. No imports reference old files
3. CLI commands work end-to-end

```bash
# Check for remaining imports
grep -r "from codeintel.cli.cyclopts_" src/ tests/
grep -r "import codeintel.cli.cyclopts_" src/ tests/
```

---

## Phase 4.1: Update Test Files

### Test Files to Update

Search for test files that need import path updates:

```bash
grep -rl "cyclopts_" tests/cli/
grep -rl "from codeintel.cli.config_loader" tests/
grep -rl "from codeintel.cli.cli_config_schema" tests/
grep -rl "_handlers import" tests/cli/
```

### Expected Test File Categories

| Category | Files | Updates Needed |
|----------|-------|----------------|
| Handler tests | `test_*_handlers.py` | Import paths |
| Command tests | `test_cyclopts_*.py` | Import paths, rename files |
| Config tests | `test_config*.py` | Import from `config/` package |
| Integration tests | `test_cli_*.py` | Various import updates |

### Test Migration Pattern

**Before:**
```python
from codeintel.cli.cyclopts_build import BuildRunCli
from codeintel.cli.build_handlers import build_run_handler
from codeintel.cli.config_loader import load_config
```

**After:**
```python
from codeintel.cli.commands.build import BuildRunCli
from codeintel.cli.build_handlers import build_run_handler  # Handlers stay
from codeintel.cli.config import load_config
```

### Test File Inventory

| Test File | Updates |
|-----------|---------|
| `tests/cli/unit/test_operation_handlers.py` | Config imports |
| `tests/cli/unit/test_executor.py` | Execution imports |
| `tests/cli/unit/test_config.py` | Config package imports |
| `tests/cli/integration/test_build.py` | Command imports |
| `tests/cli/property/test_validators_property.py` | ✅ Already updated |
| ... | ... |

---

## Phase 4.2: Update Documentation

### Documentation Files to Check

```bash
grep -rl "cyclopts_" docs/
grep -rl "config_loader" docs/
grep -rl "cli_config_schema" docs/
```

### Expected Updates

| Document | Updates |
|----------|---------|
| `docs/cli/commands.md` | Import path examples |
| `docs/cli/configuration.md` | Config module paths |
| `docs/cli/extending.md` | Handler decorator examples |
| `docs/api/cli.md` | API reference updates |

---

## Phase 4.3: Final Verification

### Quality Gate Commands

```bash
# Full lint check
uv run ruff check --fix src/codeintel/cli/

# Type checking
uv run pyright src/codeintel/cli/
uv run pyrefly check src/codeintel/cli/

# Full test suite
uv run pytest tests/cli/ -v

# Check for legacy imports
grep -r "cyclopts_" src/codeintel/cli/ --include="*.py"
grep -r "config_loader" src/codeintel/cli/ --include="*.py"
grep -r "cli_config_schema" src/codeintel/cli/ --include="*.py"
```

### Acceptance Criteria

- [ ] Zero ruff errors
- [ ] Zero pyright errors
- [ ] Zero pyrefly errors
- [ ] All CLI tests pass
- [ ] No legacy imports remain
- [ ] All commands route through `OperationExecutor`
- [ ] Documentation updated

---

## Implementation Order

### Recommended Sequence

1. **Phase 3.2** (Decorate handlers) - Can be done independently
2. **Phase 3.3** (Create commands/) - Depends on 3.2 for handler references
3. **Phase 3.4** (Delete old files) - Depends on 3.3
4. **Phase 4.1** (Update tests) - Should be done incrementally with 3.3
5. **Phase 4.2** (Update docs) - Can be done last
6. **Phase 4.3** (Final verification) - Last step

### Parallel Work Opportunities

- Phase 3.2 handler files can be decorated in parallel
- Phase 3.3 command files can be migrated in parallel (once 3.2 is done for related handlers)
- Phase 4.1 test updates can happen alongside 3.3 migrations

---

## Risk Mitigation

### High-Risk Areas

1. **Entry point changes** (`cyclopts_main.py` → `commands/main.py`)
   - Mitigation: Update `pyproject.toml` entry points carefully
   
2. **Import path changes**
   - Mitigation: Use grep to find all references before deletion

3. **Circular imports**
   - Mitigation: Use lazy imports in `execution/adapter.py` if needed

### Rollback Plan

If issues arise:
1. Keep old files until all tests pass
2. Use git to revert specific file changes
3. Add backward-compatible import shims temporarily

---

*Document Version: 1.0*
*Created: 2025-01-09*

