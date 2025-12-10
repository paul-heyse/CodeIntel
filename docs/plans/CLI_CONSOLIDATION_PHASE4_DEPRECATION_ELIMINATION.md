# Phase 4: Deprecation Elimination Plan

## Overview

This document details the steps to fully eliminate deprecated handler code and complete the migration to the unified `handlers/` architecture established in Phase 3.

**Goal**: Remove all legacy handler modules and migrate all consumers to the new unified handlers.

---

## Current State Assessment

### Legacy Handler Files (To Be Deleted)

| File | Lines | Current Usage | New Replacement |
|------|-------|---------------|-----------------|
| `build_handlers.py` | ~1118 | `cyclopts_build.py` | `handlers/build.py` |
| `datasets_handlers.py` | ~2128 | `cyclopts_datasets.py` | `handlers/datasets.py` |
| `docs_handlers.py` | ~1263 | `cyclopts_docs.py` | `handlers/docs.py` |
| `graphs_handlers.py` | ~588 | `cyclopts_graphs.py` | `handlers/graphs.py` |
| `ops_handlers.py` | ~657 | `cyclopts_ops.py`, operations | `handlers/ops.py` |
| `storage_handlers.py` | ~400 | `cyclopts_storage.py` | `handlers/storage.py` |
| `ide_handlers.py` | ~250 | `cyclopts_ide.py` | `handlers/ide.py` |
| `subsystem_handlers.py` | ~600 | `cyclopts_subsystem.py` | `handlers/subsystem.py` |

**Total**: ~7,000+ lines of legacy code to eliminate

### Files Still Importing Legacy Handlers

```
src/codeintel/cli/cyclopts_build.py       → build_handlers
src/codeintel/cli/cyclopts_datasets.py    → datasets_handlers
src/codeintel/cli/cyclopts_docs.py        → docs_handlers
src/codeintel/cli/cyclopts_graphs.py      → graphs_handlers
src/codeintel/cli/cyclopts_ops.py         → ops_handlers
src/codeintel/cli/operations/op_operations.py      → ops_handlers
src/codeintel/cli/operations/dataset_operations.py → ops_handlers
```

### sys.stdout.write Usage (85 instances in 7 files)

| File | Count | Resolution |
|------|-------|------------|
| `graphs_handlers.py` | 14 | Delete file |
| `docs_handlers.py` | 13 | Delete file |
| `cyclopts_plugins.py` | 29 | Migrate to renderer |
| `cyclopts_jobs.py` | 14 | Migrate to renderer |
| `cyclopts_health.py` | 2 | Migrate to renderer |
| `cli_completions.py` | 2 | Keep (completion output) |
| `cli_render.py` | 11 | Keep (renderer internals) |

### RuntimeCliOptions Usage (43 instances in 8 files)

| File | Count | Resolution |
|------|-------|------------|
| `datasets_handlers.py` | 4 | Delete file |
| `build_handlers.py` | 7 | Delete file |
| `subsystem_handlers.py` | 4 | Delete file |
| `ide_handlers.py` | 4 | Delete file |
| `common_handlers.py` | 7 | Consolidate to resolution/ |
| `cyclopts_common.py` | 14 | Migrate to RuntimeCLI |
| `resolution/params.py` | 2 | Keep (part of new arch) |
| `handlers/base.py` | 1 | Keep (part of new arch) |

---

## Migration Waves

### Wave 1: Update Cyclopts Files to Use New Handlers

**Priority**: High (breaks import chain to legacy handlers)

#### 1.1 cyclopts_build.py

**Current**:
```python
from codeintel.cli.build_handlers import (
    build_history_ctx,
    build_run_ctx,
    build_status_ctx,
)
```

**Target**:
```python
from codeintel.cli.command_context import command_context
from codeintel.cli.handlers.build import (
    build_history_handler,
    build_run_handler,
    build_status_handler,
)
```

**Changes**:
- Replace `CycloptsAdapter(build_run_ctx).run(...)` with:
  ```python
  with command_context("build.run", runtime_cli, output_cli, params={...}) as (ctx, renderer):
      result = build_run_handler(ctx)
      renderer.render_result(result)
  ```

#### 1.2 cyclopts_datasets.py

**Current**:
```python
from codeintel.cli.datasets_handlers import (
    datasets_catalog_ctx,
    datasets_conformance_ctx,
    # ... many more
)
```

**Target**:
```python
from codeintel.cli.command_context import command_context
from codeintel.cli.handlers.datasets import (
    datasets_list_handler,
    datasets_lint_handler,
    datasets_snapshot_handler,
    datasets_diff_handler,
)
```

**Notes**:
- Some handlers in `datasets_handlers.py` are complex (catalog, conformance, scaffold)
- May need to add more handlers to `handlers/datasets.py` or keep simplified versions
- Consider which operations are truly needed vs. rarely used

#### 1.3 cyclopts_docs.py

**Current**:
```python
from codeintel.cli.docs_handlers import (
    docs_export_ctx,
    DocsExportOptions,
    # ...
)
```

**Target**:
```python
from codeintel.cli.command_context import command_context
from codeintel.cli.handlers.docs import (
    docs_export_handler,
    docs_validate_handler,
)
```

#### 1.4 cyclopts_graphs.py

**Current**:
```python
from codeintel.cli.graphs_handlers import (
    graph_plugins_ctx,
    # ...
)
```

**Target**:
```python
from codeintel.cli.command_context import command_context
from codeintel.cli.handlers.graphs import (
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
)
```

#### 1.5 cyclopts_ops.py

**Current**:
```python
from codeintel.cli.ops_handlers import (
    dataset_list_ctx,
    op_list_ctx,
    serve_http_ctx,
    # ...
)
```

**Target**:
```python
from codeintel.cli.command_context import command_context
from codeintel.cli.handlers.ops import (
    dataset_list_handler,
    op_list_handler,
    serve_http_handler,
    # ...
)
```

---

### Wave 2: Update Operations Modules

**Priority**: Medium

#### 2.1 operations/op_operations.py

**Current**:
```python
from codeintel.cli.ops_handlers import op_list_structured
```

**Target**:
```python
from codeintel.cli.handlers.ops import op_list_handler
```

#### 2.2 operations/dataset_operations.py

**Current**:
```python
from codeintel.cli.ops_handlers import dataset_describe_structured
```

**Target**:
```python
from codeintel.cli.handlers.ops import dataset_describe_handler
```

---

### Wave 3: Migrate Remaining sys.stdout.write Files

**Priority**: Medium

#### 3.1 cyclopts_plugins.py (29 occurrences)

- Replace direct stdout writes with `CliResult` returns
- Use renderer for output formatting

#### 3.2 cyclopts_jobs.py (14 occurrences)

- Replace direct stdout writes with `CliResult` returns
- Use renderer for output formatting

#### 3.3 cyclopts_health.py (2 occurrences)

- Already partially migrated in Wave 2
- Complete migration to use `health_check_handler`

---

### Wave 4: Consolidate common_handlers.py

**Priority**: Medium

The `common_handlers.py` file contains shared utilities:
- `RuntimeCliOptions` (7 uses)
- Setup functions

**Actions**:
1. Move reusable utilities to `handlers/base.py`
2. Migrate `RuntimeCliOptions` consumers to use `RuntimeCLI` dataclass
3. Delete `common_handlers.py`

---

### Wave 5: Delete Legacy Handler Files

**Priority**: Final step (after all consumers migrated)

**Order of deletion** (least dependencies first):

1. `ide_handlers.py` - Already migrated, no complex dependencies
2. `subsystem_handlers.py` - Already migrated
3. `graphs_handlers.py` - Simpler handlers
4. `storage_handlers.py` - Already migrated
5. `ops_handlers.py` - Update operations/ first
6. `docs_handlers.py` - Medium complexity
7. `build_handlers.py` - Medium complexity
8. `datasets_handlers.py` - Most complex, delete last

**For each deletion**:
1. Verify no imports remain: `rg "from codeintel.cli.<module>" src/`
2. Verify tests pass: `pytest tests/cli/`
3. Delete file
4. Run full test suite

---

## Detailed Task Checklist

### Phase 4.1: Cyclopts Migration

- [ ] **cyclopts_build.py**
  - [ ] Update imports to use `handlers/build.py`
  - [ ] Replace `CycloptsAdapter` calls with `command_context`
  - [ ] Update parameter extraction
  - [ ] Test all build commands

- [ ] **cyclopts_datasets.py**
  - [ ] Identify which handlers need expansion in `handlers/datasets.py`
  - [ ] Add missing handlers (catalog, conformance, scaffold, validate-files, generate-schemas)
  - [ ] Update imports
  - [ ] Replace `CycloptsAdapter` calls
  - [ ] Test all datasets commands

- [ ] **cyclopts_docs.py**
  - [ ] Update imports to use `handlers/docs.py`
  - [ ] Replace `CycloptsAdapter` calls
  - [ ] Test all docs commands

- [ ] **cyclopts_graphs.py**
  - [ ] Update imports to use `handlers/graphs.py`
  - [ ] Replace `CycloptsAdapter` calls
  - [ ] Test all graphs commands

- [ ] **cyclopts_ops.py**
  - [ ] Update imports to use `handlers/ops.py`
  - [ ] Replace `CycloptsAdapter` calls
  - [ ] Test all ops commands

### Phase 4.2: Operations Module Migration

- [ ] **operations/op_operations.py**
  - [ ] Update to use `handlers/ops.py`
  - [ ] Test operation listing

- [ ] **operations/dataset_operations.py**
  - [ ] Update to use `handlers/ops.py`
  - [ ] Test dataset operations

### Phase 4.3: stdout.write Cleanup

- [ ] **cyclopts_plugins.py**
  - [ ] Create `handlers/plugins.py` with plugin handlers
  - [ ] Migrate stdout.write to CliResult
  - [ ] Test plugin commands

- [ ] **cyclopts_jobs.py**
  - [ ] Create `handlers/jobs.py` with job handlers
  - [ ] Migrate stdout.write to CliResult
  - [ ] Test job commands

- [ ] **cyclopts_health.py**
  - [ ] Complete migration to `handlers/health.py`
  - [ ] Remove remaining stdout.write
  - [ ] Test health commands

### Phase 4.4: common_handlers.py Consolidation

- [ ] Migrate utilities to `handlers/base.py`
- [ ] Update consumers of `RuntimeCliOptions`
- [ ] Delete `common_handlers.py`

### Phase 4.5: Legacy File Deletion

- [ ] Delete `ide_handlers.py`
- [ ] Delete `subsystem_handlers.py`
- [ ] Delete `graphs_handlers.py`
- [ ] Delete `storage_handlers.py`
- [ ] Delete `ops_handlers.py`
- [ ] Delete `docs_handlers.py`
- [ ] Delete `build_handlers.py`
- [ ] Delete `datasets_handlers.py`
- [ ] Delete `common_handlers.py`

---

## Acceptance Criteria

### Per-File Migration Criteria

1. **Zero imports** from legacy handler module
2. **Zero deprecation warnings** when running commands
3. **All tests pass** for affected command group
4. **Quality checks pass** (pyright, pyrefly, ruff)

### Final State Criteria

1. **Zero legacy handler files** in `src/codeintel/cli/`
2. **Zero sys.stdout.write** in handler code (allowed in renderer)
3. **Zero RuntimeCliOptions** definitions (use RuntimeCLI)
4. **All handlers** follow `EnhancedHandlerContext → CliResult[T]` pattern
5. **All commands** use `command_context()` for setup/teardown
6. **Full test coverage** for all handlers
7. **Documentation updated** to reflect new architecture

---

## Risk Assessment

### High Risk Items

1. **datasets_handlers.py complexity**: 2100+ lines with many specialized handlers
   - **Mitigation**: Phase migration, keep simplified versions, add handlers incrementally

2. **Breaking CLI behavior**: Users may depend on specific output formats
   - **Mitigation**: Comprehensive end-to-end tests, gradual rollout

### Medium Risk Items

1. **Operations module integration**: Used by MCP and serving
   - **Mitigation**: Test MCP endpoints after migration

2. **Build system commands**: Core workflow
   - **Mitigation**: Prioritize testing, get user validation

---

## Timeline Estimate

| Wave | Scope | Estimated Effort |
|------|-------|------------------|
| 4.1 | Cyclopts migration | 2-3 days |
| 4.2 | Operations migration | 0.5 day |
| 4.3 | stdout.write cleanup | 1-2 days |
| 4.4 | common_handlers consolidation | 0.5 day |
| 4.5 | Legacy file deletion | 0.5 day |

**Total**: ~5-7 days of focused work

---

## Commands for Verification

```bash
# Check for remaining legacy imports
rg "from codeintel\.cli\.(datasets_handlers|docs_handlers|graphs_handlers|build_handlers|storage_handlers|ops_handlers|ide_handlers|subsystem_handlers|common_handlers)" src/

# Check for sys.stdout.write in handler code
rg "sys\.stdout\.write" src/codeintel/cli/handlers/

# Check for RuntimeCliOptions
rg "RuntimeCliOptions" src/codeintel/cli/

# Run all CLI tests
uv run pytest tests/cli/ -v

# Run quality checks
uv run ruff check src/codeintel/cli/
uv run pyright src/codeintel/cli/
```

---

## Next Steps

1. **Start with Wave 4.1** - Cyclopts migration is the critical path
2. **Begin with simpler files** (graphs, health) before complex ones (datasets, build)
3. **Iterate**: Complete one cyclopts file fully before moving to next
4. **Test continuously**: Run tests after each file migration
