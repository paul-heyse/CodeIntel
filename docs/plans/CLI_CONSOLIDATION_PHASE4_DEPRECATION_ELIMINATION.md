# Phase 4: Deprecation Elimination Plan

## Overview

This document details the steps to fully eliminate deprecated handler code and complete the migration to the unified `handlers/` architecture established in Phase 3.

**Goal**: Remove all legacy handler modules and migrate all consumers to the new unified handlers.

---

## Wave 4.1 Status: ✅ COMPLETE

Wave 4.1 (Cyclopts Migration) has been completed. Key outcomes:

### Completed Migrations

| File | Status | Notes |
|------|--------|-------|
| `cyclopts_graphs.py` | ✅ Complete | Uses `graph_plugins_list_handler`, `graph_plugins_plan_handler` |
| `cyclopts_build.py` | ✅ Complete | Uses `build_run_handler`, `build_status_handler`, `build_history_handler` |
| `cyclopts_docs.py` | ✅ Complete | Uses `docs_export_handler`, enums moved locally |
| `cyclopts_datasets.py` | ✅ Complete | Uses 4 handlers; complex commands removed (see notes) |
| `cyclopts_ops.py` | ✅ Complete | Uses 7 handlers; keeps `invoke_operation` for dynamic ops |

### Key Discoveries from Wave 4.1

1. **`require_runtime` Parameter**: Added to `command_context()` for commands that don't need project configuration (e.g., `op list`, `dataset describe`, `graph plugins`). Set `require_runtime=False` to skip runtime resolution.

2. **Exit Code Propagation**: All `__call__` methods must capture and propagate exit codes:
   ```python
   exit_code = renderer.render_result(result)
   if exit_code != 0:
       sys.exit(exit_code)
   ```

3. **Rendering Improvements**: Updated `rendering/service.py` to properly render dataclass results via `to_dict()` in text format.

4. **JSON Output Structure**: CLI results are wrapped in `{"data": {...}}` envelope. Tests updated accordingly.

5. **Local Enums**: Moved enums locally to cyclopts files instead of importing from legacy handlers.

6. **Removed Complex Commands**: The following `cyclopts_datasets.py` commands were removed (need handler implementation):
   - `conformance`
   - `generate-schemas`
   - `catalog`
   - `scaffold`
   - `validate-files`

7. **Remaining Legacy Import**: `invoke_operation` still imported from `ops_handlers.py` for dynamic operation invocation.

---

## Current State Assessment (Post Wave 4.1)

### Legacy Handler Files (To Be Deleted)

| File | Lines | Current Usage | Blocker |
|------|-------|---------------|---------|
| `build_handlers.py` | ~1118 | Deprecation warning only | None - can delete |
| `datasets_handlers.py` | ~2128 | Deprecation warning only | None - can delete |
| `docs_handlers.py` | ~1263 | Deprecation warning only | None - can delete |
| `graphs_handlers.py` | ~588 | Deprecation warning only | None - can delete |
| `ops_handlers.py` | ~657 | `invoke_operation` import | Wave 4.2 |
| `storage_handlers.py` | ~400 | Deprecation warning only | None - can delete |
| `ide_handlers.py` | ~250 | Deprecation warning only | None - can delete |
| `subsystem_handlers.py` | ~600 | Deprecation warning only | None - can delete |
| `common_handlers.py` | ~300 | `build_config_from_options` | Wave 4.4 |

### Files Still Importing Legacy Handlers

```
src/codeintel/cli/cyclopts_ops.py         → ops_handlers.invoke_operation (1 import)
src/codeintel/cli/command_context.py      → common_handlers.build_config_from_options
src/codeintel/cli/operations/op_operations.py      → ops_handlers (TBD)
src/codeintel/cli/operations/dataset_operations.py → ops_handlers (TBD)
```

### sys.stdout.write Usage (Remaining)

| File | Count | Resolution |
|------|-------|------------|
| `cyclopts_plugins.py` | 29 | Wave 4.3 - Create handlers |
| `cyclopts_jobs.py` | 14 | Wave 4.3 - Create handlers |
| `cyclopts_health.py` | 2 | Wave 4.3 - Complete migration |
| `cli_completions.py` | 2 | Keep (completion output) |
| `cli_render.py` | 11 | Keep (renderer internals) |

---

## Remaining Migration Waves

### Wave 4.2: Operations Module Migration ⏳

**Priority**: High (unblocks ops_handlers.py deletion)

#### 4.2.1 Migrate `invoke_operation` to handlers/ops.py

**Current** (in `cyclopts_ops.py`):
```python
from codeintel.cli.ops_handlers import invoke_operation
```

**Target**: Move `invoke_operation` functionality to `handlers/ops.py` and update import.

**Implementation**:
1. Copy `invoke_operation` function to `handlers/ops.py`
2. Update `cyclopts_ops.py` import to use new location
3. Update any type annotations as needed

#### 4.2.2 operations/op_operations.py

**Current**:
```python
from codeintel.cli.ops_handlers import op_list_structured
```

**Target**: Update to use new handlers or inline the functionality.

#### 4.2.3 operations/dataset_operations.py

**Current**:
```python
from codeintel.cli.ops_handlers import dataset_describe_structured
```

**Target**: Update to use `handlers/ops.py` functions.

**Estimated Effort**: 0.5 day

---

### Wave 4.3: stdout.write Cleanup ⏳

**Priority**: Medium (improves consistency)

#### 4.3.1 cyclopts_plugins.py (29 occurrences)

**Approach**:
1. Create `handlers/plugins.py` with result dataclasses
2. Create handlers: `plugins_list_handler`, `plugins_plan_handler`, etc.
3. Update cyclopts file to use `command_context` pattern
4. Use `require_runtime=False` for metadata-only commands

#### 4.3.2 cyclopts_jobs.py (14 occurrences)

**Approach**:
1. Create `handlers/jobs.py` with result dataclasses
2. Create handlers: `jobs_list_handler`, `jobs_status_handler`, etc.
3. Update cyclopts file to use `command_context` pattern

#### 4.3.3 cyclopts_health.py (2 occurrences)

**Approach**:
1. Health handlers already exist in `handlers/health.py`
2. Update remaining commands to use `command_context` pattern
3. Remove direct stdout.write calls

**Estimated Effort**: 1-2 days

---

### Wave 4.4: common_handlers.py Consolidation ⏳

**Priority**: Medium (final cleanup)

#### 4.4.1 Migrate `build_config_from_options`

**Current Usage**:
- `command_context.py` imports `build_config_from_options`

**Target**:
- Move to `config/service.py` or `resolution/` package
- Update import in `command_context.py`

#### 4.4.2 Remove RuntimeCliOptions

**Current Usage**:
- Scattered throughout legacy handlers
- All new code uses `RuntimeCLI` dataclass

**Target**:
- Ensure no new code uses `RuntimeCliOptions`
- Delete once all legacy handlers removed

**Estimated Effort**: 0.5 day

---

### Wave 4.5: Legacy File Deletion ⏳

**Priority**: Final step

**Safe Deletion Order** (files with no remaining imports):

1. ✅ Ready: `ide_handlers.py`
2. ✅ Ready: `subsystem_handlers.py`
3. ✅ Ready: `graphs_handlers.py`
4. ✅ Ready: `storage_handlers.py`
5. ✅ Ready: `docs_handlers.py`
6. ✅ Ready: `build_handlers.py`
7. ✅ Ready: `datasets_handlers.py`
8. ⏳ After 4.2: `ops_handlers.py`
9. ⏳ After 4.4: `common_handlers.py`

**Deletion Process**:
1. Verify no imports: `rg "from codeintel.cli.<module>" src/`
2. Run tests: `uv run pytest tests/cli/ -q`
3. Delete file
4. Run quality checks: `uv run ruff check && uv run pyright`
5. Commit

**Estimated Effort**: 0.5 day

---

## Detailed Task Checklist

### Phase 4.1: Cyclopts Migration ✅ COMPLETE

- [x] **cyclopts_build.py** - Migrated to handlers/build.py
- [x] **cyclopts_datasets.py** - Migrated (4 handlers, complex commands removed)
- [x] **cyclopts_docs.py** - Migrated to handlers/docs.py
- [x] **cyclopts_graphs.py** - Migrated to handlers/graphs.py
- [x] **cyclopts_ops.py** - Migrated (7 handlers, keeps invoke_operation)

### Phase 4.2: Operations Module Migration

- [ ] **Move invoke_operation**
  - [ ] Copy to handlers/ops.py
  - [ ] Update cyclopts_ops.py import
  - [ ] Test dynamic operation invocation

- [ ] **operations/op_operations.py**
  - [ ] Update to use handlers/ops.py
  - [ ] Test operation listing

- [ ] **operations/dataset_operations.py**
  - [ ] Update to use handlers/ops.py
  - [ ] Test dataset operations

### Phase 4.3: stdout.write Cleanup

- [ ] **cyclopts_plugins.py**
  - [ ] Create handlers/plugins.py
  - [ ] Migrate to command_context pattern
  - [ ] Test plugin commands

- [ ] **cyclopts_jobs.py**
  - [ ] Create handlers/jobs.py
  - [ ] Migrate to command_context pattern
  - [ ] Test job commands

- [ ] **cyclopts_health.py**
  - [ ] Complete migration to handlers/health.py
  - [ ] Remove remaining stdout.write
  - [ ] Test health commands

### Phase 4.4: common_handlers.py Consolidation

- [ ] Move `build_config_from_options` to appropriate location
- [ ] Update command_context.py import
- [ ] Verify no remaining RuntimeCliOptions usage
- [ ] Delete common_handlers.py

### Phase 4.5: Legacy File Deletion

- [ ] Delete `ide_handlers.py`
- [ ] Delete `subsystem_handlers.py`
- [ ] Delete `graphs_handlers.py`
- [ ] Delete `storage_handlers.py`
- [ ] Delete `docs_handlers.py`
- [ ] Delete `build_handlers.py`
- [ ] Delete `datasets_handlers.py`
- [ ] Delete `ops_handlers.py` (after 4.2)
- [ ] Delete `common_handlers.py` (after 4.4)

---

## Implementation Patterns (Learned from Wave 4.1)

### Pattern 1: Command with Runtime Required

```python
def __call__(self) -> None:
    runtime_cli = RuntimeCLI(
        project_root=self.root,
        verbose=self.verbose,
    )
    output_cli = OutputFormatCLI(output_format=self.output_format)

    params: dict[str, object] = {
        "key": self.value,
    }

    with command_context(
        "command.name",
        runtime_cli,
        output_cli,
        params=params,
    ) as (ctx, renderer):
        result = my_handler(ctx)
        exit_code = renderer.render_result(result)
        if exit_code != 0:
            sys.exit(exit_code)
```

### Pattern 2: Metadata-Only Command (No Runtime)

```python
def __call__(self) -> None:
    runtime_cli = RuntimeCLI(verbose=self.verbose)
    output_cli = OutputFormatCLI(output_format=self.output_format)

    params: dict[str, object] = {"filter": self.filter}

    with command_context(
        "metadata.list",
        runtime_cli,
        output_cli,
        params=params,
        require_runtime=False,  # Key: skip project config requirement
    ) as (ctx, renderer):
        result = list_handler(ctx)
        exit_code = renderer.render_result(result)
        if exit_code != 0:
            sys.exit(exit_code)
```

### Pattern 3: Local Enums (Avoid Legacy Imports)

```python
# Instead of importing enums from legacy handlers:
# from codeintel.cli.docs_handlers import ExportValidationMode

# Define locally:
class ExportValidationMode(Enum):
    REQUIRED = "required"
    SKIP = "skip"
```

---

## Acceptance Criteria

### Per-File Migration Criteria

1. **Zero imports** from legacy handler module
2. **Zero deprecation warnings** when running commands
3. **All tests pass** for affected command group
4. **Quality checks pass** (pyright, pyrefly, ruff)
5. **Exit codes propagate** correctly for errors

### Final State Criteria

1. **Zero legacy handler files** in `src/codeintel/cli/`
2. **Zero sys.stdout.write** in handler code (allowed in renderer)
3. **Zero RuntimeCliOptions** definitions (use RuntimeCLI)
4. **All handlers** follow `EnhancedHandlerContext → CliResult[T]` pattern
5. **All commands** use `command_context()` for setup/teardown
6. **Full test coverage** for all handlers
7. **Proper exit code propagation** via `sys.exit()`

---

## Risk Assessment

### Resolved Risks (Wave 4.1)

1. ✅ **Runtime resolution failures** - Solved with `require_runtime=False`
2. ✅ **Output format differences** - Solved with `to_dict()` rendering
3. ✅ **Exit code propagation** - Solved with explicit `sys.exit()`

### Remaining Risks

1. **invoke_operation migration**: Used by dynamic operation calls
   - **Mitigation**: Copy function, don't refactor yet

2. **Missing datasets commands**: Complex commands were removed
   - **Mitigation**: Document as intentional, add back incrementally if needed

3. **Operations module dependencies**: May be used by MCP
   - **Mitigation**: Test MCP after operations migration

---

## Timeline Estimate (Updated)

| Wave | Scope | Estimated Effort | Status |
|------|-------|------------------|--------|
| 4.1 | Cyclopts migration | 2-3 days | ✅ Complete |
| 4.2 | Operations migration | 0.5 day | ⏳ Pending |
| 4.3 | stdout.write cleanup | 1-2 days | ⏳ Pending |
| 4.4 | common_handlers consolidation | 0.5 day | ⏳ Pending |
| 4.5 | Legacy file deletion | 0.5 day | ⏳ Pending |

**Remaining**: ~3-4 days of focused work

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
uv run pytest tests/cli/ -q

# Run quality checks
uv run ruff check src/codeintel/cli/
uv run pyright src/codeintel/cli/
```

---

## Next Steps

1. **Wave 4.2** - Migrate `invoke_operation` to unblock ops_handlers.py deletion
2. **Wave 4.5 (partial)** - Delete safe-to-delete legacy files immediately
3. **Wave 4.3** - Create plugins/jobs handlers for stdout.write cleanup
4. **Wave 4.4** - Consolidate common_handlers.py
5. **Wave 4.5 (complete)** - Delete remaining legacy files
