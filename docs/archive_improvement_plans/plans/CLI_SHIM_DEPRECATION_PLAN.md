# CLI Compatibility Shim Deprecation Plan

> **Status**: ✅ COMPLETED  
> **Created**: 2025-12-10  
> **Completed**: 2025-12-10  

## Executive Summary

This plan details the systematic removal of 19 compatibility shims created during the CLI reorganization. Shims are ordered by complexity (fewest dependents first) to:

1. Clear easy wins early, building momentum
2. Establish patterns for migration before tackling complex cases
3. Minimize risk of breaking changes in high-traffic modules

**Total scope**: ~150 import statements across ~90 files (src + tests)

---

## Table of Contents

1. [Shim Inventory & Dependency Analysis](#shim-inventory--dependency-analysis)
2. [Phase 1: Zero-Dependency Shims](#phase-1-zero-dependency-shims)
3. [Phase 2: Low-Complexity Shims](#phase-2-low-complexity-shims)
4. [Phase 3: Medium-Complexity Shims](#phase-3-medium-complexity-shims)
5. [Phase 4: High-Complexity Shims](#phase-4-high-complexity-shims)
6. [Phase 5: Final Cleanup & Verification](#phase-5-final-cleanup--verification)
7. [Appendix: File-by-File Migration Reference](#appendix-file-by-file-migration-reference)

---

## Shim Inventory & Dependency Analysis

### Complexity Classification

| Shim Module | External Dependents | Complexity | Phase |
|-------------|---------------------|------------|-------|
| `job_runner.py` | 0 | 🟢 Zero | 1 |
| `pipelines.py` | 0 | 🟢 Zero | 1 |
| `shell.py` | 1 (test only) | 🟢 Zero | 1 |
| `output.py` | 1 | 🟡 Low | 2 |
| `dry_run.py` | 1 | 🟡 Low | 2 |
| `help_system.py` | 1 | 🟡 Low | 2 |
| `op_params.py` | 1 | 🟡 Low | 2 |
| `cli_render.py` | 1 | 🟡 Low | 2 |
| `observability.py` | 1 | 🟡 Low | 2 |
| `telemetry.py` | 1 | 🟡 Low | 2 |
| `resilience.py` | 2 | 🟠 Medium | 3 |
| `jobs.py` | 2 | 🟠 Medium | 3 |
| `cli_validation.py` | 3 | 🟠 Medium | 3 |
| `introspection.py` | 8 | 🔴 High | 4 |
| `result_types.py` | 9 | 🔴 High | 4 |
| `project.py` | 12 | 🔴 High | 4 |
| `operation_registry.py` | 13 | 🔴 High | 4 |
| `cli_errors.py` | 15 | 🔴 High | 4 |
| `cli_types.py` | 17 | 🔴 High | 4 |
| `results.py` | 28 | 🔴 High | 4 |

### Canonical Location Reference

| Old Import Path | New Canonical Import |
|-----------------|---------------------|
| `codeintel.cli.job_runner` | `codeintel.cli.jobs` |
| `codeintel.cli.pipelines` | `codeintel.cli.project` |
| `codeintel.cli.shell` (module) | `codeintel.cli.shell` (package) |
| `codeintel.cli.output` | `codeintel.cli.core` |
| `codeintel.cli.dry_run` | `codeintel.cli.project` |
| `codeintel.cli.help_system` | `codeintel.cli.introspection` |
| `codeintel.cli.op_params` | `codeintel.cli.introspection` |
| `codeintel.cli.cli_render` | `codeintel.cli.rendering` |
| `codeintel.cli.observability` (module) | `codeintel.cli.observability` (package) |
| `codeintel.cli.telemetry` | `codeintel.cli.observability` |
| `codeintel.cli.resilience` (module) | `codeintel.cli.resilience` (package) |
| `codeintel.cli.jobs` (module) | `codeintel.cli.jobs` (package) |
| `codeintel.cli.cli_validation` | `codeintel.cli.introspection` |
| `codeintel.cli.introspection` (module) | `codeintel.cli.introspection` (package) |
| `codeintel.cli.result_types` | `codeintel.cli.core.result_types` |
| `codeintel.cli.project` (module) | `codeintel.cli.project` (package) |
| `codeintel.cli.operation_registry` | `codeintel.cli.introspection` |
| `codeintel.cli.cli_errors` | `codeintel.cli.errors` |
| `codeintel.cli.cli_types` | `codeintel.cli.rendering.types` + `codeintel.cli.resolution.params` |
| `codeintel.cli.results` | `codeintel.cli.core` |

---

## Phase 1: Zero-Dependency Shims

**Goal**: Remove shims with no external dependents (only referenced by themselves or tests)  
**Effort**: ~1 hour  
**Risk**: 🟢 Minimal  

### 1.1 Remove `job_runner.py`

**Current state**: Only referenced by itself  
**Migration**: None required  

```bash
# Verification
rg "from codeintel\.cli\.job_runner import" src/ tests/

# Action
rm src/codeintel/cli/job_runner.py

# Post-verification
uv run python -c "from codeintel.cli import app"
uv run pytest tests/cli/handlers/test_jobs.py -v
```

### 1.2 Remove `pipelines.py`

**Current state**: Only referenced by itself  
**Migration**: None required  

```bash
# Verification
rg "from codeintel\.cli\.pipelines import" src/ tests/

# Action
rm src/codeintel/cli/pipelines.py

# Post-verification
uv run python -c "from codeintel.cli import app"
uv run pytest tests/cli/ -k pipeline -v
```

### 1.3 Remove `shell.py` (module shim)

**Current state**: Only 1 test file uses it  
**Migration**: Update test import  

**Files to update**:
- `tests/cli/shell/test_shell_mode.py`

```python
# Before
from codeintel.cli.shell import (
    InteractiveShell,
    ShellSession,
    start_shell,
)

# After (same, but imports from package __init__.py, not module shim)
from codeintel.cli.shell import (
    InteractiveShell,
    ShellSession,
    start_shell,
)
```

> **Note**: The import path looks identical, but removing the shim file means
> Python will find `cli/shell/__init__.py` (the package) instead of `cli/shell.py` (the shim).

```bash
# Action
rm src/codeintel/cli/shell.py

# Post-verification
uv run pytest tests/cli/shell/ -v
```

### Phase 1 Completion Checklist

- [ ] Delete `job_runner.py`
- [ ] Delete `pipelines.py`
- [ ] Delete `shell.py`
- [ ] Run full CLI test suite
- [ ] Commit: "chore(cli): remove zero-dependency compatibility shims"

---

## Phase 2: Low-Complexity Shims

**Goal**: Remove shims with exactly 1 external dependent  
**Effort**: ~2-3 hours  
**Risk**: 🟡 Low  

### 2.1 Remove `output.py`

**Dependents (1)**:
- `src/codeintel/cli/cyclopts_ops.py`

**Migration**:

```python
# cyclopts_ops.py - Before
from codeintel.cli.output import iter_stdin_records

# cyclopts_ops.py - After
from codeintel.cli.core import iter_stdin_records
```

```bash
rm src/codeintel/cli/output.py
```

### 2.2 Remove `dry_run.py`

**Dependents (1)**:
- `src/codeintel/cli/cyclopts_ops.py`

**Migration**:

```python
# cyclopts_ops.py - Before
from codeintel.cli.dry_run import plan_dry_run, render_dry_run

# cyclopts_ops.py - After
from codeintel.cli.project import plan_dry_run, render_dry_run
```

```bash
rm src/codeintel/cli/dry_run.py
```

### 2.3 Remove `help_system.py`

**Dependents (1)**:
- `src/codeintel/cli/cyclopts_help_commands.py`

**Migration**:

```python
# cyclopts_help_commands.py - Before
from codeintel.cli.help_system import HelpRenderer, get_help_renderer

# cyclopts_help_commands.py - After
from codeintel.cli.introspection import HelpRenderer, get_help_renderer
```

```bash
rm src/codeintel/cli/help_system.py
```

### 2.4 Remove `op_params.py`

**Dependents (1)**:
- `src/codeintel/cli/cyclopts_ops.py`

**Migration**:

```python
# cyclopts_ops.py - Before
from codeintel.cli.op_params import build_cli_param_specs_for_operation

# cyclopts_ops.py - After
from codeintel.cli.introspection import build_cli_param_specs_for_operation
```

```bash
rm src/codeintel/cli/op_params.py
```

### 2.5 Remove `cli_render.py`

**Dependents (1)**:
- `src/codeintel/cli/execution/adapter.py`

**Migration**:

```python
# execution/adapter.py - Before
from codeintel.cli.cli_render import get_renderer, render_cli_result

# execution/adapter.py - After
from codeintel.cli.rendering import get_renderer, render_cli_result
```

```bash
rm src/codeintel/cli/cli_render.py
```

### 2.6 Remove `observability.py` (module shim)

**Dependents (1)**:
- `src/codeintel/cli/telemetry.py` (another shim - creates dependency chain)

> **Note**: This creates a dependency chain. We'll update the telemetry shim first,
> then remove observability shim in Phase 3 when telemetry is also removed.

**Action**: Defer to Phase 3 (dependency chain with telemetry.py)

### 2.7 Remove `telemetry.py`

**Dependents (1)**:
- `src/codeintel/cli/health.py`

**Migration**:

```python
# health.py - Before
from codeintel.cli.telemetry import TelemetryConfig

# health.py - After
from codeintel.cli.observability import TelemetryConfig
```

```bash
rm src/codeintel/cli/telemetry.py
```

### Phase 2 Completion Checklist

- [ ] Update `cyclopts_ops.py` (output, dry_run, op_params)
- [ ] Update `cyclopts_help_commands.py` (help_system)
- [ ] Update `execution/adapter.py` (cli_render)
- [ ] Update `health.py` (telemetry)
- [ ] Delete `output.py`
- [ ] Delete `dry_run.py`
- [ ] Delete `help_system.py`
- [ ] Delete `op_params.py`
- [ ] Delete `cli_render.py`
- [ ] Delete `telemetry.py`
- [ ] Run full CLI test suite
- [ ] Commit: "chore(cli): remove low-complexity compatibility shims"

---

## Phase 3: Medium-Complexity Shims

**Goal**: Remove shims with 2-5 external dependents  
**Effort**: ~3-4 hours  
**Risk**: 🟠 Moderate  

### 3.1 Remove `observability.py` (module shim)

**Dependents (0 after Phase 2)**:
- Previously: `telemetry.py` (removed in Phase 2)

```bash
rm src/codeintel/cli/observability.py
```

> **Verification**: After removing the shim file, imports like
> `from codeintel.cli.observability import X` will resolve to the
> `observability/__init__.py` package, which is the intended behavior.

### 3.2 Remove `resilience.py` (module shim)

**Dependents (2)**:
- `src/codeintel/cli/execution/executor.py`
- `src/codeintel/cli/config/model.py`

**Migration**:

```python
# execution/executor.py - Before
from codeintel.cli.resilience import RetryPolicy, execute_with_retry

# execution/executor.py - After
from codeintel.cli.resilience import RetryPolicy, execute_with_retry
# (same path, but now resolves to package __init__.py)
```

> **Note**: No code changes needed if the package `__init__.py` exports the same symbols.
> Just delete the shim file and verify.

```bash
rm src/codeintel/cli/resilience.py
uv run pytest tests/cli/unit/test_executor.py -v
```

### 3.3 Remove `jobs.py` (module shim)

**Dependents (2)**:
- `src/codeintel/cli/handlers/jobs.py`
- `src/codeintel/cli/jobs/runner.py`

**Migration**:

```python
# handlers/jobs.py - Before
from codeintel.cli.jobs import JobManager, get_job_manager

# handlers/jobs.py - After (same, resolves to package)
from codeintel.cli.jobs import JobManager, get_job_manager
```

```bash
rm src/codeintel/cli/jobs.py
uv run pytest tests/cli/handlers/test_jobs.py -v
```

### 3.4 Remove `cli_validation.py`

**Dependents (3)**:
- `src/codeintel/cli/operations/op_operations.py`
- `src/codeintel/cli/operations/dataset_operations.py`
- `src/codeintel/cli/completions/completion_model.py`

**Migration**:

```python
# operations/op_operations.py - Before
from codeintel.cli.cli_validation import ValidationSchema

# operations/op_operations.py - After
from codeintel.cli.introspection import ValidationSchema
```

```python
# operations/dataset_operations.py - Before
from codeintel.cli.cli_validation import IntValidator

# operations/dataset_operations.py - After
from codeintel.cli.introspection import IntValidator
```

```python
# completions/completion_model.py - Before
from codeintel.cli.cli_validation import Validator

# completions/completion_model.py - After
from codeintel.cli.introspection import Validator
```

```bash
rm src/codeintel/cli/cli_validation.py
uv run pytest tests/cli/property/test_validators_property.py -v
```

### Phase 3 Completion Checklist

- [ ] Delete `observability.py` (module shim)
- [ ] Delete `resilience.py` (module shim)
- [ ] Delete `jobs.py` (module shim)
- [ ] Update `operations/op_operations.py`
- [ ] Update `operations/dataset_operations.py`
- [ ] Update `completions/completion_model.py`
- [ ] Delete `cli_validation.py`
- [ ] Run full CLI test suite
- [ ] Commit: "chore(cli): remove medium-complexity compatibility shims"

---

## Phase 4: High-Complexity Shims

**Goal**: Remove shims with 5+ external dependents  
**Effort**: ~8-12 hours (spread across multiple sessions)  
**Risk**: 🔴 Higher - requires careful coordination  

### Strategy

For high-complexity shims, we use a **staged approach**:

1. **Audit**: List all files with imports
2. **Batch Update**: Update imports in logical groups (by folder)
3. **Incremental Test**: Test after each batch
4. **Delete**: Remove shim only after all imports are updated
5. **Final Test**: Full test suite

### 4.1 Remove `introspection.py` (module shim)

**Dependents (8)**:

| File | Symbols Imported |
|------|------------------|
| `src/codeintel/cli/commands/help_commands.py` | `get_operation_info` |
| `src/codeintel/cli/commands/ops.py` | `get_operation_info, search_operations` |
| `src/codeintel/cli/jobs/runner.py` | `get_operation_info` |
| `src/codeintel/cli/shell/_shell.py` | `list_all_operations, search_operations` |
| `src/codeintel/cli/project/pipelines.py` | `get_operation_info` |
| `tests/cli/test_op_params.py` | Various |
| `tests/cli/test_op_dynamic_cli.py` | Various |
| `tests/cli/unit/test_operation_handlers.py` | `get_operation_registry` |
| `tests/cli/_harness/__init__.py` | `get_operation_registry` |
| `tests/cli/performance/test_performance.py` | `get_operation_registry` |
| `tests/cli/property/test_validators_property.py` | `IntValidator, StringValidator` |
| `tests/cli/unit/test_executor.py` | `StringValidator, ValidationSchema` |

**Migration Strategy**:

All imports can stay the same path since the package `__init__.py` exports the same symbols.
Delete the shim file and verify.

```bash
rm src/codeintel/cli/introspection.py
uv run pytest tests/cli/ -v -k "op_params or op_dynamic or operation_handlers or executor or validators"
```

### 4.2 Remove `result_types.py`

**Dependents (9)**:

| File | Symbols Imported |
|------|------------------|
| `src/codeintel/cli/operations/subsystem_operations.py` | Various result types |
| `src/codeintel/cli/operations/op_operations.py` | Various result types |
| `src/codeintel/cli/operations/ide_operations.py` | Various result types |
| `src/codeintel/cli/operations/history_operations.py` | Various result types |
| `src/codeintel/cli/operations/graph_operations.py` | Various result types |
| `src/codeintel/cli/operations/docs_operations.py` | Various result types |
| `src/codeintel/cli/operations/build_operations.py` | Various result types |
| `src/codeintel/cli/operations/dataset_operations.py` | Various result types |
| `src/codeintel/cli/handlers/ops.py` | Various result types |

**Migration**:

```python
# Before (all operations/*.py files)
from codeintel.cli.result_types import DatasetResult, GraphResult, ...

# After
from codeintel.cli.core.result_types import DatasetResult, GraphResult, ...
```

**Batch 1**: Update all `operations/*.py` files (8 files)  
**Batch 2**: Update `handlers/ops.py` (1 file)  

```bash
rm src/codeintel/cli/result_types.py
uv run pytest tests/cli/operations/ -v
```

### 4.3 Remove `project.py` (module shim)

**Dependents (12)**:

| File | Symbols Imported |
|------|------------------|
| `src/codeintel/cli/commands/ops.py` | `ProjectConfig, find_project_root` |
| `src/codeintel/cli/commands/context.py` | `ProjectRuntime` |
| `src/codeintel/cli/handlers/datasets.py` | `ProjectConfig` |
| `src/codeintel/cli/handlers/docs.py` | `ProjectConfig` |
| `src/codeintel/cli/handlers/build.py` | `ProjectConfig` |
| `src/codeintel/cli/handlers/ops.py` | `ProjectConfig` |
| `src/codeintel/cli/cyclopts_ops.py` | `find_project_root` |
| `src/codeintel/cli/command_context.py` | `ProjectRuntime` |
| `src/codeintel/cli/resolution/runtime.py` | `ProjectConfig` |
| `src/codeintel/cli/resolution/types.py` | `ProjectConfig` |
| `tests/cli/test_typer_cli.py` | Various |
| `tests/cli/test_project_module.py` | Various |

**Migration Strategy**:

All imports can stay the same path since the package `__init__.py` exports the same symbols.
Delete the shim file and verify.

```bash
rm src/codeintel/cli/project.py
uv run pytest tests/cli/test_project_module.py tests/cli/test_typer_cli.py -v
```

### 4.4 Remove `operation_registry.py`

**Dependents (13)**:

| File | Symbols Imported |
|------|------------------|
| `src/codeintel/cli/execution/adapter.py` | `get_operation_registry` |
| `src/codeintel/cli/plugins/registry.py` | `OperationRegistry` |
| `src/codeintel/cli/operations/subsystem_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/storage_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/op_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/ide_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/history_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/graph_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/docs_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/build_operations.py` | `register_operation` |
| `src/codeintel/cli/operations/dataset_operations.py` | `register_operation` |
| `src/codeintel/cli/health.py` | `get_operation_registry` |
| `src/codeintel/cli/completions/completion_model.py` | `get_operation_registry` |

**Migration**:

```python
# Before (all files)
from codeintel.cli.operation_registry import register_operation, get_operation_registry

# After
from codeintel.cli.introspection import register_operation, get_operation_registry
```

**Batch 1**: Update all `operations/*.py` files (9 files)  
**Batch 2**: Update remaining files (4 files)  

```bash
rm src/codeintel/cli/operation_registry.py
uv run pytest tests/cli/ -v
```

### 4.5 Remove `cli_errors.py`

**Dependents (15)**:

| File | Symbols Imported |
|------|------------------|
| `src/codeintel/cli/handlers/plugins.py` | `CliError, ValidationError` |
| `src/codeintel/cli/handlers/history.py` | `CliError` |
| `src/codeintel/cli/rendering/service.py` | `ProblemDetail` |
| `src/codeintel/cli/handlers/datasets.py` | `CliError, ValidationError` |
| `src/codeintel/cli/handlers/subsystem.py` | `CliError` |
| `src/codeintel/cli/handlers/graphs.py` | `CliError` |
| `src/codeintel/cli/handlers/storage.py` | `CliError` |
| `src/codeintel/cli/handlers/ops.py` | `CliError, ProblemDetail` |
| `src/codeintel/cli/handlers/ide.py` | `CliError` |
| `src/codeintel/cli/handlers/docs.py` | `CliError, ValidationError` |
| `src/codeintel/cli/handlers/build.py` | `CliError, ValidationError` |
| `src/codeintel/cli/handlers/jobs.py` | `CliError` |
| `src/codeintel/cli/error_taxonomy.py` | `CliError, ProblemDetail` |
| `src/codeintel/cli/cyclopts_ops.py` | `CliError` |
| `src/codeintel/cli/cyclopts_app.py` | `CliError` |

**Migration**:

```python
# Before (all files)
from codeintel.cli.cli_errors import CliError, ValidationError, ProblemDetail

# After
from codeintel.cli.errors import CliError, ValidationError, ProblemDetail
```

**Batch 1**: Update all `handlers/*.py` files (12 files)  
**Batch 2**: Update remaining files (3 files)  

```bash
rm src/codeintel/cli/cli_errors.py
uv run pytest tests/cli/test_cli_errors.py -v
```

### 4.6 Remove `cli_types.py`

**Dependents (17)**:

| File | Symbols Imported |
|------|------------------|
| `src/codeintel/cli/execution/context.py` | `OutputFormat` |
| `src/codeintel/cli/execution/adapter.py` | `OutputFormat` |
| `src/codeintel/cli/options/common.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_storage.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_plugins.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_ops.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_subsystem.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_graphs.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_build.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_datasets.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_docs.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_common.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_health.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_jobs.py` | `OutputFormat` |
| `src/codeintel/cli/cyclopts_ide.py` | `OutputFormat` |
| `src/codeintel/cli/config/service.py` | `BackendFlags` |
| `src/codeintel/cli/command_context.py` | `OutputFormat` |

**Migration**:

```python
# Before - OutputFormat
from codeintel.cli.cli_types import OutputFormat

# After - OutputFormat
from codeintel.cli.rendering.types import OutputFormat
```

```python
# Before - BackendFlags
from codeintel.cli.cli_types import BackendFlags

# After - BackendFlags
from codeintel.cli.resolution.params import BackendFlags
```

**Batch 1**: Update all `cyclopts_*.py` files (12 files)  
**Batch 2**: Update `execution/`, `options/`, `config/` (4 files)  
**Batch 3**: Update `command_context.py` (1 file)  

```bash
rm src/codeintel/cli/cli_types.py
uv run pytest tests/cli/test_common_module.py -v
```

### 4.7 Remove `results.py`

**Dependents (28)** - Highest complexity:

| Category | Files |
|----------|-------|
| `handlers/*.py` | 12 files |
| `operations/*.py` | 8 files |
| `execution/*.py` | 4 files |
| `rendering/*.py` | 1 file |
| `plugins/*.py` | 1 file |
| Other | 2 files |

**Migration**:

```python
# Before (all files)
from codeintel.cli.results import CliResult, TextRenderer

# After
from codeintel.cli.core import CliResult, TextRenderer
```

**Batch 1**: Update `handlers/*.py` (12 files)  
**Batch 2**: Update `operations/*.py` (8 files)  
**Batch 3**: Update `execution/*.py` (4 files)  
**Batch 4**: Update remaining files (4 files)  

```bash
rm src/codeintel/cli/results.py
uv run pytest tests/cli/ -v
```

### Phase 4 Completion Checklist

**4.1 introspection.py**
- [ ] Delete shim (package exports same symbols)
- [ ] Verify tests pass

**4.2 result_types.py**
- [ ] Update `operations/*.py` (8 files)
- [ ] Update `handlers/ops.py` (1 file)
- [ ] Delete shim
- [ ] Verify tests pass

**4.3 project.py**
- [ ] Delete shim (package exports same symbols)
- [ ] Verify tests pass

**4.4 operation_registry.py**
- [ ] Update `operations/*.py` (9 files)
- [ ] Update remaining files (4 files)
- [ ] Delete shim
- [ ] Verify tests pass

**4.5 cli_errors.py**
- [ ] Update `handlers/*.py` (12 files)
- [ ] Update remaining files (3 files)
- [ ] Delete shim
- [ ] Verify tests pass

**4.6 cli_types.py**
- [ ] Update `cyclopts_*.py` (12 files)
- [ ] Update other files (5 files)
- [ ] Delete shim
- [ ] Verify tests pass

**4.7 results.py**
- [ ] Update `handlers/*.py` (12 files)
- [ ] Update `operations/*.py` (8 files)
- [ ] Update `execution/*.py` (4 files)
- [ ] Update remaining files (4 files)
- [ ] Delete shim
- [ ] Verify tests pass

- [ ] Run full test suite
- [ ] Commit: "chore(cli): remove high-complexity compatibility shims"

---

## Phase 5: Final Cleanup & Verification

**Goal**: Remove supporting infrastructure and verify clean state  
**Effort**: ~1 hour  
**Risk**: 🟢 Minimal  

### 5.1 Remove `_compat.py`

Once all shims are removed, the compatibility utilities are no longer needed.

```bash
rm src/codeintel/cli/_compat.py
```

### 5.2 Final Verification

```bash
# 1. Verify no remaining deprecation warnings
uv run python -W error::DeprecationWarning -c "from codeintel.cli import app, main"

# 2. Verify no imports from old paths
rg "from codeintel\.cli\.(cli_errors|cli_validation|cli_render|cli_types|results|result_types|operation_registry|output|project|jobs|shell|resilience|observability|telemetry|help_system|dry_run|pipelines|job_runner|op_params|introspection) import" src/ tests/

# 3. Full test suite
uv run pytest tests/cli/ -v

# 4. Type checking
uv run pyright src/codeintel/cli/
uv run pyrefly check

# 5. Lint
uv run ruff check src/codeintel/cli/
```

### 5.3 Update Documentation

- [ ] Update `AGENTS.md` if any CLI import guidance exists
- [ ] Update any README files in `cli/` subfolders
- [ ] Remove deprecation notes from docstrings in package `__init__.py` files

### Phase 5 Completion Checklist

- [ ] Delete `_compat.py`
- [ ] Verify no deprecation warnings
- [ ] Verify no old import paths remain
- [ ] Full test suite passes
- [ ] Type checking passes
- [ ] Lint passes
- [ ] Documentation updated
- [ ] Final commit: "chore(cli): complete shim deprecation, remove _compat.py"

---

## Appendix: File-by-File Migration Reference

### Quick Reference: Import Replacements

```python
# cli_errors.py → errors/
from codeintel.cli.cli_errors import X  →  from codeintel.cli.errors import X

# cli_validation.py → introspection/
from codeintel.cli.cli_validation import X  →  from codeintel.cli.introspection import X

# cli_render.py → rendering/
from codeintel.cli.cli_render import X  →  from codeintel.cli.rendering import X

# cli_types.py → rendering/types + resolution/params
from codeintel.cli.cli_types import OutputFormat  →  from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.cli_types import BackendFlags  →  from codeintel.cli.resolution.params import BackendFlags

# results.py → core/
from codeintel.cli.results import X  →  from codeintel.cli.core import X

# result_types.py → core/result_types
from codeintel.cli.result_types import X  →  from codeintel.cli.core.result_types import X

# operation_registry.py → introspection/
from codeintel.cli.operation_registry import X  →  from codeintel.cli.introspection import X

# output.py → core/
from codeintel.cli.output import X  →  from codeintel.cli.core import X

# dry_run.py → project/
from codeintel.cli.dry_run import X  →  from codeintel.cli.project import X

# pipelines.py → project/
from codeintel.cli.pipelines import X  →  from codeintel.cli.project import X

# job_runner.py → jobs/
from codeintel.cli.job_runner import X  →  from codeintel.cli.jobs import X

# help_system.py → introspection/
from codeintel.cli.help_system import X  →  from codeintel.cli.introspection import X

# op_params.py → introspection/
from codeintel.cli.op_params import X  →  from codeintel.cli.introspection import X

# telemetry.py → observability/
from codeintel.cli.telemetry import X  →  from codeintel.cli.observability import X

# Module shims (same path, different resolution):
# shell.py, jobs.py, project.py, resilience.py, observability.py, introspection.py
# No code changes needed - just delete the shim file
```

### Files by Update Count

| Files Needing Updates | Count |
|-----------------------|-------|
| `handlers/*.py` | 12 |
| `cyclopts_*.py` | 13 |
| `operations/*.py` | 9 |
| `execution/*.py` | 4 |
| `commands/*.py` | 3 |
| `resolution/*.py` | 2 |
| `config/*.py` | 2 |
| `completions/*.py` | 1 |
| `plugins/*.py` | 2 |
| `rendering/*.py` | 1 |
| Other root files | 3 |
| Tests | 10 |
| **Total** | **62** |

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 | 1 hour | None |
| Phase 2 | 3 hours | Phase 1 |
| Phase 3 | 4 hours | Phase 2 |
| Phase 4 | 10 hours | Phase 3 |
| Phase 5 | 1 hour | Phase 4 |
| **Total** | **~19 hours** | |

**Recommended Schedule**:
- Sprint 1: Phases 1-2 (easy wins)
- Sprint 2: Phase 3 (medium complexity)
- Sprint 3: Phase 4.1-4.4 (first half of high complexity)
- Sprint 4: Phase 4.5-4.7 + Phase 5 (second half + cleanup)

---

## Risk Mitigation

### Before Each Phase

1. Create a git branch: `git checkout -b cli/remove-shims-phase-N`
2. Run baseline tests: `uv run pytest tests/cli/ -v`
3. Verify no existing deprecation warnings

### During Migration

1. Update imports in batches by folder
2. Run tests after each batch
3. Commit frequently with clear messages

### Rollback Plan

If issues arise:
```bash
git stash  # Save work in progress
git checkout main  # Return to stable state
# Investigate issue
git stash pop  # Restore work when ready
```

### Known Gotchas

1. **Package vs Module confusion**: When a shim file (`shell.py`) and package folder (`shell/`) share a name, Python prefers the file. Deleting the file makes Python find the package. This is the intended behavior.

2. **Circular imports**: If removing a shim causes circular imports, check if the canonical module imports anything from the old path. Update those internal imports first.

3. **Test isolation**: Some tests may have cached imports. Run `pytest --cache-clear` if you see stale behavior.
