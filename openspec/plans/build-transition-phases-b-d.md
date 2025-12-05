# Build System Transition: Phases B-D Implementation Plan

This document details the implementation plan for integrating the build system into the serving layer, replacing legacy prerequisite checking and pipeline orchestration.

## Executive Summary

| Phase | Focus | Key Deliverable | Risk |
|-------|-------|-----------------|------|
| **B** | Serving Readiness | `OperationReadinessView` with target mapping | Low |
| **C** | Auto-Pipeline | `BuildExecutor` integration in `auto_pipeline.py` | Medium |
| **D** | Op-Planner Deprecation | Warnings + migration guide | Low |

**Dependency Order**: B → C → D (each phase builds on the previous)

---

## Phase B: Serving Readiness (High Value)

### B.1: Operation → Targets Mapping

**Problem**: Operations declare requirements as `required_datasets` (table keys) and `required_graphs` (runtime names), but the build system operates on **targets** (named outputs). We need a bidirectional mapping.

**Location**: `src/codeintel/core/build/operations.py` (new file)

#### Data Structures

```python
@dataclass(frozen=True)
class OperationTargets:
    """Build targets required for an operation.
    
    Attributes
    ----------
    operation_id
        Operation identifier (e.g., "function.summary").
    required_targets
        Build targets that must be computed for this operation.
    graph_targets
        Targets providing the required graph runtimes.
    data_targets
        Targets providing the required datasets.
    """
    operation_id: str
    required_targets: frozenset[str]
    graph_targets: frozenset[str]
    data_targets: frozenset[str]
```

#### Key Functions

1. **`_build_table_to_target_index()`**: Create mapping from table_key → target_name
   - Iterate `ALL_TARGETS` from registry
   - Build `dict[str, str]` from each target's `tables` attribute

2. **`_build_graph_to_target_index()`**: Create mapping from graph_name → target_name
   - `"callgraph"` → `"call_graph"`
   - `"importgraph"` → `"import_graph"`

3. **`get_targets_for_operation(op_id: str) -> OperationTargets`**:
   - Look up `Operation` from catalog
   - Map `required_datasets` to targets via table index
   - Map `required_graphs` to targets via graph index
   - Return combined `OperationTargets`

4. **`get_targets_for_operation_cached()`**: LRU-cached version

#### Implementation Details

```python
# Table to target mapping (computed once at module load)
_TABLE_TO_TARGET: dict[str, str] = {}
for target in ALL_TARGETS:
    for table in target.tables:
        _TABLE_TO_TARGET[table] = target.name

# Graph runtime to target mapping (explicit)
_GRAPH_TO_TARGET: dict[str, str] = {
    "callgraph": "call_graph",
    "importgraph": "import_graph",
}
```

---

### B.2: Replace `operation_prereqs_satisfied()` with `DatabaseReadinessView`

**Location**: `src/codeintel/serving/auto_pipeline.py`

**Current Implementation** (to replace):
```python
def operation_prereqs_satisfied(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
) -> bool:
    # Checks data-aware + run-based logic
```

**New Implementation**:

```python
def operation_prereqs_satisfied(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
    snapshot: SnapshotRef,
) -> bool:
    """Check if prerequisites are satisfied using build system readiness.
    
    Uses the DatabaseReadinessView to check if all targets required
    by the operation are in 'current' state.
    """
    from codeintel.core.build.operations import get_targets_for_operation
    from codeintel.core.build.readiness import DatabaseReadinessView
    from codeintel.core.build.registry import get_target_graph
    
    # Get required targets for operation
    op_targets = get_targets_for_operation(op_id)
    if not op_targets.required_targets:
        # Operation has no declared requirements - default to satisfied
        return True
    
    # Create readiness view
    graph = get_target_graph()
    view = DatabaseReadinessView(graph, gateway, snapshot)
    
    # Check if all required targets are ready
    for target_name in op_targets.required_targets:
        if target_name not in view:
            continue  # Unknown target, skip
        readiness = view[target_name]
        if not readiness.is_ready:
            return False
    
    return True
```

#### Backward Compatibility

The function signature changes (adds `snapshot` parameter). Update all call sites:
- `ensure_prereqs_for_http()`
- `ensure_prereqs_for_mcp()`
- `_run_prereqs()`

---

### B.3: Better Error Messages

**Problem**: Current errors say "no successful run" which is uninformative. The build system's readiness model provides actionable information.

**Location**: New function in `auto_pipeline.py`

```python
@dataclass(frozen=True)
class PrerequisiteError:
    """Structured error for unmet prerequisites.
    
    Attributes
    ----------
    op_id
        Operation that cannot run.
    missing_targets
        Targets that are not ready.
    bottleneck
        The ultimate blocker target.
    fix_command
        CLI command to fix the issue.
    human_message
        Human-readable explanation.
    """
    op_id: str
    missing_targets: tuple[str, ...]
    bottleneck: str | None
    fix_command: str
    human_message: str


def diagnose_prereq_failure(
    gateway: StorageGateway,
    op_id: str,
    snapshot: SnapshotRef,
) -> PrerequisiteError:
    """Diagnose why prerequisites are not satisfied.
    
    Returns structured error information with actionable fix command.
    """
    from codeintel.core.build.operations import get_targets_for_operation
    from codeintel.core.build.readiness import DatabaseReadinessView
    from codeintel.core.build.registry import get_target_graph
    
    graph = get_target_graph()
    view = DatabaseReadinessView(graph, gateway, snapshot)
    op_targets = get_targets_for_operation(op_id)
    
    missing: list[str] = []
    bottleneck: str | None = None
    
    for target_name in op_targets.required_targets:
        if target_name not in view:
            continue
        readiness = view[target_name]
        if not readiness.is_ready:
            missing.append(target_name)
            if readiness.ultimate_bottleneck:
                bottleneck = readiness.ultimate_bottleneck.target
    
    # Use bottleneck or first missing as fix target
    fix_target = bottleneck or (missing[0] if missing else None)
    fix_command = f"codeintel build run {fix_target}" if fix_target else "codeintel build run --all"
    
    human_message = (
        f"Operation '{op_id}' requires data that hasn't been computed. "
        f"Missing targets: {', '.join(missing)}. "
        f"Run: {fix_command}"
    )
    
    return PrerequisiteError(
        op_id=op_id,
        missing_targets=tuple(missing),
        bottleneck=bottleneck,
        fix_command=fix_command,
        human_message=human_message,
    )
```

#### Integration Points

Update HTTP and MCP error responses to use `PrerequisiteError`:

```python
# In HTTP route handlers
if not operation_prereqs_satisfied(...):
    error = diagnose_prereq_failure(gateway, op_id, snapshot)
    raise HTTPException(
        status_code=503,
        detail={
            "error": "prerequisites_not_met",
            "message": error.human_message,
            "fix_command": error.fix_command,
            "missing_targets": error.missing_targets,
        },
    )
```

---

### B.4: Tests for Phase B

**Location**: `tests/serving/test_operation_readiness.py` (new file)

| Test Case | Description |
|-----------|-------------|
| `test_table_to_target_mapping` | Verify all registered tables map to targets |
| `test_graph_to_target_mapping` | Verify graph runtimes map correctly |
| `test_operation_targets_for_function_summary` | Operation with graph requirements |
| `test_operation_targets_for_dataset_rows` | Operation with no requirements |
| `test_prereqs_satisfied_all_ready` | All required targets computed |
| `test_prereqs_satisfied_missing_data` | Required target has no data |
| `test_diagnose_prereq_failure_bottleneck` | Error shows correct bottleneck |
| `test_diagnose_prereq_failure_fix_command` | Error shows correct CLI command |

---

## Phase C: Auto-Pipeline Execution (Build on B)

### C.1: Replace `_run_prereqs()` with Build System

**Current Implementation** (legacy):
```python
def _run_prereqs(...) -> PipelineRunRecord | None:
    # Uses ensure_prerequisites_for_operation() which runs full pipeline spec
    return ensure_prerequisites_for_operation(op_id=op_id, options=prereq_options)
```

**Problem**: 
- Runs entire pipeline stages (all analytics, all graphs) even if only one target needed
- Doesn't leverage minimal work resolution

**New Implementation**:

```python
def _run_prereqs(
    *,
    op_id: str,
    config: ServingConfig,
    gateway: StorageGateway,
    trigger: TriggerKind,
    snapshot: SnapshotRef,
) -> BuildResult | None:
    """Execute minimal prerequisites for an operation using build system.
    
    Uses BuildResolver to determine minimal work, then BuildExecutor
    to run only what's needed.
    
    Returns
    -------
    BuildResult | None
        Build result if execution occurred, None if already satisfied.
    """
    from codeintel.core.build.executor import BuildExecutor
    from codeintel.core.build.operations import get_targets_for_operation
    from codeintel.core.build.plan import PlanGenerator
    from codeintel.core.build.registry import get_target_graph
    from codeintel.core.build.resolver import BuildResolver
    from codeintel.core.build.state import StateValidator
    
    # Check if already satisfied
    if operation_prereqs_satisfied(gateway, op_id, repo=config.repo, commit=config.commit, snapshot=snapshot):
        LOG.debug("auto_pipeline skipped: prerequisites already satisfied")
        return None
    
    # Get required targets
    op_targets = get_targets_for_operation(op_id)
    if not op_targets.required_targets:
        LOG.debug("auto_pipeline skipped: no targets required for %s", op_id)
        return None
    
    # Build system pipeline
    graph = get_target_graph()
    paths = build_paths_for_serving(config)
    tools = ToolsConfig.default()
    
    # Phase 2: Validate state
    validator = StateValidator(graph, gateway, snapshot)
    state = validator.validate()
    
    # Phase 3: Resolve minimal work
    resolver = BuildResolver(graph, state)
    goals = list(op_targets.required_targets)
    resolution = resolver.resolve(goals)
    
    # Phase 4: Generate plan
    generator = PlanGenerator(graph)
    plan = generator.generate(resolution)
    
    # Check if anything to do
    if not plan.stages:
        LOG.debug("auto_pipeline: nothing to compute, all targets current")
        return None
    
    # Phase 5: Execute
    LOG.info(
        "auto_pipeline executing op=%s targets=%d trigger=%s",
        op_id,
        len(resolution.to_compute),
        trigger,
    )
    executor = BuildExecutor(
        graph=graph,
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=tools,
    )
    return executor.execute(plan)
```

### C.2: Update Return Types

**Change**: Return `BuildResult` instead of `PipelineRunRecord`

**Impact Areas**:
- `_run_prereqs()` return type
- `ensure_prereqs_for_http()` return type
- `ensure_prereqs_for_mcp()` return type
- Any code checking the return value

**Adapter Function** (for backward compatibility during transition):

```python
def _build_result_to_run_record(result: BuildResult) -> PipelineRunRecord:
    """Convert BuildResult to PipelineRunRecord for backward compatibility.
    
    Deprecated: Will be removed when all consumers use BuildResult directly.
    """
    from codeintel.storage.tracking import PipelineRunRecord
    from datetime import datetime, timezone
    
    return PipelineRunRecord(
        run_id=result.run_id,
        kind="build",
        pipeline_name="auto_prereqs",
        status="succeeded" if result.success else "failed",
        trigger="auto_pipeline",
        repo="",  # Not available in BuildResult
        commit="",  # Not available in BuildResult
        started_at=datetime.now(tz=timezone.utc),
        completed_at=datetime.now(tz=timezone.utc),
        requested_datasets=None,
        requested_operation=None,
    )
```

### C.3: Update `ensure_prereqs_for_http()` and `ensure_prereqs_for_mcp()`

```python
def ensure_prereqs_for_http(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> BuildResult | None:
    """Ensure prerequisites are run for an HTTP operation if needed.
    
    Uses the build system for minimal work resolution.
    """
    should_run, gateway, skip_reason = should_run_auto_pipeline(config, backend)
    if not should_run or gateway is None:
        LOG.debug("auto_pipeline skipped: %s", skip_reason)
        return None
    
    snapshot = SnapshotRef(
        repo=config.repo,
        commit=config.commit,
        repo_root=config.repo_root or Path.cwd(),
    )
    
    return _run_prereqs(
        op_id=op_id,
        config=config,
        gateway=gateway,
        trigger="http",
        snapshot=snapshot,
    )
```

### C.4: Remove Legacy Dependencies

After C.1-C.3 are complete, remove:

```python
# From auto_pipeline.py imports, remove:
from codeintel.pipeline.planning.op_planner import (
    OperationPrereqOptions,
    build_prereq_summary,
    ensure_prerequisites_for_operation,
)
```

### C.5: Tests for Phase C

**Location**: `tests/serving/test_auto_pipeline_build.py` (new file)

| Test Case | Description |
|-----------|-------------|
| `test_run_prereqs_minimal_work` | Only required targets computed |
| `test_run_prereqs_already_satisfied` | Returns None when ready |
| `test_run_prereqs_builds_dependencies` | Dependencies computed transitively |
| `test_ensure_prereqs_http_integration` | End-to-end HTTP flow |
| `test_ensure_prereqs_mcp_integration` | End-to-end MCP flow |
| `test_build_result_returned` | Correct return type |

---

## Phase D: Op-Planner Deprecation (Cleanup)

### D.1: Add Deprecation Warnings

**Location**: `src/codeintel/pipeline/planning/op_planner.py`

```python
import warnings

def ensure_prerequisites_for_operation(
    *,
    op_id: str,
    options: OperationPrereqOptions,
) -> PipelineRunRecord:
    """Run prerequisites for an operation.
    
    .. deprecated:: 0.X.0
        Use :func:`codeintel.core.build.executor.BuildExecutor.execute` instead.
        The build system provides minimal work resolution.
    """
    warnings.warn(
        "ensure_prerequisites_for_operation is deprecated. "
        "Use the build system via 'codeintel build run' or BuildExecutor.execute() "
        "for minimal work resolution.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation


def build_pipeline_for_operation(
    op_id: str,
    _snapshot: SnapshotRef,
    *,
    include_analytics: bool = True,
) -> PipelineSpec:
    """Build a PipelineSpec for an operation.
    
    .. deprecated:: 0.X.0
        Use :func:`codeintel.core.build.operations.get_targets_for_operation` 
        to get required targets, then use the build system.
    """
    warnings.warn(
        "build_pipeline_for_operation is deprecated. "
        "Use get_targets_for_operation() + BuildResolver for target-based planning.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation
```

### D.2: Migration Guide

**Location**: `docs/migration/op-planner-to-build.md` (new file)

```markdown
# Migrating from op_planner to Build System

## Overview

The `op_planner` module provided operation-driven pipeline planning that mapped
operations to entire pipeline stages. The new build system provides target-level
granularity with minimal work resolution.

## Key Changes

| Old | New |
|-----|-----|
| `ensure_prerequisites_for_operation()` | `BuildExecutor.execute(plan)` |
| `build_pipeline_for_operation()` | `get_targets_for_operation()` + `BuildResolver` |
| `build_prereq_summary()` | `DatabaseReadinessView` + `OperationTargets` |
| Returns `PipelineRunRecord` | Returns `BuildResult` |

## Migration Steps

### 1. Check Prerequisites

**Before:**
```python
from codeintel.pipeline.planning.op_planner import (
    ensure_prerequisites_for_operation,
    OperationPrereqOptions,
)

run = ensure_prerequisites_for_operation(
    op_id="function.summary",
    options=OperationPrereqOptions(...),
)
```

**After:**
```python
from codeintel.serving.auto_pipeline import (
    operation_prereqs_satisfied,
    ensure_prereqs_for_http,  # or ensure_prereqs_for_mcp
)

# Check if satisfied
if not operation_prereqs_satisfied(gateway, op_id, repo=..., commit=..., snapshot=...):
    result = ensure_prereqs_for_http(op_id=op_id, config=config, backend=backend)
```

### 2. Get Required Targets

**Before:**
```python
from codeintel.pipeline.planning.op_planner import build_prereq_summary

summary = build_prereq_summary(op_id, snapshot)
tables = summary.expanded_tables
```

**After:**
```python
from codeintel.core.build.operations import get_targets_for_operation

targets = get_targets_for_operation(op_id)
target_names = targets.required_targets
```

### 3. Understand Blocking

**Before:**
```python
# Limited to "no successful run"
```

**After:**
```python
from codeintel.serving.auto_pipeline import diagnose_prereq_failure

error = diagnose_prereq_failure(gateway, op_id, snapshot)
print(error.fix_command)  # "codeintel build run ast"
print(error.bottleneck)   # "ast"
```
```

### D.3: Update CLI to Remove Legacy Commands (Already Done in Phase A)

The `pipeline run-full` and `pipeline run-op` commands were already removed in Phase A.

### D.4: Removal Timeline

1. **Immediate (This PR)**: Add deprecation warnings
2. **+1 release**: Mark as `DeprecationWarning` visible by default
3. **+2 releases**: Remove deprecated functions
4. **+3 releases**: Remove `op_planner.py` module entirely

### D.5: Tests for Phase D

| Test Case | Description |
|-----------|-------------|
| `test_ensure_prereqs_emits_warning` | Deprecation warning raised |
| `test_build_pipeline_emits_warning` | Deprecation warning raised |
| `test_warning_includes_migration_hint` | Message includes fix suggestion |

---

## Implementation Order

### Step 1: Phase B.1 - Create `operations.py`

```
src/codeintel/core/build/operations.py  (new)
tests/core/build/test_operations.py     (new)
```

### Step 2: Phase B.2-B.3 - Update `auto_pipeline.py`

```
src/codeintel/serving/auto_pipeline.py  (modify)
tests/serving/test_operation_readiness.py (new)
```

### Step 3: Phase C.1-C.4 - Replace `_run_prereqs()`

```
src/codeintel/serving/auto_pipeline.py  (modify)
tests/serving/test_auto_pipeline_build.py (new)
```

### Step 4: Phase D - Deprecation

```
src/codeintel/pipeline/planning/op_planner.py (modify)
docs/migration/op-planner-to-build.md (new)
```

---

## Quality Gates

All changes must pass:
- `uv run ruff check --fix`
- `uv run pyright --warnings --pythonversion=3.13`
- `uv run pyrefly check`
- `uv run pytest tests/core/build/ tests/serving/ -v`

No `type: ignore` or `noqa` suppressions permitted.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking HTTP/MCP APIs | Return type adapters during transition |
| Performance regression | `DatabaseReadinessView` is lazy/cached |
| Missing target mappings | Comprehensive test coverage for all operations |
| Circular imports | Use TYPE_CHECKING guards consistently |

---

## Success Criteria

1. **Phase B**: `operation_prereqs_satisfied()` uses build system readiness
2. **Phase C**: `_run_prereqs()` uses `BuildExecutor` with minimal work
3. **Phase D**: Deprecation warnings in place, migration guide written
4. All tests pass with zero quality gate errors

