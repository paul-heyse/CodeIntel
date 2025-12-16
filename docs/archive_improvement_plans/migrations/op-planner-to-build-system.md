# Migration Guide: op_planner to Build System

This guide explains how to migrate from the legacy `op_planner` module to the new build system.

## Overview

The `codeintel.pipeline.planning.op_planner` module is deprecated. The build system provides:

- **Minimal work computation** via content-addressable hashing
- **Better error messages** via `DatabaseReadinessView`
- **Single source of truth** for target dependencies
- **Declarative readiness model** - ask "what's ready?" without orchestration

## Quick Reference

| Legacy | Build System |
|--------|--------------|
| `build_prereq_summary(op_id, snapshot)` | `get_targets_for_operation(op_id)` |
| `ensure_prerequisites_for_operation(op_id=..., options=...)` | `BuildExecutor.execute(plan)` |
| `OperationPrereqOptions` | `StateValidator`, `BuildResolver`, `PlanGenerator`, `BuildExecutor` |

## Migration Patterns

### Checking if an Operation is Ready

**Before (legacy):**

```python
from codeintel.pipeline.planning.op_planner import (
    build_prereq_summary,
    OperationPrereqOptions,
    ensure_prerequisites_for_operation,
)

# Check prerequisites
summary = build_prereq_summary(op_id, snapshot)
has_work = bool(summary.required_tables)
```

**After (build system):**

```python
from codeintel.core.build.operations import get_targets_for_operation
from codeintel.core.build.readiness import DatabaseReadinessView
from codeintel.core.build.registry import get_target_graph

# Check readiness
op_targets = get_targets_for_operation(op_id)
graph = get_target_graph()
view = DatabaseReadinessView(graph, gateway, snapshot)

# Check if all targets are ready
all_ready = all(
    view[t].is_ready
    for t in op_targets.required_targets
    if t in view
)
```

### Running Prerequisites

**Before (legacy):**

```python
from codeintel.pipeline.planning.op_planner import (
    OperationPrereqOptions,
    ensure_prerequisites_for_operation,
)

options = OperationPrereqOptions(
    snapshot=snapshot,
    paths=paths,
    gateway=gateway,
    tools=tools,
    include_analytics=True,
    trigger="http",
)

run_record = ensure_prerequisites_for_operation(op_id=op_id, options=options)
```

**After (build system):**

```python
from codeintel.core.build import (
    BuildExecutor,
    BuildResolver,
    PlanGenerator,
    StateValidator,
    get_target_graph,
)
from codeintel.core.build.operations import get_targets_for_operation

# Get targets for operation
op_targets = get_targets_for_operation(op_id)
goals = list(op_targets.required_targets)

# Validate state and resolve minimal work
graph = get_target_graph()
validator = StateValidator(graph, gateway, snapshot)
state = validator.validate()

resolver = BuildResolver(graph, state)
resolution = resolver.resolve(goals=goals)

# If nothing to compute, we're done
if not resolution.to_compute:
    return None

# Generate and execute plan
planner = PlanGenerator(graph)
plan = planner.generate(resolution)

executor = BuildExecutor(graph, gateway, snapshot, paths, tools)
result = executor.execute(plan)
```

### Getting Error Information

**Before (legacy):**

```python
# Limited error information
if not has_successful_prereq_run(...):
    # Could only say "no successful run"
    raise RuntimeError("Pipeline prerequisites not satisfied")
```

**After (build system):**

```python
from codeintel.serving.auto_pipeline import (
    diagnose_prereq_failure,
    operation_prereqs_satisfied,
)

if not operation_prereqs_satisfied(gateway, op_id, repo=repo, commit=commit):
    error = diagnose_prereq_failure(gateway, op_id, snapshot)
    # error.fix_command = "codeintel build run ast"
    # error.human_message = "Operation 'function.summary' requires..."
    raise RuntimeError(error.human_message)
```

## Key Concepts

### OperationTargets

The `get_targets_for_operation()` function returns an `OperationTargets` object:

```python
@dataclass(frozen=True)
class OperationTargets:
    operation_id: str
    required_targets: frozenset[str]  # All targets needed
    graph_targets: frozenset[str]     # Targets producing graph runtimes
    data_targets: frozenset[str]      # Targets producing datasets
```

### DatabaseReadinessView

Provides declarative querying of system state:

```python
view = DatabaseReadinessView(graph, gateway, snapshot)

# Check single target
if view["call_graph"].is_ready:
    print("Call graph is computed and current")

# Get fix command
if not view["ast"].is_ready:
    print(f"Fix: {view['ast'].fix_command}")

# Find bottlenecks
bottlenecks = view.bottlenecks()  # Returns targets blocking progress

# Get summary
summary = view.summary()  # {"ready": 5, "stale": 2, "missing": 3, ...}
```

### BuildResult vs PipelineRunRecord

The build system returns `BuildResult` instead of `PipelineRunRecord`:

```python
@dataclass(frozen=True)
class BuildResult:
    run_id: str
    plan: BuildPlan
    status: Literal["succeeded", "failed"]
    completed_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]
    skipped_targets: tuple[str, ...]
    duration_ms: float
    error_summary: str | None

    @property
    def success(self) -> bool:
        return self.status == "succeeded"
```

## CLI Commands

### Before

```bash
codeintel pipeline run-full
codeintel pipeline run-op function.summary
```

### After

```bash
codeintel build run --all
codeintel build run call_graph function_metrics
codeintel build status
```

## Timeline

- **Current**: `op_planner` functions emit `DeprecationWarning`
- **Future**: `op_planner` will be removed; use the build system

## See Also

- `codeintel.core.build` - Build system package
- `codeintel.core.build.operations` - Operation to target mapping
- `codeintel.core.build.readiness` - Declarative readiness queries
- `codeintel.serving.auto_pipeline` - Updated auto-pipeline implementation

