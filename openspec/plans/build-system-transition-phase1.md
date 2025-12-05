# Build System Transition Plan: Phase 1

> **Goal**: Establish the build system as the canonical basis for pipeline orchestration, deprecating redundant systems incrementally.

## Executive Summary

The build system (Phases 2-6) provides:
- **State Validation** - Know what's computed, stale, or missing
- **Minimal Work Resolution** - Compute only what's needed
- **Plan Generation** - Ordered, estimated execution plans
- **Execution** - Actually run the targets
- **Readiness Model** - Declarative "what can I do" queries

This plan identifies systems that become redundant and proposes a careful, incremental transition where each step is fully validated before proceeding.

---

## Current Landscape

### Systems to Evaluate

| System | Location | Purpose | Build System Equivalent |
|--------|----------|---------|------------------------|
| `op_planner` | `pipeline/planning/op_planner.py` | Map operations → pipeline stages | `StateValidator` + `BuildResolver` |
| `auto_pipeline` | `serving/auto_pipeline.py` | Check/run prereqs for serving | `DatabaseReadinessView` + `BuildExecutor` |
| `pipeline run-full` | `cli/main.py` | Run full pipeline | `codeintel build run --module=*` |
| `pipeline run-op` | `cli/main.py` | Run operation prereqs | `codeintel build run <targets>` |

### Dependency Graph

```
auto_pipeline.py
    └── op_planner.py
         └── PipelineSpec / run_pipeline

cli/main.py (pipeline commands)
    └── op_planner.py
    └── run_pipeline
    
HTTP/MCP routes
    └── auto_pipeline.py
```

---

## Phase A: CLI Command Migration

**Scope**: Replace `pipeline run-full` and `pipeline run-op` with deprecation warnings pointing to `build run`.

**Why First**: 
- Self-contained change
- Validates build system in user-facing context
- No runtime dependencies (not used by serving layer)

### A.1: Add Deprecation Warnings

Update `cli/main.py`:

```python
@pipeline_app.command("run-full")
def pipeline_run_full(...) -> None:
    """Run the full pipeline (ingest → graphs → analytics).
    
    .. deprecated::
        Use ``codeintel build run --all`` instead.
    """
    typer.secho(
        "⚠️  'pipeline run-full' is deprecated. Use 'codeintel build run --all' instead.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    # ... existing implementation continues for now ...
```

### A.2: Add `--all` Flag to Build CLI

Update `cli/commands/build.py` to support `--all` for running all targets:

```python
@build_app.command("run")
def build_run(
    targets: list[str] | None = None,
    module: str | None = None,
    all_targets: bool = typer.Option(False, "--all", help="Run all targets"),
    ...
) -> None:
    """Run build targets."""
    if all_targets:
        targets = [t.name for t in graph.all_targets]
```

### A.3: Tests

- Test that `build run --all` produces same outputs as `pipeline run-full`
- Test that `build run goids call_graph` produces same outputs as `pipeline run-op function.summary`

### A.4: Validation Criteria

- [ ] All existing `pipeline run-full` workflows work with `build run --all`
- [ ] Deprecation warning is visible
- [ ] No performance regression

---

## Phase B: Serving Readiness Integration

**Scope**: Replace `operation_prereqs_satisfied()` with build system's `DatabaseReadinessView`.

**Why Second**:
- High value - serving layer is primary consumer
- Readiness model is more accurate (hash-based, not run-based)
- Enables better error messages ("missing ast" vs "no successful run")

### B.1: Create Operation → Targets Mapping

Add to `serving/auto_pipeline.py` or new module:

```python
def targets_for_operation(op_id: str) -> tuple[str, ...]:
    """Map an operation to the build targets it requires.
    
    This bridges the operation catalog to the build system.
    """
    op = get_operation(op_id)
    if op is None:
        return ()
    
    # Map required_datasets to targets
    # Map required_graphs to targets
    targets: set[str] = set()
    
    for dataset in op.required_datasets:
        # Look up which target produces this dataset
        target = TARGET_BY_TABLE_KEY.get(dataset)
        if target:
            targets.add(target)
    
    for graph in op.required_graphs:
        # Look up which target produces this graph
        target = TARGET_BY_GRAPH.get(graph)
        if target:
            targets.add(target)
    
    return tuple(sorted(targets))
```

### B.2: Replace `operation_prereqs_satisfied()`

```python
def operation_prereqs_satisfied(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
) -> bool:
    """Check if prerequisites are satisfied using build system readiness."""
    from codeintel.core.build.readiness import DatabaseReadinessView
    from codeintel.core.build.registry import get_target_graph
    
    targets = targets_for_operation(op_id)
    if not targets:
        return True  # No targets = no prerequisites
    
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path.cwd())
    view = DatabaseReadinessView(get_target_graph(), gateway, snapshot)
    
    # All required targets must be ready
    return all(view[target].is_ready for target in targets)
```

### B.3: Improve Error Messages

The readiness model can provide actionable error messages:

```python
def get_missing_prereqs_for_operation(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
) -> list[str]:
    """Get list of missing prerequisites with fix commands."""
    targets = targets_for_operation(op_id)
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path.cwd())
    view = DatabaseReadinessView(get_target_graph(), gateway, snapshot)
    
    missing = []
    for target in targets:
        if not view[target].is_ready:
            fix = view[target].fix_command or f"codeintel build run {target}"
            missing.append(f"{target}: {fix}")
    
    return missing
```

### B.4: Tests

- Test `operation_prereqs_satisfied` returns same results as before
- Test new error messages are actionable
- Test readiness propagates correctly through operation dependencies

### B.5: Validation Criteria

- [ ] HTTP routes work correctly with new prereq checking
- [ ] MCP tools work correctly with new prereq checking
- [ ] Error messages are clearer than before
- [ ] No false positives/negatives in readiness

---

## Phase C: Auto-Pipeline Execution Migration

**Scope**: Replace `ensure_prereqs_for_http/mcp` to use build system execution.

**Why Third**:
- Depends on Phase B (readiness)
- Enables smart execution (minimal work, not full pipeline)

### C.1: Update `_run_prereqs()`

```python
def _run_prereqs(
    *,
    op_id: str,
    config: ServingConfig,
    gateway: StorageGateway,
    trigger: TriggerKind,
) -> BuildResult | None:
    """Execute prerequisites using build system."""
    from codeintel.core.build.executor import BuildExecutor
    from codeintel.core.build.plan import PlanGenerator
    from codeintel.core.build.readiness import DatabaseReadinessView
    from codeintel.core.build.resolver import BuildResolver
    from codeintel.core.build.state import StateValidator
    
    targets = targets_for_operation(op_id)
    if not targets:
        return None
    
    snapshot = SnapshotRef(
        repo=config.repo,
        commit=config.commit,
        repo_root=config.repo_root or Path.cwd(),
    )
    graph = get_target_graph()
    
    # Check readiness first
    view = DatabaseReadinessView(graph, gateway, snapshot)
    if all(view[t].is_ready for t in targets):
        LOG.debug("auto_pipeline skipped: all targets ready")
        return None
    
    # Resolve minimal work
    validator = StateValidator(graph, gateway, snapshot)
    state = validator.validate_all()
    resolver = BuildResolver(graph, state)
    result = resolver.resolve(goals=list(targets))
    
    if not result.to_compute:
        LOG.debug("auto_pipeline skipped: nothing to compute")
        return None
    
    # Generate and execute plan
    generator = PlanGenerator(graph)
    plan = generator.generate(result)
    
    paths = build_paths_for_serving(config)
    executor = BuildExecutor(
        graph=graph,
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=ToolsConfig.default(),
    )
    
    LOG.info("auto_pipeline executing %d targets for op=%s", len(result.to_compute), op_id)
    return executor.execute(plan)
```

### C.2: Update Return Types

The functions will return `BuildResult` instead of `PipelineRunRecord`. Update callers to handle this.

### C.3: Tests

- End-to-end test: fresh database → serve operation → verify auto-pipeline runs
- Test minimal work: only missing targets are computed
- Test error handling: failed builds produce clear errors

### C.4: Validation Criteria

- [ ] Auto-pipeline runs minimal targets, not full pipeline
- [ ] Serving layer handles BuildResult correctly
- [ ] Performance is same or better

---

## Phase D: Op-Planner Deprecation

**Scope**: Deprecate `op_planner.py` functions, remove after migration period.

**Why Last**:
- Only after all consumers migrated
- Provides fallback during transition

### D.1: Deprecation Markers

```python
import warnings

def ensure_prerequisites_for_operation(...) -> PipelineRunRecord:
    """Run prerequisites for an operation.
    
    .. deprecated::
        Use the build system: ``BuildExecutor.execute()`` with targets
        from ``targets_for_operation()``.
    """
    warnings.warn(
        "ensure_prerequisites_for_operation is deprecated. "
        "Use BuildExecutor with targets_for_operation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation ...
```

### D.2: Migration Guide

Document in CHANGELOG or migration guide:

```markdown
## Migrating from op_planner to Build System

### Before (op_planner)
```python
from codeintel.pipeline.planning import ensure_prerequisites_for_operation

result = ensure_prerequisites_for_operation(op_id=op_id, options=options)
```

### After (build system)
```python
from codeintel.core.build import BuildExecutor, PlanGenerator, BuildResolver
from codeintel.serving.auto_pipeline import targets_for_operation

targets = targets_for_operation(op_id)
# ... resolve, plan, execute ...
```
```

### D.3: Removal Timeline

- **v0.x.0**: Deprecation warnings added
- **v0.x+1.0**: Mark as `@deprecated` in docs
- **v0.x+2.0**: Remove code

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase A: CLI Migration                                          │
│ ├── A.1 Add deprecation warnings to pipeline commands           │
│ ├── A.2 Add --all flag to build run                            │
│ ├── A.3 Write tests                                            │
│ └── A.4 Validate & merge                                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase B: Serving Readiness                                      │
│ ├── B.1 Create operation → targets mapping                     │
│ ├── B.2 Replace operation_prereqs_satisfied                    │
│ ├── B.3 Improve error messages                                 │
│ └── B.4 Validate & merge                                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase C: Auto-Pipeline Execution                                │
│ ├── C.1 Update _run_prereqs to use build system               │
│ ├── C.2 Update return types                                    │
│ └── C.3 Validate & merge                                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase D: Op-Planner Deprecation                                 │
│ ├── D.1 Add deprecation warnings                               │
│ ├── D.2 Write migration guide                                  │
│ └── D.3 Schedule removal                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Impact Summary

### Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `cli/main.py` | A | Deprecation warnings |
| `cli/commands/build.py` | A | Add `--all` flag |
| `serving/auto_pipeline.py` | B, C | Use readiness + executor |
| `pipeline/planning/op_planner.py` | D | Deprecation warnings |

### Files to Add

| File | Phase | Purpose |
|------|-------|---------|
| `serving/build_bridge.py` (optional) | B | Operation → targets mapping |

### Tests to Add

| Test File | Phase | Coverage |
|-----------|-------|----------|
| `tests/cli/test_build_migration.py` | A | CLI equivalence |
| `tests/serving/test_readiness_integration.py` | B | Readiness in serving |
| `tests/serving/test_build_auto_pipeline.py` | C | Auto-pipeline with build |

---

## Success Metrics

1. **No Regression**: All existing tests pass
2. **Better Accuracy**: Readiness uses hashes, not just run records
3. **Minimal Work**: Auto-pipeline computes only what's needed
4. **Clear Errors**: Users know exactly what's missing and how to fix it
5. **Simpler Code**: Less orchestration logic, more declarative state

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hash computation differs from old logic | High | Side-by-side comparison tests |
| Performance regression in readiness | Medium | Pre-compute manifests in batch |
| Breaking API changes | High | Long deprecation period, clear docs |
| Missing edge cases | Medium | Comprehensive test matrix |

---

## What Gets Deprecated (Eventually)

After all phases complete:

1. **`codeintel.pipeline.planning.op_planner`**
   - `build_pipeline_for_operation()`
   - `ensure_prerequisites_for_operation()`
   - `build_prereq_summary()`
   - `OpPrereqSummary`
   - `OperationPrereqOptions`

2. **Parts of `codeintel.serving.auto_pipeline`**
   - `has_successful_prereq_run()` - replaced by readiness
   - Old signature of `_run_prereqs()`

3. **CLI commands**
   - `pipeline run-full` → `build run --all`
   - `pipeline run-op` → `build run <targets>`

---

## Recommended First Step

**Start with Phase A.1**: Add deprecation warnings to the CLI commands. This is:
- Zero-risk (doesn't change behavior)
- Validates the approach with users
- Sets expectations for the transition

```python
# cli/main.py - First change
typer.secho(
    "⚠️  'pipeline run-full' is deprecated. Use 'codeintel build run --all' instead.",
    fg=typer.colors.YELLOW,
    err=True,
)
```

