# Epic 9 Implementation Comments

> Detailed comparison of the actual unified pipeline orchestration implementation versus the original `app-integration-epic-3.md` plan. These notes are intended to help improve future implementation planning.

---

## 1. Simplified Planner Architecture

### Epic-3 Plan:
- Used `PipelineContext` from `pipeline.orchestration.core` as an intermediary
- Required `ConfigRegistry` for tool binaries, profiles, and config lookup
- Used helper functions like `_graph_runtime(ctx)`, `_function_catalog(ctx)` from existing orchestration
- Had more complex profile resolution via `profile_from_env(default_code_profile(...))`

### Actual Implementation:
- **Avoided `PipelineContext` entirely** - directly constructed engine-specific contexts
- **Removed `ConfigRegistry` dependency** - simplified to just `ToolsConfig`
- Used existing utility functions like `default_code_profile()` and `default_config_profile()` directly from `infrastructure_utilities.source_scanner`
- Contexts are built inline with minimal dependencies

**Rationale**: The `PipelineContext` from `pipeline.orchestration.core` is designed for step-based orchestration with Prefect. The unified executor doesn't need that complexity since it manages its own lifecycle.

---

## 2. Engine Execution Functions

### Epic-3 Plan:
```python
# For ingestion:
execute_recipe_for_context(recipe, run_context, context, config)

# For analytics:
run_analytics_plugins_for_context(unified_run_context, plan, run_context)
```

### Actual Implementation:
```python
# For ingestion:
execute_recipe(recipe, context, config)

# For graphs:
run_graph_plugins(plan, context)

# For analytics:
run_analytics_plugins(plan, run_context, enable_middleware)
```

**Rationale**: The `*_for_context` variants already do their own run tracking internally (calling `start_run`/`complete_run`). This would cause **duplicate run tracking records**. The unified executor manages run tracking at the orchestrator level and calls the lower-level engine functions that don't duplicate this tracking.

---

## 3. Graph Planning Approach

### Epic-3 Plan:
```python
plan_ctx = GraphPlanContext(
    cfg=ctx.config_builder().graph_metrics(),
    runtime_snapshot=snapshot,
    target=(snapshot.repo, snapshot.commit),
    policy=policy,
    run_options={},
    prior_manifest=prior_manifest,
)
```

### Actual Implementation:
```python
plan_ctx = GraphPlanContext(
    runtime_snapshot=snapshot,
    target=(snapshot.repo, snapshot.commit),
    policy=policy,
    prior_manifest=None,  # Simplified for initial implementation
)
```

**Rationale**: 
1. `GraphPlanContext` doesn't have a `cfg` parameter - the plan assumed an API that doesn't exist
2. `run_options` isn't a parameter either
3. Prior manifest loading was deferred - the existing codebase loads manifests lazily within the runtime

---

## 4. Analytics Context Construction

### Epic-3 Plan:
```python
run_context = AnalyticsRunContext(
    gateway=ctx.gateway,
    snapshot=snapshot,
    graph_runtime=_graph_runtime(ctx),
    cfgs={"graph_metrics": cfg},
    extra={"tool_runner": ctx.tool_runner},
    catalog_provider=_function_catalog(ctx),
)
```

### Actual Implementation:
```python
context = AnalyticsRunContext(
    gateway=gateway,
    graph_runtime=None,  # Resolved lazily by plugins
    cfgs={},
    extra={},
    catalog_provider=None,  # Resolved lazily by plugins
    snapshot=snapshot,
)
```

**Rationale**: The analytics plugins use resource providers (`GraphProvider`, `CatalogProvider`) that resolve dependencies lazily via `ctx.require()`. Pre-populating these would require spinning up the full graph runtime unnecessarily.

---

## 5. Run Tracking API Differences

### Epic-3 Plan:
```python
runs.start_run(run_ctx, pipeline_name=spec.id, status="running")
runs.complete_run(run_id, status=overall_status, error_summary=msg)
# ...
return runs.fetch_run(run_id)  # type: ignore[return-value]
```

### Actual Implementation:
Same signatures, but with proper `None` handling:

```python
run = runs.fetch_run(run_id)
if run is None:
    message = f"Failed to fetch run record for run_id={run_id}"
    raise RuntimeError(message)
return run
```

**Rationale**: The plan used `# type: ignore[return-value]` comments to suppress type errors instead of properly handling the `None` case. This violates the project's strict typing requirements.

---

## 6. Step Recording

### Epic-3 Plan:
```python
def _complete_stage_step(...):
    runs.record_step(
        PipelineStepRecord(
            ...
            started_at=_now(),  # "we don't persist the original start_t"
            completed_at=_now(),
            ...
        )
    )
```

### Actual Implementation:
```python
def _start_stage_step(...) -> datetime:
    started_at = _now()
    runs.record_step(...)
    return started_at  # Return for use in completion

def _complete_stage_step(..., started_at: datetime):
    runs.record_step(
        ...
        started_at=started_at,  # Use actual start time
        completed_at=_now(),
        ...
    )
```

**Rationale**: Properly tracks actual stage duration by passing `started_at` through the execution flow, rather than approximating with the completion timestamp.

---

## 7. Recipe Resolution

### Epic-3 Plan:
```python
def _resolve_ingest_recipe(stage: PipelineStage) -> IngestRecipe:
    name = stage.name
    if name == "builtin.default":
        return FULL_RECIPE
    if name == "builtin.incremental":
        return INCREMENTAL_RECIPE
    if name.startswith("builtin."):
        return get_builtin_recipe(name.split(".", 1)[1])
    return get_builtin_recipe(name)
```

### Actual Implementation:
Added proper error handling:
```python
def _resolve_ingest_recipe(stage: PipelineStage) -> IngestRecipe:
    ...
    if name.startswith("builtin."):
        recipe_name = name.split(".", 1)[1]
        recipe = get_builtin_recipe(recipe_name)
        if recipe is None:
            message = f"Unknown builtin ingestion recipe: {recipe_name}"
            raise ValueError(message)
        return recipe
    ...
```

**Rationale**: `get_builtin_recipe()` can return `None` for unknown recipes. The original plan would pass `None` to the executor, causing cryptic failures later.

---

## 8. Import Structure

### Epic-3 Plan:
- Top-level imports for all dependencies
- No consideration of circular import issues

### Actual Implementation:
```python
# ruff: noqa: PLC0415  # Allow in-function imports

from __future__ import annotations

# Minimal top-level imports
from codeintel.pipeline.spec import PipelineSpec, PipelineStage
from codeintel.runtime import RunKind, TriggerKind, new_run_context

if TYPE_CHECKING:
    # Heavy imports only for type hints
    from codeintel.analytics.core.pipeline_bridge import ...
```

With in-function imports for heavy dependencies:
```python
def _plan_ingestion_stage(...):
    from codeintel.ingestion.infrastructure_utilities.source_scanner import (
        default_code_profile,
        default_config_profile,
    )
    from codeintel.ingestion.recipes.dsl import RecipeOptions
    from codeintel.ingestion.recipes.executor import RecipeExecutorContext
    ...
```

**Rationale**: The codebase has strict typing gates to prevent runtime imports of heavy dependencies. In-function imports defer loading until actually needed and avoid circular dependency issues.

---

## 9. Docstring Style

### Epic-3 Plan:
Brief docstrings without full NumPy format:
```python
def _start_stage_step(...) -> None:
    """Insert a 'running' stage-level step record."""
```

### Actual Implementation:
Full NumPy-style docstrings with all required sections:
```python
def _start_stage_step(...) -> datetime:
    """Record start of a stage-level step.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor.
    run_id
        Run identifier.
    stage
        Pipeline stage being started.

    Returns
    -------
    datetime
        Start timestamp for use in completion.
    """
```

**Rationale**: The project's `AGENTS.md` requires NumPy-style docstrings with Parameters, Returns, and Raises sections for all public/private functions.

---

## 10. Test Structure

### Epic-3 Plan:
- Class-based test fixtures with `@pytest.fixture` methods
- Test classes grouping related tests

### Actual Implementation:
- Module-level functions (not methods) per Ruff PLR6301 requirements
- Fixtures as standalone functions
- Proper return type documentation in fixture docstrings

**Rationale**: Ruff's PLR6301 rule flags methods that don't use `self` and suggests making them functions or static methods. Module-level test functions are cleaner.

---

## Key Lessons for Future Planning

1. **Verify API signatures exist** - Don't assume parameters like `cfg` or `run_options` without checking the actual codebase

2. **Consider run tracking duplication** - When engines have `*_for_context` variants, they likely do their own tracking

3. **Account for lazy resource resolution** - Modern plugin architectures use resource providers, not pre-populated contexts

4. **Handle None returns explicitly** - Don't use `# type: ignore` to suppress legitimate type errors

5. **In-function imports for heavy deps** - Codebase may have import hygiene requirements

6. **Follow project docstring conventions** - Check `AGENTS.md` or style guides for required documentation format

7. **Test structure conventions** - Linter rules may require functions over methods for stateless tests

8. **Verify existing helper availability** - Functions like `_graph_runtime(ctx)` may be tightly coupled to specific contexts (like Prefect flows) and not reusable

9. **Check for `None` return possibilities** - Functions like `get_builtin_recipe()` and `fetch_run()` can return `None`

10. **Consider the PLR0913 rule** - Functions with more than 5 parameters need `# noqa: PLR0913` or restructuring

