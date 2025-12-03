# Epic 8 Implementation Comments

This document summarizes the implementation of Epic 8 (Run Registry & Metadata Tables) and highlights deviations from the original plan in `docs/app-integration-epic-2.md`.

---

## 1. Gateway-Based Architecture (Major Change)

**Original Plan:**
```python
# Engines import and call module-level functions directly
from codeintel.pipeline import run_registry
run_registry.start_run(con, ctx, pipeline_name="...")
run_registry.complete_run(con, run_id, status="succeeded")
```

**Our Implementation:**
```python
# Implementation lives in storage layer, accessed via gateway
from codeintel.storage.run_tracking import PipelineRunTracking
# Engines use:
gateway.runs.start_run(ctx, pipeline_name="...")
gateway.runs.complete_run(run_id, status="succeeded")
```

**Rationale:** The original approach violated two architectural constraints:
1. **DuckDB encapsulation** — Direct `DuckDBPyConnection` usage is only permitted inside `src/codeintel/storage/`
2. **Layer isolation** — Engine modules (`analytics`, `graphs`, `ingestion`) should not import from `codeintel.pipeline`

---

## 2. Module Organization

| Component | Original Plan | Our Implementation |
|-----------|---------------|-------------------|
| Core implementation | `src/codeintel/pipeline/run_registry.py` | `src/codeintel/storage/run_tracking.py` |
| API style | Module-level functions | `PipelineRunTracking` class on gateway |
| Facade for CLI/orchestration | N/A | `src/codeintel/pipeline/run_registry.py` (re-export only) |

---

## 3. `PipelineRunRecord` Structure

**Original Plan:**
```python
@dataclass(frozen=True)
class PipelineRunRecord:
    ctx: RunContext  # Nested RunContext
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_summary: str | None = None
    pipeline_name: str | None = None
```

**Our Implementation:**
```python
@dataclass(frozen=True)
class PipelineRunRecord:
    run_id: str           # Flattened fields
    repo: str
    commit: str
    kind: str
    trigger: str
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime | None = None
    requested_operation: str | None = None
    requested_datasets: tuple[str, ...] = ()
    error_summary: str | None = None
    pipeline_name: str | None = None
```

**Rationale:** Flattened structure is simpler for DB serialization and avoids needing to reconstruct `RunContext` (which requires `repo_root: Path` that we don't persist).

---

## 4. SnapshotRef Handling

**Original Plan:** Assumed `SnapshotRef` has `root` and `profile` fields, with comments about potentially storing them or leaving them as `None`.

**Our Implementation:** `SnapshotRef` actually has `repo_root: Path` (not `root`), and we only persist `repo` and `commit` to the database. Full `SnapshotRef` reconstruction is the caller's responsibility.

---

## 5. API Signature Changes

**Original Plan:** Functions accept `con: DuckDBConnection` as first parameter:
```python
def start_run(con: DuckDBConnection, ctx: RunContext, *, pipeline_name: str | None = None) -> None
```

**Our Implementation:** Methods on `PipelineRunTracking` class (connection is held internally):
```python
def start_run(self, ctx: RunContext, *, pipeline_name: str | None = None) -> None
```

---

## 6. `StepCompletionParams` Addition

**Original Plan:** `complete_step` had many individual parameters.

**Our Implementation:** Added `StepCompletionParams` dataclass to bundle completion parameters, reducing argument count and improving ergonomics:

```python
@dataclass(frozen=True)
class StepCompletionParams:
    run_id: str
    module: ModuleKind
    stage: str
    name: str
    status: StepStatus
    started_at: datetime
    row_counts: Mapping[str, int] | None = None
    extra: Mapping[str, Any] | None = None

    def to_record(self) -> PipelineStepRecord: ...

# Usage:
gateway.runs.complete_step(StepCompletionParams(...))
```

---

## 7. Engine Wiring Pattern

**Original Plan:** Each engine imports `run_registry` and calls functions with `gateway.con`:
```python
from codeintel.pipeline import run_registry
run_registry.start_run(con=gateway.con, ctx=ctx, ...)
```

**Our Implementation:** Engines access via `gateway.runs` property:
```python
runs = context.gateway.runs
runs.start_run(ctx, pipeline_name="...")
```

---

## 8. What We Kept the Same

- ✅ Table schemas (`metadata.pipeline_runs`, `metadata.pipeline_steps`) — identical DDL
- ✅ Indexes — identical
- ✅ Core CRUD operations — same semantics (`start_run`, `complete_run`, `fetch_run`, `record_step`, `fetch_steps`, `start_step`, `complete_step`)
- ✅ Status types (`PipelineStatus`, `StepStatus`, `ModuleKind`) — identical
- ✅ JSON serialization for `requested_datasets`, `row_counts`, `extra` — identical
- ✅ Test structure — similar test cases, adapted for new API

---

## Recommendations for Future Plans

1. **Consider architectural constraints upfront** — Plans should account for existing layering rules (DuckDB encapsulation, module import restrictions)

2. **Gateway-first design** — For new persistence features, start with "how does this fit on the gateway?" rather than standalone modules

3. **Inspect actual types** — `SnapshotRef` actual signature differs from the plan; checking existing code prevents mismatches

4. **Parameter bundling** — For functions with >5 parameters, consider dataclass wrappers upfront

5. **Re-export facades** — When moving implementation to a different layer, plan for facade modules to maintain convenient import paths for appropriate consumers

---

## Files Created/Modified

### Created:
- `src/codeintel/storage/run_tracking.py` — Core implementation with `PipelineRunTracking` class
- `tests/pipeline/test_run_registry.py` — Unit tests for run tracking API
- `tests/pipeline/test_run_registry_integration.py` — Integration tests via gateway

### Modified:
- `src/codeintel/storage/gateway.py` — Added `runs: PipelineRunTracking` property
- `src/codeintel/storage/metadata_bootstrap.py` — Added DDL for pipeline tables
- `src/codeintel/ingestion/recipes/executor.py` — Wired run tracking via `gateway.runs`
- `src/codeintel/analytics/core/pipeline_bridge.py` — Wired run tracking via `gateway.runs`
- `src/codeintel/graphs/runtime/executor.py` — Wired run tracking via `gateway.runs`
- `src/codeintel/pipeline/run_registry.py` — Converted to thin re-export facade

