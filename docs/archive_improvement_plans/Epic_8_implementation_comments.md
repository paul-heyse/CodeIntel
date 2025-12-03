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

---

## CLI Consolidation Refactoring

This section documents the consolidation of `cli/` into `pipeline/cli/`, executed as a follow-up architectural improvement.

### Motivation

The original codebase had three separate top-level packages related to "running the program":
- `codeintel.cli` — CLI entry points and argument parsing
- `codeintel.pipeline` — Orchestration, Prefect flows, and export logic
- `codeintel.runtime` — Unified run context and orchestration primitives

The goal was to consolidate `cli/` into `pipeline/cli/` to:
1. Group all "run the program" orchestration code together
2. Reduce import path complexity
3. Align with the layering constraints (CLI is an application-layer concern, same as pipeline orchestration)

**Note:** `runtime/` was intentionally kept separate because it provides primitives (`RunContext`, `RunKind`, `TriggerKind`) that are imported by lower-level modules (`analytics`, `graphs`, `ingestion`). Merging `runtime` into `pipeline` would violate the existing layering guardrails.

---

### Final Structure

```
src/codeintel/
├── pipeline/
│   ├── cli/              # NEW — moved from src/codeintel/cli/
│   │   ├── __init__.py   # Re-exports main module
│   │   └── main.py       # CLI entrypoint (~2700 lines)
│   ├── orchestration/    # unchanged
│   ├── export/           # unchanged
│   ├── __init__.py
│   └── run_registry.py
├── runtime/              # UNCHANGED (layering requirement)
│   ├── __init__.py
│   ├── context.py
│   ├── ids.py
│   └── orchestrator.py
└── ... (other packages unchanged)
```

---

### Changes Made

#### 1. Created New Package

**File:** `src/codeintel/pipeline/cli/__init__.py`

```python
"""CLI entry points and helpers for running CodeIntel pipelines from the command line."""

from __future__ import annotations

from codeintel.pipeline.cli import main

__all__ = ["main"]
```

This enables both:
- `from codeintel.pipeline.cli.main import main` (explicit)
- `from codeintel.pipeline.cli import main` (convenient)

---

#### 2. Moved CLI Implementation

**From:** `src/codeintel/cli/main.py`
**To:** `src/codeintel/pipeline/cli/main.py`

The logger was updated to reflect the new location:

```python
# Before
LOG = logging.getLogger("codeintel.cli")

# After
LOG = logging.getLogger("codeintel.pipeline.cli")
```

All other code remained unchanged — the file contains ~2700 lines of CLI commands, argument parsing, and handler functions.

---

#### 3. Deleted Redundant Shim

**Deleted:** `src/codeintel/cli/nx_backend.py`

This file was a thin shim that re-exported `maybe_enable_nx_gpu` from `codeintel.graphs.nx_backend`. It was unnecessary because `main.py` already imports directly from `codeintel.graphs.nx_backend`.

---

#### 4. Updated Entry Point

**File:** `pyproject.toml`

```toml
# Before
[project.scripts]
codeintel = "codeintel.cli.main:main"

# After
[project.scripts]
codeintel = "codeintel.pipeline.cli.main:main"
```

---

#### 5. Updated Layering Guardrails

**File:** `tests/test_layering_serving_imports.py`

```python
# Before
ALLOWED_SERVING_IMPORTERS = {"cli", "pipeline", "serving", "tests"}
ALLOWED_PIPELINE_IMPORTERS = {"pipeline", "serving", "cli", "tests", "storage"}

# After
ALLOWED_SERVING_IMPORTERS = {"pipeline", "serving", "tests"}
ALLOWED_PIPELINE_IMPORTERS = {"pipeline", "serving", "tests", "storage"}
```

Since `cli` is now part of `pipeline`, it no longer needs to be listed separately in the allowed importers.

---

#### 6. Updated Test Imports

All test files that imported from `codeintel.cli` were updated to use `codeintel.pipeline.cli`:

| File | Old Import | New Import |
|------|-----------|------------|
| `tests/cli/test_cli_scope_and_plan.py` | `import codeintel.cli.main as cli_main` | `import codeintel.pipeline.cli.main as cli_main` |
| `tests/cli/test_docs_export_cli.py` | `from codeintel.cli.main import ...` | `from codeintel.pipeline.cli.main import ...` |
| `tests/cli/test_history_timeseries_cli.py` | `from codeintel.cli.main import main` | `from codeintel.pipeline.cli.main import main` |
| `tests/cli/test_pipeline_cli.py` | `from codeintel.cli.main import main, make_parser` | `from codeintel.pipeline.cli.main import main, make_parser` |
| `tests/storage/test_dataset_scaffold.py` | `from codeintel.cli.main import ...` | `from codeintel.pipeline.cli.main import ...` |
| `tests/analytics/test_cli_parser.py` | `from codeintel.cli.main import make_parser` | `from codeintel.pipeline.cli.main import make_parser` |
| `tests/test_pipeline_smoke.py` | `from codeintel.cli.main import main` | `from codeintel.pipeline.cli.main import main` |
| `tests/orchestration/test_prefect_flow_smoke.py` | `from codeintel.cli.main import main as cli_main` | `from codeintel.pipeline.cli.main import main as cli_main` |
| `tests/storage/test_dataset_catalog.py` | `from codeintel.cli.main import run_datasets_catalog` | `from codeintel.pipeline.cli.main import run_datasets_catalog` |
| `tests/docs_export/test_export_validation_flag.py` | `from codeintel.cli import main as cli_main` | `from codeintel.pipeline.cli import main as cli_main` |

---

#### 7. Deleted Obsolete Test File

**Deleted:** `tests/cli/test_nx_backend.py`

This test file tested the now-deleted `codeintel.cli.nx_backend` shim. The underlying functionality (`codeintel.graphs.nx_backend.maybe_enable_nx_gpu`) is tested elsewhere in the graphs test suite.

---

#### 8. Deleted Old Package

**Deleted:** `src/codeintel/cli/` (entire directory)

After moving `main.py` and deleting `nx_backend.py`, the old `cli/` directory was removed entirely, including:
- `src/codeintel/cli/__init__.py`
- `src/codeintel/cli/main.py`
- `src/codeintel/cli/__pycache__/`

---

### Documentation References

The following archived documentation files reference `codeintel.cli` but were **not updated** as they are historical records:

- `docs/archive_improvement_plans/Epic F - Layering Cleanup.md`
- `docs/archive_improvement_plans/architectural_layering_refactor.md`
- `docs/archive_improvement_plans/ARCHITECTURE_LAYERING.md`
- `docs/archive_improvement_plans/networkx_gpu_backend_implementation.md`

These files document the architecture as it existed at the time of writing and serve as historical context.

---

### Verification

All changes were verified with:

1. **Ruff** — `uv run ruff check --fix` (0 errors after auto-fix)
2. **Pyright** — `uv run pyright --warnings --pythonversion=3.13` (0 errors)
3. **Pyrefly** — `uv run pyrefly check` (0 errors)
4. **Pytest** — All 39 affected tests passed:
   - `tests/cli/` (4 test files)
   - `tests/test_layering_serving_imports.py`
   - `tests/analytics/test_cli_parser.py`
   - `tests/orchestration/test_prefect_flow_smoke.py`
   - `tests/storage/test_dataset_scaffold.py`
   - `tests/storage/test_dataset_catalog.py`
   - `tests/docs_export/test_export_validation_flag.py`
   - `tests/test_pipeline_smoke.py`

5. **CLI smoke test** — `uv run codeintel --help` works correctly

---

### Summary of Files

#### Created:
- `src/codeintel/pipeline/cli/__init__.py`
- `src/codeintel/pipeline/cli/main.py` (moved from `src/codeintel/cli/main.py`)

#### Modified:
- `pyproject.toml` — Updated entry point
- `tests/test_layering_serving_imports.py` — Removed `cli` from allowed importers
- `tests/cli/test_cli_scope_and_plan.py` — Updated import
- `tests/cli/test_docs_export_cli.py` — Updated import
- `tests/cli/test_history_timeseries_cli.py` — Updated import
- `tests/cli/test_pipeline_cli.py` — Updated import
- `tests/storage/test_dataset_scaffold.py` — Updated import
- `tests/analytics/test_cli_parser.py` — Updated import
- `tests/test_pipeline_smoke.py` — Updated import
- `tests/orchestration/test_prefect_flow_smoke.py` — Updated import
- `tests/storage/test_dataset_catalog.py` — Updated import
- `tests/docs_export/test_export_validation_flag.py` — Updated import

#### Deleted:
- `src/codeintel/cli/__init__.py`
- `src/codeintel/cli/main.py`
- `src/codeintel/cli/nx_backend.py`
- `tests/cli/test_nx_backend.py`

