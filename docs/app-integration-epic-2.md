Nice, congrats on landing Epic 7 🎉 — now we can give those `RunContext`s somewhere “real” to live.

Below is a concrete, file‑level implementation plan for **Epic 8 – Run Registry & Metadata Tables**, broken into:

1. Schema changes (`metadata.pipeline_runs` / `metadata.pipeline_steps`)
2. `pipeline.run_registry` API
3. Wiring ingestion
4. Wiring analytics
5. Wiring graphs
6. Tests + rollout

I’ll assume the storage bootstrap follows the same pattern you used when adding `schema_version` to `metadata.datasets`. 

---

## 1. Add `metadata.pipeline_runs` and `metadata.pipeline_steps`

**File:** `src/codeintel/storage/metadata_bootstrap.py` (or equivalent)

You already have DDL + bootstrap code for `metadata.datasets`. 
We’ll add new tables and a small helper function.

### 1.1. New DDL

Near your existing `CREATE TABLE` statements, add:

```python
# storage/metadata_bootstrap.py

PIPELINE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS metadata.pipeline_runs (
    run_id              TEXT PRIMARY KEY,
    repo                TEXT NOT NULL,
    commit              TEXT NOT NULL,
    kind                TEXT NOT NULL,  -- "ingest", "graphs", "analytics", "full", "op_prereqs"
    trigger             TEXT NOT NULL,  -- "cli", "http", "mcp", "api", ...
    requested_operation TEXT,
    requested_datasets  JSON,           -- JSON-encoded list of table_keys
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ,
    status              TEXT NOT NULL,  -- "running", "succeeded", "failed", "partial"
    error_summary       TEXT,
    pipeline_name       TEXT            -- optional user-facing name (e.g. "full", "op:functions.summary")
);
"""

PIPELINE_STEPS_DDL = """
CREATE TABLE IF NOT EXISTS metadata.pipeline_steps (
    run_id          TEXT NOT NULL REFERENCES metadata.pipeline_runs(run_id),
    module          TEXT NOT NULL,   -- "ingestion", "graphs", "analytics"
    stage           TEXT NOT NULL,   -- e.g. ingest stage or analytics plugin stage
    name            TEXT NOT NULL,   -- plugin/recipe identifier
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL,   -- "pending", "running", "succeeded", "failed", "skipped"
    row_counts      JSON,            -- table_key -> row_count
    extra           JSON,            -- free-form metrics
    PRIMARY KEY (run_id, module, name)
);
"""

PIPELINE_INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_repo_commit
    ON metadata.pipeline_runs (repo, commit, started_at);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
    ON metadata.pipeline_runs (status, repo, commit);

CREATE INDEX IF NOT EXISTS idx_pipeline_steps_run
    ON metadata.pipeline_steps (run_id, module, stage);
"""
```

DuckDB supports `JSON` and `TIMESTAMPTZ`, so this is consistent with your existing use of `JSON` in dataset contracts and metadata.

### 1.2. Bootstrap / migration

In whatever function currently ensures metadata tables exist (often `bootstrap_metadata(con: DuckDBConnection) -> None`), add:

```python
def bootstrap_metadata(con: DuckDBConnection) -> None:
    # existing: datasets, change tracker, etc.
    con.execute(DATASETS_DDL)
    con.execute("ALTER TABLE metadata.datasets ADD COLUMN IF NOT EXISTS schema_version TEXT;")
    # ...

    # NEW: pipeline tables
    con.execute(PIPELINE_RUNS_DDL)
    con.execute(PIPELINE_STEPS_DDL)
    con.execute(PIPELINE_INDEXES_DDL)
```

No need for explicit `ALTER TABLE` here because the `CREATE TABLE IF NOT EXISTS` pattern is forward‑compatible as you roll this out to new DBs; if you’re already in production and worried about column shape drift, you *can* add `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements for future evolution, but for initial creation this is enough.

---

## 2. Implement `codeintel.pipeline.run_registry`

**File:** `src/codeintel/pipeline/run_registry.py` (new)

This is your “single source of truth” API.

### 2.1. Dataclasses

```python
# codeintel/pipeline/run_registry.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Mapping, Any, Sequence

import json

from codeintel.runtime import RunContext, SnapshotRef
from codeintel.storage.gateway import DuckDBConnection


PipelineStatus = Literal["running", "succeeded", "failed", "partial"]
StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


@dataclass(frozen=True)
class PipelineRunRecord:
    ctx: RunContext
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_summary: str | None = None
    pipeline_name: str | None = None


@dataclass(frozen=True)
class PipelineStepRecord:
    run_id: str
    module: Literal["ingestion", "graphs", "analytics"]
    stage: str               # Ingestion stage / analytics plugin stage / graphs stage
    name: str                # Plugin / recipe / stage identifier
    status: StepStatus
    started_at: datetime
    completed_at: datetime | None = None
    row_counts: Mapping[str, int] | None = None
    extra: Mapping[str, Any] | None = None
```

### 2.2. Internal helpers (serialize/deserialize)

```python
def _now() -> datetime:
    # Always store UTC in DB
    return datetime.now(timezone.utc)


def _serialize_requested_datasets(datasets: Sequence[str]) -> str:
    return json.dumps(list(datasets), separators=(",", ":"))


def _deserialize_requested_datasets(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return tuple(str(x) for x in value)
    except json.JSONDecodeError:
        pass
    return ()
```

### 2.3. Converting between `RunContext` and DB rows

```python
def _ctx_to_row(ctx: RunContext, pipeline_name: str | None, status: PipelineStatus) -> tuple[Any, ...]:
    return (
        ctx.run_id,
        ctx.snapshot.repo,
        ctx.snapshot.commit,
        ctx.kind,
        ctx.trigger,
        ctx.requested_operation,
        _serialize_requested_datasets(ctx.requested_datasets),
        _now(),
        None,                 # completed_at
        status,
        None,                 # error_summary
        pipeline_name,
    )


def _row_to_ctx(row: tuple[Any, ...]) -> PipelineRunRecord:
    (
        run_id,
        repo,
        commit,
        kind,
        trigger,
        requested_operation,
        requested_datasets_raw,
        started_at,
        completed_at,
        status,
        error_summary,
        pipeline_name,
    ) = row

    snapshot = SnapshotRef(
        repo=repo,
        commit=commit,
        # NOTE: we don't know root/profile from the DB; callers can re‑hydrate if needed.
        # For now we store minimal snapshot identity and let the orchestrator provide root.
        root=None,        # type: ignore[arg-type]
        profile="default",
    )
    ctx = RunContext(
        run_id=run_id,
        kind=kind,        # type: ignore[arg-type]
        snapshot=snapshot,
        trigger=trigger,  # type: ignore[arg-type]
        requested_operation=requested_operation,
        requested_datasets=_deserialize_requested_datasets(requested_datasets_raw),
    )
    return PipelineRunRecord(
        ctx=ctx,
        status=status,     # type: ignore[arg-type]
        started_at=started_at,
        completed_at=completed_at,
        error_summary=error_summary,
        pipeline_name=pipeline_name,
    )
```

> If you *do* have a way to persist `root`/`profile` for a snapshot somewhere else (e.g. in a snapshots table), you can join later to fully hydrate `SnapshotRef`. For now, the registry focuses on identifying runs (repo/commit) rather than reconstructing full file system details.

### 2.4. Public API

```python
def start_run(
    con: DuckDBConnection,
    ctx: RunContext,
    *,
    pipeline_name: str | None = None,
    status: PipelineStatus = "running",
) -> None:
    """
    Create or overwrite the pipeline_runs row for a given RunContext.

    Intended to be called exactly once at the beginning of an orchestrated run.
    """
    row = _ctx_to_row(ctx, pipeline_name=pipeline_name, status=status)
    con.execute(
        """
        INSERT OR REPLACE INTO metadata.pipeline_runs (
            run_id,
            repo,
            commit,
            kind,
            trigger,
            requested_operation,
            requested_datasets,
            started_at,
            completed_at,
            status,
            error_summary,
            pipeline_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )
```

```python
def complete_run(
    con: DuckDBConnection,
    run_id: str,
    *,
    status: PipelineStatus,
    error_summary: str | None = None,
) -> None:
    """
    Mark a run finished with a final status and optional error summary.
    """
    con.execute(
        """
        UPDATE metadata.pipeline_runs
        SET status = ?,
            error_summary = ?,
            completed_at = ?
        WHERE run_id = ?
        """,
        [status, error_summary, _now(), run_id],
    )
```

```python
def fetch_run(con: DuckDBConnection, run_id: str) -> PipelineRunRecord | None:
    cur = con.execute(
        """
        SELECT
            run_id,
            repo,
            commit,
            kind,
            trigger,
            requested_operation,
            requested_datasets,
            started_at,
            completed_at,
            status,
            error_summary,
            pipeline_name
        FROM metadata.pipeline_runs
        WHERE run_id = ?
        """,
        [run_id],
    )
    row = cur.fetchone()
    if row is None:
        return None
    return _row_to_ctx(row)
```

Now for steps:

```python
def record_step(
    con: DuckDBConnection,
    record: PipelineStepRecord,
) -> None:
    """
    Insert or replace a pipeline_steps row.

    Can be called:
      - once at the end of a step, or
      - once at 'start' (status='running') and again at 'end' (status='succeeded'/'failed').
    """
    row_counts_json = json.dumps(record.row_counts, separators=(",", ":")) if record.row_counts else None
    extra_json = json.dumps(record.extra, separators=(",", ":")) if record.extra else None

    con.execute(
        """
        INSERT OR REPLACE INTO metadata.pipeline_steps (
            run_id,
            module,
            stage,
            name,
            started_at,
            completed_at,
            status,
            row_counts,
            extra
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record.run_id,
            record.module,
            record.stage,
            record.name,
            record.started_at,
            record.completed_at,
            record.status,
            row_counts_json,
            extra_json,
        ],
    )
```

```python
def fetch_steps(
    con: DuckDBConnection,
    run_id: str,
) -> list[PipelineStepRecord]:
    cur = con.execute(
        """
        SELECT
            run_id,
            module,
            stage,
            name,
            started_at,
            completed_at,
            status,
            row_counts,
            extra
        FROM metadata.pipeline_steps
        WHERE run_id = ?
        ORDER BY module, stage, name
        """,
        [run_id],
    )

    rows = cur.fetchall()
    results: list[PipelineStepRecord] = []
    for (
        run_id,
        module,
        stage,
        name,
        started_at,
        completed_at,
        status,
        row_counts_raw,
        extra_raw,
    ) in rows:
        row_counts = json.loads(row_counts_raw) if row_counts_raw else None
        extra = json.loads(extra_raw) if extra_raw else None
        results.append(
            PipelineStepRecord(
                run_id=run_id,
                module=module,         # type: ignore[arg-type]
                stage=stage,
                name=name,
                status=status,        # type: ignore[arg-type]
                started_at=started_at,
                completed_at=completed_at,
                row_counts=row_counts,
                extra=extra,
            )
        )
    return results
```

Optionally, add small convenience helpers for “start step”/“complete step”:

```python
def start_step(
    con: DuckDBConnection,
    *,
    run_id: str,
    module: Literal["ingestion", "graphs", "analytics"],
    stage: str,
    name: str,
) -> None:
    record = PipelineStepRecord(
        run_id=run_id,
        module=module,
        stage=stage,
        name=name,
        status="running",
        started_at=_now(),
        completed_at=None,
        row_counts=None,
        extra=None,
    )
    record_step(con, record)


def complete_step(
    con: DuckDBConnection,
    *,
    run_id: str,
    module: Literal["ingestion", "graphs", "analytics"],
    stage: str,
    name: str,
    status: StepStatus,
    row_counts: Mapping[str, int] | None = None,
    extra: Mapping[str, Any] | None = None,
    started_at: datetime | None = None,
) -> None:
    record = PipelineStepRecord(
        run_id=run_id,
        module=module,
        stage=stage,
        name=name,
        status=status,
        started_at=started_at or _now(),
        completed_at=_now(),
        row_counts=row_counts,
        extra=extra,
    )
    record_step(con, record)
```

---

## 3. Wire ingestion into the registry

From the architecture doc, ingestion has:

* `IngestPluginPlan` (ordered plugin list)
* `IngestPluginResult` with `table_counts: Mapping[str, int]`
* `IngestExecutionContext` that holds `snapshot` and now a `RunContext` from Epic 7.

And orchestration in `recipes/executor.py` to run a recipe with a given `RunContext`.

### 3.1. Start a run when executing a recipe

**File:** `ingestion/recipes/executor.py`

Right at the top of your `run_recipe_for_context` (from Epic 7), add:

```python
# ingestion/recipes/executor.py
from codeintel.pipeline import run_registry
from codeintel.storage.gateway import StorageGateway  # whatever holds the DuckDBConnection

def run_recipe_for_context(
    recipe: IngestRecipe,
    ctx: RunContext,
    options: IngestOptions,
    *,
    gateway: StorageGateway,
) -> IngestRunResult:
    """
    Execute an ingestion recipe for a given RunContext, now recording pipeline metadata.
    """
    con = gateway.con

    # 1) Register the top-level run if not already registered
    run_registry.start_run(
        con=con,
        ctx=ctx,
        pipeline_name=recipe.name,  # or recipe.id, or "ingest:<recipe-name>"
    )

    # 2) Normal ingestion flow (plan + execute)
    plan = registry.plan(recipe, options=options)
    # (Assume you have something like IngestPluginPlan here)

    for plugin in plan.plugins:
        # Start step
        run_registry.start_step(
            con=con,
            run_id=ctx.run_id,
            module="ingestion",
            stage=plugin.metadata.stage,
            name=plugin.metadata.name,
        )

        started_at = datetime.now(timezone.utc)
        try:
            result = plugin.execute(exec_ctx)  # or executor.execute_plugin(...)
            # result.table_counts: Mapping[table_key, int] or None
            row_counts = getattr(result, "table_counts", None)

            run_registry.complete_step(
                con=con,
                run_id=ctx.run_id,
                module="ingestion",
                stage=plugin.metadata.stage,
                name=plugin.metadata.name,
                status="succeeded",
                row_counts=row_counts,
                extra=None,
                started_at=started_at,
            )
        except Exception as exc:
            run_registry.complete_step(
                con=con,
                run_id=ctx.run_id,
                module="ingestion",
                stage=plugin.metadata.stage,
                name=plugin.metadata.name,
                status="failed",
                row_counts=None,
                extra={"error": repr(exc)},
                started_at=started_at,
            )
            # Re-raise or handle according to ingestion’s existing error policy
            raise

    # After all plugins, mark the run as succeeded/failed based on your own logic.
    run_registry.complete_run(
        con=con,
        run_id=ctx.run_id,
        status="succeeded",
        error_summary=None,
    )

    # 3) Return the existing IngestRunResult as today
    return ingest_result
```

You might already have a higher-level `run_recipe_for_context` that wraps `plan` + `execute`. If so, the `start_run` / `complete_run` calls live there; the per‑plugin `start_step` / `complete_step` live in the executor that loops over the plugins.

The key mapping:

* `module="ingestion"`
* `stage=plugin.metadata.stage` (e.g. `scan`, `parse`, `index`) 
* `name=plugin.metadata.name` (plugin identifier like `"repo_scan"`, `"scip_ingest"`). 

---

## 4. Wire analytics into the registry

Analytics already produces a rich `AnalyticsRunReport` with per‑plugin `AnalyticsRunRecord`s.

```python
@dataclass(frozen=True)
class AnalyticsRunReport:
    repo: str
    commit: str
    run_id: str
    scope: AnalyticsScope
    records: tuple[AnalyticsRunRecord, ...]
    plan: AnalyticsPlanInfo
    tags: Mapping[str, str]
```

Each `AnalyticsRunRecord` has `name`, `kind`, `status`, timestamps, and `meta` with row counts. 

### 4.1. Record steps from the analytics executor

**File:** `analytics/core/executor.py` or `analytics/core/pipeline_bridge.py` (where you call `run_analytics_plugins`)

Right after you obtain the `AnalyticsRunReport`, or during plugin execution, call into the registry:

```python
# analytics/core/pipeline_bridge.py
from codeintel.pipeline import run_registry
from codeintel.runtime import RunContext

def run_pipeline_for_context(
    ctx: RunContext,
    request: AnalyticsPlanRequest,
    gateway: StorageGateway,
    # ...
) -> AnalyticsRunReport:
    plan = plan_analytics_plugin_run(request)
    run_ctx = AnalyticsRunContext(
        gateway=gateway,
        # ...
        snapshot=ctx.snapshot,
        # ...
    )
    report = run_analytics_plugins(plan, run_ctx)

    # Persist pipeline run metadata
    con = gateway.con

    # 1) Ensure the run header exists (or leave this to the orchestrator once you have it)
    run_registry.start_run(
        con=con,
        ctx=ctx,
        pipeline_name="analytics",  # or f"analytics:{request.scope}"
    )

    # 2) Record each plugin as a pipeline step
    for rec in report.records:
        # Assume row_counts stored under meta["row_counts"] as Mapping[str, int]
        meta = rec.meta or {}
        row_counts = meta.get("row_counts") if isinstance(meta.get("row_counts"), Mapping) else None

        run_registry.record_step(
            con=con,
            record=PipelineStepRecord(
                run_id=ctx.run_id,
                module="analytics",
                stage=rec.kind,         # e.g. "function", "coverage", "graph"
                name=rec.name,          # plugin_name
                status=_map_analytics_status(rec.status),
                started_at=rec.started_at,
                completed_at=rec.ended_at,
                row_counts=row_counts,
                extra=meta,
            ),
        )

    # 3) Final run status
    overall_status: PipelineStatus
    if any(r.status == "failed" for r in report.records):
        overall_status = "failed"
    elif any(r.partial for r in report.records):
        overall_status = "partial"
    else:
        overall_status = "succeeded"

    run_registry.complete_run(
        con=con,
        run_id=ctx.run_id,
        status=overall_status,
        error_summary=None,  # or aggregate from rec.error
    )

    return report
```

Helper to map analytics statuses to step statuses:

```python
def _map_analytics_status(status: str) -> StepStatus:
    if status == "succeeded":
        return "succeeded"
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "skipped"
    # Fallback
    return "failed"
```

> Later, when you have a pipeline orchestrator (Epic 9), you can centralise `start_run` / `complete_run` there and let analytics just emit step records. For now, it’s fine to let analytics ensure the header exists (using `INSERT OR REPLACE`).

---

## 5. Wire graphs into the registry

Graphs run plugins via `GraphPluginProtocol.execute(ctx) -> GraphPluginResult`, where metadata includes fields like `name`, `kind`, `stage`, `produces_tables`, `produces_graphs`.

### 5.1. Record per-plugin execution

Graphs likely have a central `run_graphs_for_context` (from Epic 7). Inside the loop over the ordered plugins (often in a `graph_runtime` or `recipes/executor.py`):

```python
# graphs/recipes/executor.py or graphs/core/runtime.py
from codeintel.pipeline import run_registry
from codeintel.pipeline.run_registry import PipelineStepRecord
from codeintel.runtime import RunContext

def run_graph_recipe_for_context(
    recipe: GraphRecipe,
    ctx: RunContext,
    options: GraphRunOptions,
    *,
    gateway: StorageGateway,
) -> GraphRunResult:
    con = gateway.con

    # Ensure run header (if orchestrator hasn't already)
    run_registry.start_run(
        con=con,
        ctx=ctx,
        pipeline_name=recipe.name,
    )

    exec_ctx = GraphExecutionContext(
        snapshot=ctx.snapshot,
        # ...
        run_context=ctx,
    )

    for plugin in recipe.plugins:
        meta = plugin.metadata

        run_registry.start_step(
            con=con,
            run_id=ctx.run_id,
            module="graphs",
            stage=meta.stage,
            name=meta.name,
        )

        started_at = _now()
        try:
            result: GraphPluginResult = plugin.execute(exec_ctx)
            # result.row_counts: Mapping[str, int] or None
            step_extra = {
                "kind": meta.kind,
                "produces_tables": tuple(meta.produces_tables),
                "produces_graphs": [g.value for g in meta.produces_graphs],
                "status": result.status,
                "error": result.error,
            }

            status: StepStatus = "succeeded" if result.status == "ok" else "failed"
            run_registry.complete_step(
                con=con,
                run_id=ctx.run_id,
                module="graphs",
                stage=meta.stage,
                name=meta.name,
                status=status,
                row_counts=result.row_counts,
                extra=step_extra,
                started_at=started_at,
            )
        except Exception as exc:
            run_registry.complete_step(
                con=con,
                run_id=ctx.run_id,
                module="graphs",
                stage=meta.stage,
                name=meta.name,
                status="failed",
                row_counts=None,
                extra={"exception": repr(exc)},
                started_at=started_at,
            )
            # Respect existing error-handling behaviour (severity, etc.)
            raise

    # Compute overall status and complete the run
    overall_status: PipelineStatus = "succeeded"   # or compute from plugin results
    run_registry.complete_run(
        con=con,
        run_id=ctx.run_id,
        status=overall_status,
    )

    return GraphRunResult(...)
```

Mapping assumptions:

* `module="graphs"`
* `stage=plugin.metadata.stage` (e.g. `"goid"`, `"edges"`, `"core"`, `"validation"`) 
* `name=plugin.metadata.name` (e.g. `"callgraph_builder"`, `"core_graph_metrics"`).

---

## 6. Tests & rollout strategy

### 6.1. Unit tests for the registry

**File:** `tests/pipeline/test_run_registry.py` (new)

```python
from datetime import datetime
from pathlib import Path

from codeintel.pipeline import run_registry
from codeintel.runtime import SnapshotRef, RunContext
from codeintel.storage.gateway import StorageGateway


def test_start_and_fetch_run(test_gateway: StorageGateway) -> None:
    con = test_gateway.con
    snapshot = SnapshotRef(
        repo="github.com/demo/repo",
        commit="deadbeef" * 5,
        root=Path("/tmp/repo"),
        profile="default",
    )
    ctx = RunContext(
        run_id="ci-123",
        kind="analytics",
        snapshot=snapshot,
        trigger="cli",
        requested_operation="functions.summary",
        requested_datasets=("analytics.function_metrics",),
    )

    run_registry.start_run(con, ctx, pipeline_name="analytics:full")

    rec = run_registry.fetch_run(con, "ci-123")
    assert rec is not None
    assert rec.ctx.run_id == "ci-123"
    assert rec.ctx.snapshot.repo == "github.com/demo/repo"
    assert rec.pipeline_name == "analytics:full"
    assert rec.status == "running"


def test_record_step_and_fetch_steps(test_gateway: StorageGateway) -> None:
    con = test_gateway.con
    run_id = "ci-456"
    snapshot = SnapshotRef(
        repo="repo",
        commit="deadbeef",
        root=Path("/tmp/repo"),
        profile="default",
    )
    ctx = RunContext(
        run_id=run_id,
        kind="ingest",
        snapshot=snapshot,
        trigger="cli",
    )
    run_registry.start_run(con, ctx, pipeline_name="ingest:default")

    start = datetime.now()
    run_registry.record_step(
        con,
        run_registry.PipelineStepRecord(
            run_id=run_id,
            module="ingestion",
            stage="scan",
            name="repo_scan",
            status="succeeded",
            started_at=start,
            completed_at=start,
            row_counts={"core.modules": 10},
            extra={"note": "ok"},
        ),
    )

    steps = run_registry.fetch_steps(con, run_id)
    assert len(steps) == 1
    step = steps[0]
    assert step.module == "ingestion"
    assert step.stage == "scan"
    assert step.row_counts == {"core.modules": 10}
```

Make sure `test_gateway` runs `bootstrap_metadata(con)` so the new tables exist before tests.

### 6.2. Integration tests for ingestion / analytics / graphs

Use your existing harnesses.

* Ingestion:

  * In `tests/ingestion/test_recipe_pipeline.py`, after running a recipe with `RunContext`, query `metadata.pipeline_runs` and `metadata.pipeline_steps` and assert:

    * There’s one run row with `kind="ingest"`.
    * Steps exist for the plugins executed (at least `repo_scan`).

* Analytics:

  * In `tests/analytics/test_pipeline_bridge.py`, after `run_pipeline_for_context`, assert:

    * `metadata.pipeline_steps` has rows with `module="analytics"` and names matching the plugins from the plan.
    * `row_counts` is present for plugins that write analytics tables.

* Graphs:

  * In `tests/graphs/test_recipe_executor.py`, after `run_graph_recipe_for_context`, assert:

    * `metadata.pipeline_steps` has rows with `module="graphs"` and `stage` consistent with builder/metric/validation.

### 6.3. Rollout order

To minimise risk:

1. **Step 1–2:** Add tables + `run_registry` module, plus unit tests. Nothing calls it yet.
2. **Step 3:** Wire ingestion to record runs/steps; run ingestion tests.
3. **Step 4:** Wire analytics; run analytics tests.
4. **Step 5:** Wire graphs; run graphs tests.

Once all three are recording, you can:

* Add simple CLI commands like `codeintel pipeline status --run-id <id>` backed by `fetch_run` / `fetch_steps` (that’s more Epic 11, but the plumbing is now in place).
* Build UI views over `metadata.pipeline_runs` and `metadata.pipeline_steps` without having to know any engine internals.

---

If you’d like, the next natural step after Epic 8 is to design the **pipeline orchestrator** (Epic 9) that becomes the *only* caller of `start_run` / `complete_run`, with ingestion/graphs/analytics only emitting step records. That’s where you get things like “re‑run from graphs only”, “incremental only”, and “run for operation X” as first‑class flows.
