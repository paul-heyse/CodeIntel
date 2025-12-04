> **Note**: Import paths in this document are historical and may not reflect current structure.
> The ingestion package has been reorganized: `steps/` is now `compute/`, `utilities/` is now `infrastructure/`,
> `tools/` is now `engine/`, `change_tracker.py` is now `tracker.py`, and contracts moved to `validation/`.

# Ingestion epic 5 detailed implementation plan #

Here’s a concrete, patch-level plan for **Epic 5 — Robustness, observability, and ingestion “control plane”**, wired into your *current* ingestion runtime.

Big idea:

* Introduce an **`IngestRun`** model + sink(s).
* Instrument **every ingestion step** via `runner._run_ingest_step`.
* Derive **row-level metrics** (inserted/deleted) from counts before/after for each step’s `produces_tables`.
* Classify errors (tool vs DB vs parse, etc.).
* Optionally emit metrics to OpenTelemetry/Prometheus via a pluggable sink.

I’ll walk through:

1. New `ingestion/ingest_runs.py` (the control-plane model + sink interface).
2. Extending `IngestionContext` with a sink & flag.
3. Instrumenting `_run_ingest_step` to create/populate `IngestRun`.
4. (Optional) DB-backed `core.ingest_runs` table + row model.
5. A couple of tests in `tests/ingestion/test_ingest_run_reporting.py`.

---

## 1. New control-plane module: `ingestion/ingest_runs.py`

**File:** `src/codeintel/ingestion/ingest_runs.py`

This module defines:

* The `IngestRun` record.
* Status/mode enums.
* Error classification helper.
* A simple JSONL sink that writes runs to disk (and can easily be extended / replaced by a DB sink or OTEL sink).

```python
# src/codeintel/ingestion/ingest_runs.py

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Mapping, Protocol

from codeintel.ingestion.tool_runner import ToolExecutionError, ToolNotFoundError
from codeintel.storage.gateway import DuckDBError

log = logging.getLogger(__name__)


class IngestRunStatus(StrEnum):
    """Outcome for an ingestion step run."""

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class IngestRunMode(StrEnum):
    """High-level mode for a dataset step."""

    FULL = "full"
    INCREMENTAL = "incremental"
    UNKNOWN = "unknown"


@dataclass
class IngestRun:
    """
    Structured record describing a single ingestion step execution.

    Fields are deliberately redundant so they can be shipped directly into
    JSONL or a logging DuckDB without further transformation.
    """

    run_id: str
    repo: str
    commit: str
    step: str
    datasets: tuple[str, ...]
    mode: IngestRunMode
    started_at: datetime
    finished_at: datetime | None = None
    duration_s: float | None = None

    rows_before: Mapping[str, int] = field(default_factory=dict)
    rows_after: Mapping[str, int] = field(default_factory=dict)
    rows_inserted: int = 0
    rows_deleted: int = 0

    status: IngestRunStatus = IngestRunStatus.OK
    error_kind: str | None = None
    error_message: str | None = None


class IngestRunSink(Protocol):
    """Abstraction for recording IngestRun objects somewhere."""

    def record(self, run: IngestRun) -> None:
        """Persist or emit the run record."""
        ...


def classify_error(exc: BaseException) -> str:
    """
    Map exceptions into coarse error kinds suitable for dashboards.

    This can be extended over time (e.g. tagging parse errors, validation
    errors, etc.).
    """
    if isinstance(exc, ToolNotFoundError):
        return "tool_not_found"
    if isinstance(exc, ToolExecutionError):
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            return "tool_timeout"
        return "tool_execution"
    if isinstance(exc, DuckDBError):
        return "db_error"
    if isinstance(exc, ValueError):
        return "parse_error"
    return exc.__class__.__name__


@dataclass
class JsonlIngestRunSink:
    """
    Sink that appends IngestRun records as JSON lines on disk.

    Default path suggestion:
        BuildPaths.build_dir / "logs" / "ingest_runs.jsonl"
    """

    path: Path

    def record(self, run: IngestRun) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(run)
        # Serialize datetimes as ISO-8601 strings.
        payload["started_at"] = run.started_at.isoformat()
        if run.finished_at is not None:
            payload["finished_at"] = run.finished_at.isoformat()
        with self.path.open("a", encoding="utf8") as f:
            json.dump(payload, f, sort_keys=True)
            f.write("\n")
```

*(If/when you want DB-backed logging, we’ll add a `DuckDBIngestRunSink` that uses `gateway.log_db_path` or `core.ingest_runs` — see section 4.)*

---

## 2. Extend `IngestionContext` to carry the sink & metrics flag

**File:** `src/codeintel/ingestion/runner.py`

Add imports:

```python
from datetime import UTC, datetime
import uuid

from codeintel.ingestion.ingest_runs import (
    IngestRun,
    IngestRunMode,
    IngestRunStatus,
    IngestRunSink,
    classify_error,
)
from codeintel.storage.gateway import DuckDBError
```

Then extend the `IngestionContext` dataclass to include a sink and a toggle for row metrics:

```python
@dataclass
class IngestionContext:
    """Shared parameters required for all ingestion steps."""

    snapshot: SnapshotRef
    paths: BuildPaths
    gateway: StorageGateway
    tools: ToolsConfig
    code_profile_cfg: ScanProfile
    config_profile_cfg: ScanProfile
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    scip_runner: Callable[..., ScipIngestResult] | None = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None
    change_tracker: change_tracker_module.ChangeTracker | None = None

    # NEW: observability / control-plane hooks
    ingest_run_sink: IngestRunSink | None = None
    enable_run_metrics: bool = False
```

> **Pattern:**
>
> * For production / observability runs, you’ll create `IngestionContext` with `ingest_run_sink=JsonlIngestRunSink(paths.build_dir / "logs" / "ingest_runs.jsonl")` and `enable_run_metrics=True`.
> * For simple tests or local scripts, you can leave these defaulted and get just log lines.

---

## 3. Instrument `_run_ingest_step` to populate `IngestRun`

We’re going to replace `_log_step_start` / `_log_step_done` with richer logic inside `_run_ingest_step`. You can keep those helpers or inline them; here I’ll show replacing `_run_ingest_step` and deprecating the old helpers.

### 3.1. A tiny helper for counting rows per table

Add near the top of `runner.py` (after imports):

```python
def _count_rows(gateway: StorageGateway, table_key: str) -> int:
    """
    Return COUNT(*) for a table key, or 0 if the table is missing.

    This is intentionally forgiving: it should never cause a step to fail
    just because metrics are enabled.
    """
    try:
        row = gateway.con.execute(f"SELECT COUNT(*) FROM {table_key}").fetchone()
    except DuckDBError:
        return 0
    if row is None:
        return 0
    return int(row[0])
```

### 3.2. Heuristic for mode (full vs incremental)

Add a small helper in `runner.py`:

```python
def _guess_run_mode(ctx: IngestionContext, step_name: str) -> IngestRunMode:
    """
    Coarse heuristic for determining run mode.

    - If no change_tracker is present, treat as FULL.
    - For known incremental datasets (those that use run_incremental_ingest),
      label as INCREMENTAL; we do not (yet) distinguish full_rebuild vs true
      incremental inside the harness.
    """
    if ctx.change_tracker is None:
        return IngestRunMode.FULL

    incremental_steps = {
        "ast_extract",
        "cst_extract",
        "scip_ingest",
        "typing_ingest",
        "docstrings_ingest",
    }
    return IngestRunMode.INCREMENTAL if step_name in incremental_steps else IngestRunMode.FULL
```

You can tweak the `incremental_steps` set over time (e.g. once coverage/tests/config join the incremental harness).

### 3.3. Replace `_run_ingest_step` with IngestRun-aware version

**BEFORE** (current):

```python
def _run_ingest_step(
    ctx: IngestionContext,
    name: str,
    *,
    registry: IngestStepRegistry = DEFAULT_REGISTRY,
) -> object | None:
    """
    Run a single ingestion step by name with logging.
    """
    start = _log_step_start(name, ctx)
    step = registry.get(name)
    result = step.run(ctx)
    _log_step_done(name, start, ctx)
    return result
```

**AFTER** — full IngestRun instrumentation:

```python
def _run_ingest_step(
    ctx: IngestionContext,
    name: str,
    *,
    registry: IngestStepRegistry = DEFAULT_REGISTRY,
) -> object | None:
    """
    Run a single ingestion step by name with structured IngestRun reporting.

    This wraps the step execution to:
      - assign a run_id
      - measure duration
      - optionally compute row deltas for produced tables
      - classify and record errors
    """
    step = registry.get(name)
    datasets = tuple(step.produces_tables)
    mode = _guess_run_mode(ctx, name)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)
    start_ts = time.perf_counter()

    # Optional metrics: row counts before the step for each produced table.
    rows_before: dict[str, int] = {}
    if ctx.enable_run_metrics and datasets:
        for table_key in datasets:
            rows_before[table_key] = _count_rows(ctx.gateway, table_key)

    ingest_run = IngestRun(
        run_id=run_id,
        repo=ctx.repo,
        commit=ctx.commit,
        step=name,
        datasets=datasets,
        mode=mode,
        started_at=started_at,
        rows_before=rows_before,
    )

    log.info(
        "ingest start: step=%s repo=%s commit=%s run_id=%s",
        name,
        ctx.repo,
        ctx.commit,
        run_id,
    )

    error: BaseException | None = None
    result: object | None = None

    try:
        result = step.run(ctx)
    except BaseException as exc:  # noqa: BLE001
        error = exc
        # We still want to capture metrics & record the run before re-raising.
        raise
    finally:
        finished_at = datetime.now(UTC)
        duration = time.perf_counter() - start_ts
        ingest_run.finished_at = finished_at
        ingest_run.duration_s = duration

        # Metrics: row counts after, deltas.
        rows_after: dict[str, int] = {}
        if ctx.enable_run_metrics and datasets:
            for table_key in datasets:
                rows_after[table_key] = _count_rows(ctx.gateway, table_key)
        ingest_run.rows_after = rows_after

        if ctx.enable_run_metrics and datasets:
            inserted = 0
            deleted = 0
            for table_key in datasets:
                before = rows_before.get(table_key, 0)
                after = rows_after.get(table_key, 0)
                if after >= before:
                    inserted += after - before
                else:
                    deleted += before - after
            ingest_run.rows_inserted = inserted
            ingest_run.rows_deleted = deleted

        if error is None:
            # Mark SKIPPED for incremental-style steps that changed no rows.
            status = IngestRunStatus.OK
            if (
                mode is IngestRunMode.INCREMENTAL
                and ingest_run.rows_inserted == 0
                and ingest_run.rows_deleted == 0
            ):
                status = IngestRunStatus.SKIPPED
            ingest_run.status = status
            log.info(
                "ingest done: step=%s repo=%s commit=%s run_id=%s "
                "status=%s rows_inserted=%d rows_deleted=%d duration=%.2fs",
                name,
                ctx.repo,
                ctx.commit,
                run_id,
                ingest_run.status.value,
                ingest_run.rows_inserted,
                ingest_run.rows_deleted,
                duration,
            )
        else:
            ingest_run.status = IngestRunStatus.ERROR
            ingest_run.error_kind = classify_error(error)
            ingest_run.error_message = str(error)
            log.error(
                "ingest error: step=%s repo=%s commit=%s run_id=%s "
                "status=%s error_kind=%s",
                name,
                ctx.repo,
                ctx.commit,
                run_id,
                ingest_run.status.value,
                ingest_run.error_kind,
            )

        # Emit to sink / telemetry if configured.
        if ctx.ingest_run_sink is not None:
            try:
                ctx.ingest_run_sink.record(ingest_run)
            except Exception:  # pragma: no cover - sink errors shouldn't break ingestion
                log.exception("Failed to record ingest run for step=%s run_id=%s", name, run_id)

    return result
```

> Note: We’re not calling `_log_step_start` / `_log_step_done` anymore; this is strictly richer. You can leave those helpers in place if other code uses them, or delete them once you’re comfortable with this.

---

## 4. (Optional) `core.ingest_runs` table & row model

If you want to store these runs in DuckDB as well (for dashboards and ad-hoc querying), you can:

1. **Add a schema entry** in `config/schemas/tables.py`:

   ```python
   TABLE_SCHEMAS["core.ingest_runs"] = TableSchema(
       columns=[
           Column("repo", "TEXT"),
           Column("commit", "TEXT"),
           Column("step", "TEXT"),
           Column("run_id", "TEXT"),
           Column("mode", "TEXT"),
           Column("started_at", "TIMESTAMP_TZ"),
           Column("finished_at", "TIMESTAMP_TZ"),
           Column("duration_s", "DOUBLE"),
           Column("rows_inserted", "BIGINT"),
           Column("rows_deleted", "BIGINT"),
           Column("status", "TEXT"),
           Column("error_kind", "TEXT"),
           Column("error_message", "TEXT"),
           Column("datasets", "TEXT"),        # JSON-encoded list
       ],
   )
   ```

2. **Add a row helper** in `storage/rows.py`:

   ```python
   @dataclass(frozen=True)
   class IngestRunRow:
       repo: str
       commit: str
       step: str
       run_id: str
       mode: str
       started_at: datetime
       finished_at: datetime | None
       duration_s: float | None
       rows_inserted: int
       rows_deleted: int
       status: str
       error_kind: str | None
       error_message: str | None
       datasets: str  # JSON

   def ingest_run_to_tuple(run: IngestRun) -> tuple[object, ...]:
       return (
           run.repo,
           run.commit,
           run.step,
           run.run_id,
           run.mode.value,
           run.started_at,
           run.finished_at,
           run.duration_s,
           run.rows_inserted,
           run.rows_deleted,
           run.status.value,
           run.error_kind,
           run.error_message,
           json.dumps(list(run.datasets)),
       )
   ```

3. **Add a `DuckDBIngestRunSink`** in `ingestion/ingest_runs.py`:

   ```python
   @dataclass
   class DuckDBIngestRunSink:
       gateway: StorageGateway

       def record(self, run: IngestRun) -> None:
           from codeintel.storage.rows import ingest_run_to_tuple
           from codeintel.ingestion.common import run_batch

           row = ingest_run_to_tuple(run)
           run_batch(
               self.gateway,
               "core.ingest_runs",
               [row],
               delete_params=None,
               scope=f"{run.repo}@{run.commit}",
           )
   ```

Then wire `ctx.ingest_run_sink = DuckDBIngestRunSink(ctx.gateway)` (or use both DB + JSONL with a composite sink).

If you don’t want to touch schema right now, stick with `JsonlIngestRunSink` — it’s perfectly usable and testable.

---

## 5. Telemetry hooks (OpenTelemetry/Prometheus)

Once `IngestRun` flows through a sink, you can trivially hook it into metrics. For example:

```python
# ingestion/otel_sink.py (optional)

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import metrics

from codeintel.ingestion.ingest_runs import IngestRun, IngestRunSink, IngestRunStatus


@dataclass
class OtelIngestRunSink(IngestRunSink):
    """Example sink that emits IngestRun metrics via OpenTelemetry."""

    meter = metrics.get_meter(__name__)

    def __post_init__(self) -> None:
        self._duration = self.meter.create_histogram(
            "codeintel.ingest.duration",
            unit="s",
            description="Ingestion step duration in seconds",
        )
        self._rows_inserted = self.meter.create_histogram(
            "codeintel.ingest.rows_inserted",
            unit="rows",
            description="Rows inserted by an ingestion step",
        )

    def record(self, run: IngestRun) -> None:
        labels = {
            "repo": run.repo,
            "step": run.step,
            "status": run.status.value,
            "mode": run.mode.value,
        }
        if run.duration_s is not None:
            self._duration.record(run.duration_s, labels)
        self._rows_inserted.record(run.rows_inserted, labels)
```

You can chain sinks by writing a small `MultiSink` that forwards `record` to a list of sinks.

---

## 6. Tests: `tests/ingestion/test_ingest_run_reporting.py`

Finally, some tests to assert:

* `IngestRun` gets populated.
* Error classification is sane.

**File:** `tests/tests/ingestion/test_ingest_run_reporting.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from codeintel.config import ConfigBuilder
from codeintel.ingestion.ingest_runs import (
    IngestRun,
    IngestRunSink,
    IngestRunStatus,
)
from codeintel.ingestion.runner import IngestionContext, run_repo_scan, run_docstrings_ingest
from tests._helpers.gateway import open_ingestion_gateway


@dataclass
class RecordingSink(IngestRunSink):
    runs: List[IngestRun]

    def record(self, run: IngestRun) -> None:
        self.runs.append(run)


def _build_context(tmp_path: Path) -> IngestionContext:
    builder = ConfigBuilder.from_repo_root(tmp_path / "repo")
    config = builder.build()
    snapshot = config.snapshots[0]
    paths = config.paths
    gateway = open_ingestion_gateway(paths.db_path)

    return IngestionContext(
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=config.tools,
        code_profile_cfg=config.code_profile,
        config_profile_cfg=config.config_profile,
    )


def test_ingest_run_success_reporting(tmp_path: Path) -> None:
    # Minimal repo with one Python file to exercise repo_scan + docstrings.
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text('"""docstring"""\n', encoding="utf8")

    ctx = _build_context(tmp_path)
    sink = RecordingSink(runs=[])
    ctx.ingest_run_sink = sink
    ctx.enable_run_metrics = True

    # Run a couple of steps via the high-level helpers.
    run_repo_scan(ctx)
    run_docstrings_ingest(ctx)

    # Expect two runs recorded.
    assert len(sink.runs) == 2
    step_names = {run.step for run in sink.runs}
    assert step_names == {"repo_scan", "docstrings_ingest"}

    # Check metrics for docstrings step.
    doc_run = next(run for run in sink.runs if run.step == "docstrings_ingest")
    assert doc_run.status in {IngestRunStatus.OK, IngestRunStatus.SKIPPED}
    assert "core.docstrings" in doc_run.datasets or "core.docstrings" in doc_run.rows_after
    # At least one row should be inserted for docstrings (unless profile excludes it).
    assert doc_run.rows_inserted >= 0
    assert doc_run.duration_s is not None
    assert doc_run.run_id


def test_ingest_run_error_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force an error in docstrings_ingest and assert error_kind is set."""
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("not:python:code\n", encoding="utf8")

    ctx = _build_context(tmp_path)
    sink = RecordingSink(runs=[])
    ctx.ingest_run_sink = sink
    ctx.enable_run_metrics = False

    # Monkeypatch docstrings to raise a parse error.
    from codeintel.ingestion import docstrings_ingest

    original = docstrings_ingest.ingest_docstrings

    def _boom(*args, **kwargs):
        raise ValueError("synthetic parse error")

    monkeypatch.setattr(docstrings_ingest, "ingest_docstrings", _boom)

    with pytest.raises(ValueError):
        run_docstrings_ingest(ctx)

    error_runs = [run for run in sink.runs if run.step == "docstrings_ingest"]
    assert error_runs, "Expected an error IngestRun to be recorded"
    run = error_runs[0]
    assert run.status is IngestRunStatus.ERROR
    assert run.error_kind in {"parse_error", "ValueError"}
    assert "synthetic parse error" in (run.error_message or "")
```

*(You can refine these tests to be tighter once you wire everything in, but they give you a good starting scaffold.)*

---

If you’d like, next we can zoom in on **one specific incremental step** (e.g. `ast_extract` via `run_incremental_ingest`) and show how to optionally push *per-view* metrics (changed/deleted ratios, full-rebuild vs incremental) into `IngestRun`, using a small observer callback from `run_incremental_ingest`.


# zoom in on ast_extract via run_incremental_ingest #

Nice, this is a fun layer to add.

Let’s wire **AST extraction** so that:

* `run_incremental_ingest` exposes **per-dataset view metrics** (total modules, changed, deleted, full-rebuild vs incremental).
* Those metrics are attached to the **`IngestRun`** record for the `ast_extract` step.
* We do it in a way you can reuse for other incremental steps later (CST, SCIP, typing, docstrings).

I’ll assume you’ve already:

* Introduced `IngestRun` + sink and
* Are wrapping steps via something like `_run_ingest_step` **or** are at least comfortable adding `IngestRun` instrumentation to `run_ast_extract`.

Below is a patch-style plan.

---

## 1. Add an observer hook to `run_incremental_ingest`

We want `run_incremental_ingest` to be able to call a callback with the **`ChangeTrackerDatasetView`** it computed, so we can feed that into `IngestRun`.

### 1.1. Define an observer type

**File:** `ingestion/change_tracker.py`

Add near the top, next to `ModuleFilter`, `RowT`, `ExecutorFactory`:

```python
ModuleFilter = Callable[[ModuleRecord], bool]
RowT = TypeVar("RowT")
ExecutorFactory = Callable[[], Executor]

# NEW: observer hook for incremental ingest
IncrementalIngestObserver = Callable[
    [str, "ChangeTrackerDatasetView"],  # dataset_name, view
    None,
]
```

> Using a `Callable` here is simple and flexible; if you prefer stricter typing, you can make it a `Protocol` instead.

### 1.2. Extend `run_incremental_ingest` signature & call observer

Find `run_incremental_ingest` and change it from:

```python
def run_incremental_ingest[RowT](
    tracker: ChangeTracker,
    ops: IncrementalIngestOps[RowT],
    *,
    executor_factory: ExecutorFactory | None = None,
) -> None:
    """Shared driver for per-module ingestion using a precomputed change tracker."""
    view = tracker.view_for_dataset(dataset_name=ops.dataset_name, module_filter=ops.module_filter)
    ...
```

to:

```python
def run_incremental_ingest[RowT](
    tracker: ChangeTracker,
    ops: IncrementalIngestOps[RowT],
    *,
    executor_factory: ExecutorFactory | None = None,
    observer: IncrementalIngestObserver | None = None,  # NEW
) -> None:
    """
    Shared driver for per-module ingestion using a precomputed change tracker.

    Parameters
    ----------
    tracker
        ChangeTracker containing the precomputed ChangeSet and modules.
    ops
        Dataset-specific operations for delete/process/insert.
    executor_factory
        Optional factory yielding an Executor for parallel processing.
    observer
        Optional callback invoked with (dataset_name, view) before any rows
        are deleted or inserted. This is ideal for recording metrics.
    """
    view = tracker.view_for_dataset(
        dataset_name=ops.dataset_name,
        module_filter=ops.module_filter,
    )

    # NEW: let observers see the view-level metrics (changed/deleted/full rebuild)
    if observer is not None:
        try:
            observer(ops.dataset_name, view)
        except Exception:
            log.exception(
                "Incremental ingest observer failed for dataset %s",
                ops.dataset_name,
            )

    if view.use_full_rebuild and isinstance(ops, SupportsFullRebuild):
        handled = ops.run_full_rebuild(tracker)
        if handled:
            return
    ...
```

The rest of `run_incremental_ingest` stays the same.

Now any code calling `run_incremental_ingest` can opt in to view-level metrics by providing `observer=`.

---

## 2. Enrich `IngestRun` with incremental-view fields

We want `IngestRun` to carry:

* `modules_total`
* `modules_changed`
* `modules_deleted`
* `modules_changed_ratio`
* `modules_deleted_ratio`
* whether this step actually chose **full rebuild** or **incremental** for its dataset.

### 2.1. Extend `IngestRun`

**File:** `ingestion/ingest_runs.py`

Find your `IngestRun` dataclass and extend it like this:

```python
@dataclass
class IngestRun:
    ...
    rows_inserted: int = 0
    rows_deleted: int = 0

    status: IngestRunStatus = IngestRunStatus.OK
    error_kind: str | None = None
    error_message: str | None = None

    # NEW: incremental view metrics (only populated for steps that use
    # run_incremental_ingest and register an observer).
    modules_total: int | None = None
    modules_changed: int | None = None
    modules_deleted: int | None = None
    modules_changed_ratio: float | None = None
    modules_deleted_ratio: float | None = None
    use_full_rebuild: bool | None = None
```

No existing code breaks; these are optional and default to `None`.

---

## 3. Let steps update the current `IngestRun`: `IngestionContext.current_ingest_run`

We need some way for the `observer` callback (which lives inside the step) to **attach metrics to the active `IngestRun`**.

Simplest: let `IngestionContext` carry a pointer to the currently active run.

### 3.1. Add `current_ingest_run` to `IngestionContext`

**File:** `ingestion/runner.py`

Add the import:

```python
from codeintel.ingestion.ingest_runs import (
    IngestRun,
    IngestRunMode,
    IngestRunStatus,
    IngestRunSink,
    classify_error,
)
```

Then extend the dataclass:

```python
@dataclass
class IngestionContext:
    ...
    artifact_writer: Callable[[Path, Path, Path], None] | None = None
    change_tracker: change_tracker_module.ChangeTracker | None = None

    # Observability / control-plane hooks (from Epic 5)
    ingest_run_sink: IngestRunSink | None = None
    enable_run_metrics: bool = False

    # NEW: pointer to the active IngestRun for the current step.
    current_ingest_run: IngestRun | None = None
```

### 3.2. Set/clear `current_ingest_run` in your step wrapper

If you followed the earlier Epic 5 plan and have `_run_ingest_step`, update it like this (only the new bits shown):

```python
def _run_ingest_step(
    ctx: IngestionContext,
    name: str,
    *,
    registry: IngestStepRegistry = DEFAULT_REGISTRY,
) -> object | None:
    step = registry.get(name)
    datasets = tuple(step.produces_tables)
    mode = _guess_run_mode(ctx, name)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)
    start_ts = time.perf_counter()

    ...

    ingest_run = IngestRun(
        run_id=run_id,
        repo=ctx.repo,
        commit=ctx.commit,
        step=name,
        datasets=datasets,
        mode=mode,
        started_at=started_at,
        rows_before=rows_before,
    )

    # NEW: expose this run to downstream code (e.g. incremental observers).
    ctx.current_ingest_run = ingest_run

    log.info("ingest start ...")

    error: BaseException | None = None
    result: object | None = None

    try:
        result = step.run(ctx)
    except BaseException as exc:
        error = exc
        raise
    finally:
        ...
        # Recording / logging code
        ...

        if ctx.ingest_run_sink is not None:
            ...
                ctx.ingest_run_sink.record(ingest_run)
            ...

        # NEW: clear pointer
        ctx.current_ingest_run = None

    return result
```

If you haven’t added `_run_ingest_step` and are still instrumenting each step explicitly, you can apply the same pattern in `run_ast_extract` specifically (see step 5).

---

## 4. Wire ast’s incremental observer through `ingest_python_ast`

Now we want `py_ast_extract.ingest_python_ast` to accept an optional observer and pass it down to `run_incremental_ingest`.

### 4.1. Change `ingest_python_ast` signature & call

**File:** `ingestion/py_ast_extract.py`

First, import the observer type:

```python
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    IncrementalIngestObserver,   # NEW
    run_incremental_ingest,
)
```

Then update `ingest_python_ast`:

#### BEFORE

```python
def ingest_python_ast(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
) -> None:
    """Parse modules listed in core.modules using the stdlib ast and populate tables."""
    worker_count = _resolve_worker_count(max_workers)
    ops = AstIngestOps(
        repo=tracker.change_request.repo,
        commit=tracker.change_request.commit,
    )

    def _executor_factory() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=worker_count)

    run_incremental_ingest(
        tracker,
        ops,
        executor_factory=_executor_factory,
    )
```

#### AFTER — add optional `observer`

```python
def ingest_python_ast(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
    observer: IncrementalIngestObserver | None = None,
) -> None:
    """
    Parse modules listed in core.modules using the stdlib ast and populate tables.

    When an observer is provided, it will be invoked with (dataset_name, view)
    before any rows are deleted or inserted, allowing the caller to record
    view-level metrics.
    """
    worker_count = _resolve_worker_count(max_workers)
    ops = AstIngestOps(
        repo=tracker.change_request.repo,
        commit=tracker.change_request.commit,
    )

    def _executor_factory() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=worker_count)

    run_incremental_ingest(
        tracker,
        ops,
        executor_factory=_executor_factory,
        observer=observer,   # NEW
    )
```

All existing callers still work if they don’t pass `observer=`.

---

## 5. Attach AST view metrics to `IngestRun` via observer

Now we can plug everything together for `ast_extract`.

We’ll assume you’re using `run_ast_extract(ctx)` as your step entrypoint (either directly or wrapped by `_run_ingest_step`).

### 5.1. If you have `_run_ingest_step` + step registry

If your `AstExtractStep` calls `ingest_python_ast` directly, modify its `run` method.

**File:** `ingestion/steps.py` (or where your `AstExtractStep` lives)

```python
from codeintel.ingestion.change_tracker import ChangeTracker, ChangeTrackerDatasetView
from codeintel.ingestion.py_ast_extract import ingest_python_ast
from codeintel.ingestion.runner import IngestionContext  # or IngestionContextProtocol

...

@dataclass(frozen=True)
class AstExtractStep:
    """Parse stdlib AST and persist rows/metrics."""
    name: str = "ast_extract"
    description: str = "Parse Python AST and persist rows + metrics into core.ast_* tables."
    produces_tables: tuple[str, ...] = ("core.ast_nodes", "core.ast_metrics")
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> None:
        tracker = _require_change_tracker(ctx, self.name)

        def _observer(dataset_name: str, view: ChangeTrackerDatasetView) -> None:
            # Attach incremental-view metrics to the current IngestRun, if any.
            run = ctx.current_ingest_run
            if run is None:
                return
            run.modules_total = view.total_modules_considered
            run.modules_changed = view.changed_modules_count
            run.modules_deleted = view.deleted_modules_count
            if view.total_modules_considered > 0:
                run.modules_changed_ratio = (
                    view.changed_modules_count + view.deleted_modules_count
                ) / view.total_modules_considered
                run.modules_deleted_ratio = (
                    view.deleted_modules_count / view.total_modules_considered
                )
            else:
                run.modules_changed_ratio = 0.0
                run.modules_deleted_ratio = 0.0
            run.use_full_rebuild = view.use_full_rebuild

        ingest_python_ast(
            tracker,
            observer=_observer,
        )
```

Notes:

* The observer simply **reads `ctx.current_ingest_run`** and fills in the incremental fields.
* `_require_change_tracker` is whatever helper you already use to ensure `ctx.change_tracker` is present.

### 5.2. If you don’t have step registry and still use `run_ast_extract`

If your `runner.py` still has:

```python
def run_ast_extract(ctx: IngestionContext) -> None:
    start = _log_step_start("ast_extract", ctx)
    tracker = _require_change_tracker(ctx)
    py_ast_extract.ingest_python_ast(tracker)
    _log_step_done("ast_extract", start, ctx)
```

You can either:

* Migrate to `_run_ingest_step` as previously described, or
* Instrument `run_ast_extract` directly with `IngestRun`.

Here’s a direct instrumentation version that uses the same pattern:

```python
from codeintel.ingestion.change_tracker import ChangeTrackerDatasetView
from codeintel.ingestion.py_ast_extract import ingest_python_ast
from codeintel.ingestion.ingest_runs import (
    IngestRun,
    IngestRunMode,
    IngestRunStatus,
)

def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    tracker = _require_change_tracker(ctx)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)
    start_ts = time.perf_counter()

    ingest_run = IngestRun(
        run_id=run_id,
        repo=ctx.repo,
        commit=ctx.commit,
        step="ast_extract",
        datasets=("core.ast_nodes", "core.ast_metrics"),
        mode=IngestRunMode.INCREMENTAL,
        started_at=started_at,
    )
    ctx.current_ingest_run = ingest_run

    def _observer(dataset_name: str, view: ChangeTrackerDatasetView) -> None:
        run = ctx.current_ingest_run
        if run is None:
            return
        run.modules_total = view.total_modules_considered
        run.modules_changed = view.changed_modules_count
        run.modules_deleted = view.deleted_modules_count
        if view.total_modules_considered > 0:
            run.modules_changed_ratio = (
                view.changed_modules_count + view.deleted_modules_count
            ) / view.total_modules_considered
            run.modules_deleted_ratio = (
                view.deleted_modules_count / view.total_modules_considered
            )
        else:
            run.modules_changed_ratio = 0.0
            run.modules_deleted_ratio = 0.0
        run.use_full_rebuild = view.use_full_rebuild

    error: BaseException | None = None
    try:
        ingest_python_ast(
            tracker,
            observer=_observer,
        )
    except BaseException as exc:  # noqa: BLE001
        error = exc
        raise
    finally:
        finished_at = datetime.now(UTC)
        duration = time.perf_counter() - start_ts
        ingest_run.finished_at = finished_at
        ingest_run.duration_s = duration

        # You can add row_before/after logic here or reuse the helper we
        # wrote earlier for all steps; omitted for brevity.

        if error is None:
            ingest_run.status = IngestRunStatus.OK
        else:
            ingest_run.status = IngestRunStatus.ERROR
            ingest_run.error_kind = classify_error(error)
            ingest_run.error_message = str(error)

        if ctx.ingest_run_sink is not None:
            ctx.ingest_run_sink.record(ingest_run)

        ctx.current_ingest_run = None
```

That gives you all the control-plane goodness for `ast_extract` even without a generic `_run_ingest_step`.

---

## 6. Optional: quick test to verify the metrics path

Finally, a tiny test that:

* Runs `repo_scan` + `ast_extract` on a small repo.
* Confirms that `modules_total`, `modules_changed`, etc., get populated on the `IngestRun`.

**File:** `tests/tests/ingestion/test_ingest_run_incremental_ast.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from codeintel.config.primitives import SnapshotRef, BuildPaths
from codeintel.config.builder import RepoScanStepConfig
from codeintel.ingestion.ingest_runs import IngestRun, IngestRunSink
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.runner import IngestionContext, run_ast_extract
from codeintel.ingestion.source_scanner import default_code_profile
from tests._helpers.gateway import open_ingestion_gateway


@dataclass
class RecordingSink(IngestRunSink):
    runs: List[IngestRun]

    def record(self, run: IngestRun) -> None:
        self.runs.append(run)


def test_ast_extract_ingest_run_includes_incremental_view_metrics(tmp_path: Path) -> None:
    # Build a tiny repo.
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("print('a')\n", encoding="utf8")
    (src_dir / "b.py").write_text("print('b')\n", encoding="utf8")

    snapshot = SnapshotRef(repo="demo/ast", commit="abc123", repo_root=repo_root)
    paths = BuildPaths.for_snapshot(snapshot)
    gw = open_ingestion_gateway(paths.db_path)

    profile = default_code_profile(snapshot.repo_root)
    scan_cfg = RepoScanStepConfig(snapshot=snapshot, paths=paths)
    ingest_repo(gw, cfg=scan_cfg, code_profile=profile, apply_schema=True)

    ctx = IngestionContext(
        snapshot=snapshot,
        paths=paths,
        gateway=gw,
        tools=...,   # however you construct ToolsConfig in tests
        code_profile_cfg=profile,
        config_profile_cfg=profile,
    )
    sink = RecordingSink(runs=[])
    ctx.ingest_run_sink = sink
    ctx.enable_run_metrics = False  # we only care about view metrics

    run_ast_extract(ctx)

    ast_runs = [r for r in sink.runs if r.step == "ast_extract"]
    assert ast_runs, "expected at least one ast_extract IngestRun"
    run = ast_runs[0]

    assert run.modules_total is not None and run.modules_total >= 2
    assert run.modules_changed is not None
    assert run.modules_deleted is not None
    assert run.modules_changed_ratio is not None
    assert run.use_full_rebuild in {True, False}
```

---

With these pieces in place, `ast_extract` is now:

* Still using `ChangeTracker + IncrementalIngestOps + run_incremental_ingest`.
* Emitting **per-view incremental metrics** into `IngestRun`:

  * “how many modules changed”
  * “how many deleted”
  * “did we flip to full rebuild?”
* Compatible with your broader Epic 5 control-plane story (JSONL, DuckDB, OTEL, etc.).

You can copy the same pattern for `cst_extract`, `scip_ingest`, `typing_ingest`, and `docstrings_ingest` by:

1. Passing `observer=` into `run_incremental_ingest` in each dataset’s `ingest_*` function.
2. Having each step’s `run` method use a closure that writes `view` metrics to `ctx.current_ingest_run`.
