Perfect, let’s turn Epic 7 into a concrete “you can actually implement this” plan.

I’ll keep the scope tight: **introduce `RunContext` once, then thread it cleanly through ingestion → graphs → analytics → CLI**, without disturbing the architectures you just finished polishing. I’ll note files, signatures, and rough code.

I’ll assume the current behavior from your architecture doc (Snapshot-ish types, per-engine run IDs, etc.). 

---

## 7.1 Introduce `codeintel.runtime.context` + `ids`

### 7.1.1 New package layout

Create:

```text
src/codeintel/runtime/
    __init__.py
    context.py
    ids.py
```

**`codeintel/runtime/context.py`**

```python
# codeintel/runtime/context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple


@dataclass(frozen=True)
class SnapshotRef:
    """
    Canonical description of the code snapshot a run is operating over.
    """
    repo: str              # "github.com/org/repo"
    commit: str            # full SHA
    root: Path             # local checkout root
    profile: str = "default"  # ingest/analysis profile name


RunKind = Literal["ingest", "graphs", "analytics", "full", "op_prereqs"]
TriggerKind = Literal["cli", "http", "mcp", "api"]


@dataclass(frozen=True)
class RunContext:
    """
    Shared run metadata across ingestion, graphs, analytics, and orchestrators.
    """
    run_id: str
    kind: RunKind
    snapshot: SnapshotRef
    trigger: TriggerKind

    # Optional higher-level context
    requested_operation: str | None = None      # e.g. "functions.summary"
    requested_datasets: Tuple[str, ...] = ()    # e.g. ("functions", "modules")
```

**`codeintel/runtime/ids.py`**

```python
# codeintel/runtime/ids.py
from __future__ import annotations

from uuid import uuid4


def new_run_id(prefix: str = "ci") -> str:
    """
    Generate a new opaque run identifier.

    Example: "ci-0f3a0b02c00349b99d62d1b67b4a0c8a"
    """
    return f"{prefix}-{uuid4().hex}"
```

**`codeintel/runtime/__init__.py`**

```python
# codeintel/runtime/__init__.py
from .context import SnapshotRef, RunContext, RunKind, TriggerKind
from .ids import new_run_id

__all__ = [
    "SnapshotRef",
    "RunContext",
    "RunKind",
    "TriggerKind",
    "new_run_id",
]
```

### 7.1.2 Migrate existing `SnapshotRef` to runtime

Somewhere today you already have a `SnapshotRef` or something close (e.g. in graphs config / analytics pipeline). Epic 7’s idea was: **move that here and re‑export from the old module** for compatibility.

Example if it currently lives in `codeintel.graphs.config.steps_graphs`:

```python
# codeintel/graphs/config/steps_graphs.py

# NEW: import the canonical type
from codeintel.runtime import SnapshotRef

# If callers import from here, keep the name available:
__all__ = ["SnapshotRef", ...]
```

Do the same wherever else `SnapshotRef` was previously defined; delete duplicate definitions and import the canonical one.

---

## 7.2 Thread `RunContext` through ingestion

Goal: **ingestion executor understands `RunContext`, but public surface remains compatible.**

I’ll assume you have something like `ingestion/recipes/executor.py` with a core entrypoint `run_recipe(...)` that currently takes `repo_root`, `profile`, etc., and creates an `IngestRun` record. 

### 7.2.1 Define a context-aware API

In `codeintel/ingestion/recipes/executor.py`:

```python
# codeintel/ingestion/recipes/executor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from codeintel.runtime import RunContext, SnapshotRef, new_run_id

from .models import IngestRecipe, IngestOptions, IngestRunResult
# (names: adjust to your real models)


def run_recipe_for_context(
    recipe: "IngestRecipe",
    ctx: RunContext,
    options: "IngestOptions",
) -> "IngestRunResult":
    """
    New preferred entrypoint: run a recipe for a given RunContext.

    - Uses ctx.snapshot.root / profile instead of raw path/profile parameters.
    - Uses ctx.run_id as the ingest run identifier.
    """
    # Derive legacy args
    repo_root: Path = ctx.snapshot.root
    profile: str = ctx.snapshot.profile

    # Optionally thread run_id into your existing IngestRun model
    ingest_run_id = ctx.run_id

    # Internally delegate to the existing implementation to keep behavior stable:
    return _run_recipe_impl(
        recipe=recipe,
        repo_root=repo_root,
        profile=profile,
        options=options,
        run_id=ingest_run_id,
    )
```

Then refactor your existing `run_recipe` to be a **thin adapter**:

```python
def run_recipe(
    recipe: "IngestRecipe",
    repo_root: Path,
    profile: str,
    options: "IngestOptions",
) -> "IngestRunResult":
    """
    Legacy entrypoint; wraps the unified RunContext API.
    """
    snapshot = SnapshotRef(
        repo=_infer_repo_from_root(repo_root),  # if you have this
        commit=_infer_commit(repo_root),        # or take explicit args if available
        root=repo_root,
        profile=profile,
    )
    ctx = RunContext(
        run_id=new_run_id("ingest"),
        kind="ingest",
        snapshot=snapshot,
        trigger="cli",  # or inject from caller later
    )
    return run_recipe_for_context(recipe=recipe, ctx=ctx, options=options)
```

> If your legacy `run_recipe` already takes `repo`/`commit`, just thread those through `SnapshotRef` instead of calling `_infer_*`.

### 7.2.2 Extend `IngestRun` model to carry `run_id` + trigger/operation

In `codeintel/ingestion/run_models.py` (or wherever your run records live):

```python
# codeintel/ingestion/run_models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from codeintel.runtime import RunContext


@dataclass
class IngestRun:
    id: str                        # primary key
    run_id: str                    # == ctx.run_id
    repo: str
    commit: str
    profile: str
    started_at: datetime
    finished_at: datetime | None
    status: Literal["pending", "running", "success", "error"]
    trigger: str | None = None
    requested_operation: str | None = None
    # ... any existing fields
```

When creating an `IngestRun` inside `_run_recipe_impl`, populate from `RunContext`:

```python
def _run_recipe_impl(..., run_id: str, ...) -> IngestRunResult:
    ctx_run = RunContext(
        run_id=run_id,
        kind="ingest",
        snapshot=snapshot,
        trigger=trigger,
        # requested_operation & requested_datasets can be threaded in later
    )

    ingest_run = IngestRun(
        id=_new_ingest_run_pk(),
        run_id=ctx_run.run_id,
        repo=ctx_run.snapshot.repo,
        commit=ctx_run.snapshot.commit,
        profile=ctx_run.snapshot.profile,
        started_at=now(),
        status="running",
        trigger=ctx_run.trigger,
        requested_operation=ctx_run.requested_operation,
    )
    # ... persist ingest_run, then execute, then mark finished
```

**DB schema:** if `IngestRun` is persisted to DuckDB, this step includes a migration adding `run_id`, `trigger`, and `requested_operation` columns. You can keep legacy columns for backward compatibility.

---

## 7.3 Thread `RunContext` into analytics

Per architecture, analytics already has a `SnapshotRef` concept and run manifests in the pipeline bridge. 

Goal here is **not** to redesign analytics; just:

* Add a `RunContext` field to the pipeline’s top-level execution context.
* Prefer a `run_*_for_context(...)` entrypoint.

### 7.3.1 Add `RunContext` to the analytics execution context

Look for something like `ExecutionContext` or `AnalyticsExecutionContext` in `analytics/core/execution_context.py`:

```python
# codeintel/analytics/core/execution_context.py
from dataclasses import dataclass

from codeintel.runtime import RunContext, SnapshotRef


@dataclass
class AnalyticsExecutionContext:
    snapshot: SnapshotRef
    # existing fields: resources, scratch, run_manifest, etc.
    run_context: RunContext | None = None   # NEW
```

Then ensure the `run_id` in the existing context is either:

* removed (if you only used it for logging), or
* kept but always derived from `run_context.run_id`.

Example:

```python
@property
def run_id(self) -> str | None:
    if self.run_context is not None:
        return self.run_context.run_id
    return self._run_id  # deprecated
```

### 7.3.2 Add a context-aware pipeline entrypoint

In `analytics/core/pipeline_bridge.py`:

```python
from codeintel.runtime import RunContext, SnapshotRef, new_run_id
from .execution_context import AnalyticsExecutionContext


def run_pipeline_for_context(
    ctx: RunContext,
    pipeline_name: str,
    options: PipelineOptions,
) -> PipelineResult:
    """
    New preferred entrypoint for analytics: ties everything to RunContext.
    """
    exec_ctx = AnalyticsExecutionContext(
        snapshot=ctx.snapshot,
        # ... other existing fields like resources, manifest, etc.
        run_context=ctx,
    )
    # Internally call the existing run function:
    return _run_pipeline_impl(exec_ctx, pipeline_name, options)
```

Legacy entrypoint:

```python
def run_pipeline(
    snapshot: SnapshotRef,
    pipeline_name: str,
    options: PipelineOptions,
) -> PipelineResult:
    """
    Legacy API for callers that don't know about RunContext yet.
    """
    ctx = RunContext(
        run_id=new_run_id("analytics"),
        kind="analytics",
        snapshot=snapshot,
        trigger="cli",  # or inject later
    )
    return run_pipeline_for_context(ctx=ctx, pipeline_name=pipeline_name, options=options)
```

Anywhere internally that logs run IDs or writes run manifests should now read `exec_ctx.run_id` or `exec_ctx.run_context.run_id`.

---

## 7.4 Thread `RunContext` into graphs

Your `GraphExecutionContext` already has `snapshot: SnapshotRef` and `run_id: str | None`. 

We’ll keep that, but make `run_id` **derive from `RunContext` when present** and add an optional `run_context` field.

### 7.4.1 Extend `GraphExecutionContext`

In `graphs/core/execution_context.py` (names adjusted):

```python
# codeintel/graphs/core/execution_context.py
from dataclasses import dataclass
from typing import Optional

from codeintel.runtime import SnapshotRef, RunContext


@dataclass
class GraphExecutionContext:
    snapshot: SnapshotRef
    resources: ResourceContainer
    _gateway: StorageGateway | None
    _engine: GraphEngine | None
    _catalog_provider: FunctionCatalogProvider | None
    paths: BuildPaths | None
    scratch: GraphRuntimeScratch
    options: object | None
    plugin_name: str | None
    run_id: str | None               # keep for backward compat
    scope: GraphRunScope | None

    # NEW:
    run_context: Optional[RunContext] = None

    # Optional: unify access
    @property
    def effective_run_id(self) -> str | None:
        if self.run_context is not None:
            return self.run_context.run_id
        return self.run_id
```

### 7.4.2 Context-aware graph runtime entrypoint

In the graph runtime (e.g. `graphs/core/runtime.py`):

```python
from codeintel.runtime import RunContext


def run_graphs_for_context(
    ctx: RunContext,
    graph_plan: GraphPlan,
    options: GraphRunOptions,
) -> GraphRunResult:
    """
    New preferred entrypoint for running graphs.
    """
    exec_ctx = GraphExecutionContext(
        snapshot=ctx.snapshot,
        resources=_build_resources(ctx.snapshot),
        _gateway=None,
        _engine=None,
        _catalog_provider=None,
        paths=_build_paths(ctx.snapshot),
        scratch=GraphRuntimeScratch(),
        options=options,
        plugin_name=None,
        run_id=ctx.run_id,
        scope=None,
        run_context=ctx,
    )
    return _run_graphs_impl(exec_ctx, graph_plan)
```

Legacy entrypoint:

```python
def run_graphs(
    snapshot: SnapshotRef,
    graph_plan: GraphPlan,
    options: GraphRunOptions,
) -> GraphRunResult:
    """
    Legacy API, for callers that don't yet pass RunContext.
    """
    ctx = RunContext(
        run_id=new_run_id("graphs"),
        kind="graphs",
        snapshot=snapshot,
        trigger="cli",
    )
    return run_graphs_for_context(ctx=ctx, graph_plan=graph_plan, options=options)
```

If you have a `GraphRun` record (similar to `IngestRun`), add a `run_id` field and populate with `ctx.run_id`.

---

## 7.5 Create a tiny orchestrator that all CLIs call

Now that ingestion, graphs, and analytics all have context-aware entrypoints, we can:

* Centralize **run_id generation**.
* Ensure **all three use the exact same SnapshotRef, run kind, trigger**.
* Make the CLI and OperationCatalog call this instead of bespoke wiring.

### 7.5.1 New orchestrator module

Create:

```text
codeintel/runtime/orchestrator.py
```

With something like:

```python
# codeintel/runtime/orchestrator.py
from __future__ import annotations

from typing import Iterable

from .context import RunContext, SnapshotRef
from .ids import new_run_id

from codeintel.ingestion.recipes.executor import run_recipe_for_context
from codeintel.analytics.core.pipeline_bridge import run_pipeline_for_context
from codeintel.graphs.core.runtime import run_graphs_for_context

# Adjust imports to your actual modules / functions


def new_run_context(
    *,
    snapshot: SnapshotRef,
    kind: str,
    trigger: str,
    requested_operation: str | None = None,
    requested_datasets: Iterable[str] = (),
    prefix: str | None = None,
) -> RunContext:
    """
    Factory for consistent RunContext creation across entrypoints.
    """
    if prefix is None:
        prefix = kind
    return RunContext(
        run_id=new_run_id(prefix),
        kind=kind,        # type: ignore[arg-type] if needed
        snapshot=snapshot,
        trigger=trigger,  # type: ignore[arg-type]
        requested_operation=requested_operation,
        requested_datasets=tuple(requested_datasets),
    )
```

You can add higher-level helpers later, e.g. `run_full_pipeline_for_operation(ctx, op)` that uses the OperationCatalog to decide which parts to run. That’s a natural next epic after 7; it also ties nicely into your domain-first serving & OperationCatalog work.

---

## 7.6 Wire the CLI to `RunContext`

Now that ingestion/graphs/analytics all accept `RunContext`, your CLI should:

1. Build a `SnapshotRef` from CLI flags (`--repo`, `--commit`, `--root`, `--profile`).
2. Create a `RunContext` via `runtime.orchestrator.new_run_context`.
3. Call the context-aware engine functions.

### 7.6.1 Example: analytics CLI command

In `codeintel/cli/analytics_cli.py` (name approximate):

```python
# codeintel/cli/analytics_cli.py
import click
from pathlib import Path

from codeintel.runtime import SnapshotRef
from codeintel.runtime.orchestrator import new_run_context
from codeintel.analytics.core.pipeline_bridge import run_pipeline_for_context


@click.command("run-analytics")
@click.option("--repo", required=True)
@click.option("--commit", required=True)
@click.option("--root", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--profile", default="default")
@click.option("--pipeline", "pipeline_name", required=True)
def run_analytics_cmd(repo: str, commit: str, root: str, profile: str, pipeline_name: str) -> None:
    snapshot = SnapshotRef(
        repo=repo,
        commit=commit,
        root=Path(root),
        profile=profile,
    )
    ctx = new_run_context(
        snapshot=snapshot,
        kind="analytics",
        trigger="cli",
    )
    result = run_pipeline_for_context(ctx=ctx, pipeline_name=pipeline_name, options={})
    # print / log result as you already do
```

Do the same for ingestion and graphs CLI commands.

Later, for “full app” commands like `codeintel op run functions.summary`, you’ll:

* Look up the Operation in the OperationCatalog. 
* Compute `requested_datasets` for that operation.
* Create a `RunContext` with `kind="full"` and `requested_operation="functions.summary"`.
* Run ingestion → graphs → analytics using the same `RunContext`.

---

## 7.7 Tests & gradual migration

### 7.7.1 Unit tests for `RunContext` & `SnapshotRef`

New file: `tests/runtime/test_run_context.py`:

```python
from pathlib import Path

from codeintel.runtime import SnapshotRef, RunContext, new_run_id


def test_new_run_id_prefix():
    rid = new_run_id("test")
    assert rid.startswith("test-")
    assert len(rid.split("-", 1)[1]) == 32


def test_run_context_roundtrip_snapshot():
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
        requested_datasets=("functions",),
    )

    assert ctx.snapshot.repo == "github.com/demo/repo"
    assert ctx.kind == "analytics"
    assert "functions" in ctx.requested_datasets
```

### 7.7.2 Adapters preserve behavior

For each engine:

* Add tests that calling the **legacy** entrypoint (`run_recipe`, `run_graphs`, `run_pipeline`) still produces identical side effects / outputs as before (using your existing test helpers).
* Add new tests that:

  * Create an explicit `RunContext`
  * Call the `*_for_context` functions
  * Verify that:

    * Data is written in the right place.
    * Run metadata tables receive the same `repo`, `commit`, `profile`.
    * The `run_id` in per-engine run records matches `ctx.run_id`.

### 7.7.3 Opt‑in rollout

You can stage the rollout safely:

1. **Phase 1:** Introduce `RunContext` + adapters (`*_for_context`), but keep all upstream callers using legacy APIs.
2. **Phase 2:** Update the CLI to use `RunContext` factory (`new_run_context`).
3. **Phase 3:** Update serving / OperationCatalog / orchestrated flows to require `RunContext` and remove legacy entrypoints once everything is migrated.

---

If you’d like, next step we can layer **“Epic 7.5”**: a small `runtime.orchestrator.run_operation(ctx, operation_id)` that:

* Reads the OperationCatalog to determine required datasets & graphs.
* Runs ingestion/graphs/analytics in the minimal necessary order for that operation.
* Persists a single “top-level run” record keyed by `RunContext.run_id`.

That’s where this unified context really pays off for your “true application” feel.
