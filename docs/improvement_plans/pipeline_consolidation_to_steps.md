Nice, this is the fun one: making the *pipeline* live in your code, with Prefect as “just” an execution harness.

I’ll structure this as a concrete refactor you can hand to an AI agent:

1. Goals & constraints
2. Make `steps.py` a self-contained pipeline engine
3. Make `PipelineContext` the one runtime object for steps
4. Phase A: Prefect wraps the whole pipeline (simple, safe)
5. Phase B: Prefect tasks per step using `PIPELINE_STEPS`
6. Clean up duplication and old code
7. Tests & guardrails

I’ll assume the other refactors we designed (core configs, StorageGateway, ToolService, ScanProfile, ChangeTracker) are either done or in progress, and reference them where useful.

---

## 1. Goals & constraints

**End state:**

* `orchestration/steps.py` is the **only place** that defines:

  * What steps exist
  * Their order and dependencies
  * What each step actually does

* `orchestration/prefect_flow.py`:

  * Knows nothing about ingestion/analytics implementation details.
  * Only:

    * Builds configs & runtime services
    * Constructs a `PipelineContext`
    * Uses `PIPELINE_STEPS` to build a Prefect DAG (or call the pipeline directly)
    * Manages gateway lifecycle and logging/snapshot helpers

* You can run the pipeline in two ways:

  * **Pure Python** (no Prefect): useful for local debugging, agent-driven runs, tests.
  * **Via Prefect**: for scheduling, retries, observability.

---

## 2. Make `steps.py` a self-contained pipeline engine

### 2.1 Standardize `PipelineStep` interface

In `src/codeintel/orchestration/steps.py`, define a canonical interface for steps:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, ClassVar


class PipelineStep(Protocol):
    """
    One node in the CodeIntel pipeline DAG.
    """

    name: ClassVar[str]
    deps: ClassVar[tuple[str, ...]]

    def run(self, ctx: "PipelineContext") -> None:
        ...
```

Your existing step dataclasses (`RepoScanStep`, `ScipStep`, `AstStep`, `GoidsStep`, etc.) should implement:

```python
@dataclass
class RepoScanStep:
    name: ClassVar[str] = "repo_scan"
    deps: ClassVar[tuple[str, ...]] = ()

    def run(self, ctx: PipelineContext) -> None:
        ...
```

```python
@dataclass
class AstStep:
    name: ClassVar[str] = "ast_ingest"
    deps: ClassVar[tuple[str, ...]] = ("repo_scan", "goids")  # example

    def run(self, ctx: PipelineContext) -> None:
        ...
```

Key: **all step logic lives here** (they call ingestion/analytics modules, not Prefect).

### 2.2 Canonical `PIPELINE_STEPS` and lookup maps

At the bottom of `steps.py`, define the ordered list and supporting maps:

```python
PIPELINE_STEPS: list[PipelineStep] = [
    RepoScanStep(),
    ScipStep(),
    AstStep(),
    CstStep(),
    GoidsStep(),
    GraphMetricsStep(),
    SubsystemsStep(),
    FunctionHistoryStep(),
    HistoryTimeseriesStep(),
    # ... whatever else you have
]

PIPELINE_STEPS_BY_NAME: dict[str, PipelineStep] = {
    step.name: step for step in PIPELINE_STEPS
}
PIPELINE_DEPS: dict[str, tuple[str, ...]] = {
    step.name: step.deps for step in PIPELINE_STEPS
}
```

Optional: add a tiny DAG validator in `steps.py`:

```python
def _validate_pipeline() -> None:
    for name, deps in PIPELINE_DEPS.items():
        for dep in deps:
            if dep not in PIPELINE_STEPS_BY_NAME:
                raise RuntimeError(f"Step {name!r} depends on unknown step {dep!r}")

_validate_pipeline()
```

This ensures the DAG is internally consistent and independent of Prefect.

### 2.3 Central `run_pipeline` function

Add a function that runs the pipeline sequentially (no Prefect) based on the DAG:

```python
from collections import deque

def _topological_order(step_names: Sequence[str]) -> list[str]:
    """
    Simple topo-sort based on PIPELINE_DEPS for the given subset of steps.
    Assumes DAG is valid.
    """
    deps = {name: set(PIPELINE_DEPS[name]) for name in step_names}
    remaining = set(step_names)
    result: list[str] = []

    no_deps = deque([name for name in step_names if not deps[name]])

    while no_deps:
        name = no_deps.popleft()
        result.append(name)
        remaining.remove(name)
        for other in list(remaining):
            deps[other].discard(name)
            if not deps[other]:
                no_deps.append(other)

    if remaining:
        raise RuntimeError(f"Circular dependency detected among: {sorted(remaining)}")

    return result


def run_pipeline(
    ctx: "PipelineContext",
    *,
    selected_steps: Sequence[str] | None = None,
) -> None:
    """
    Run the pipeline in pure Python, using the DAG in PIPELINE_STEPS.

    selected_steps: optional subset by name (e.g. ["repo_scan", "ast_ingest"]);
                    dependencies are automatically included.
    """
    if selected_steps is None:
        step_names = [step.name for step in PIPELINE_STEPS]
    else:
        step_names = sorted(set(selected_steps))  # gather unique

        # Ensure dependencies of selected steps are included
        expanded: set[str] = set()

        def _add_with_deps(name: str) -> None:
            if name in expanded:
                return
            for dep in PIPELINE_DEPS[name]:
                _add_with_deps(dep)
            expanded.add(name)

        for name in step_names:
            _add_with_deps(name)
        step_names = list(expanded)

    ordered = _topological_order(step_names)

    for name in ordered:
        step = PIPELINE_STEPS_BY_NAME[name]
        ctx.logger.info("Running pipeline step %s", name)
        step.run(ctx)
```

Now you can run the whole thing locally with:

```python
ctx = PipelineContext(...)
run_pipeline(ctx)
```

…with no Prefect at all.

---

## 3. Make `PipelineContext` the one runtime object for steps

In `steps.py`, define `PipelineContext` based on our earlier core-config refactor:

```python
from dataclasses import dataclass
from codeintel.core.config import SnapshotConfig, ExecutionConfig, PathsConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.ingestion.change_tracker import ChangeTracker
from codeintel.ingestion.tool_service import ToolService

import logging


@dataclass
class PipelineContext:
    snapshot: SnapshotConfig
    execution: ExecutionConfig
    paths: PathsConfig

    gateway: StorageGateway
    tool_service: ToolService

    change_tracker: ChangeTracker | None = None
    logger: logging.Logger = logging.getLogger("codeintel.pipeline")
```

Each step’s `run` implementation should:

* Use `ctx.gateway`, `ctx.tool_service`, `ctx.snapshot`, `ctx.execution`, `ctx.paths`.
* Never import Prefect or call Prefect-specific APIs.
* Set `ctx.change_tracker` if they need to pass it down (e.g. `RepoScanStep`).

Example:

```python
@dataclass
class RepoScanStep:
    name: ClassVar[str] = "repo_scan"
    deps: ClassVar[tuple[str, ...]] = ()

    def run(self, ctx: PipelineContext) -> None:
        from codeintel.ingestion.repo_scan import ingest_repo

        tracker = ingest_repo(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            execution=ctx.execution,
            paths=ctx.paths,
        )
        ctx.change_tracker = tracker
```

---

## 4. Phase A: Prefect wraps the whole pipeline (simple, safe)

First, do the **minimal** move: use `steps.run_pipeline` inside a *single* Prefect task. This gets you “steps.py is canonical” immediately, without having to wire per-step tasks yet.

In `orchestration/prefect_flow.py`:

### 4.1 Use configs and context builder

Assuming we already added `ExportArgs` → core configs (from previous plan):

```python
from prefect import flow, task

from codeintel.core.config import SnapshotConfig, ExecutionConfig, PathsConfig
from codeintel.storage.gateway import open_gateway
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.ingestion.tools_config import build_tools_config_from_env  # or similar
from codeintel.ingestion.source_scanner import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.orchestration.steps import PipelineContext, run_pipeline
```

Helper to build configs & context (reusing what we sketched earlier):

```python
def _build_context(args: ExportArgs) -> PipelineContext:
    snapshot = args.snapshot_config()
    storage_config = args.storage_config()

    # Build execution config
    tools = build_tools_config_from_env()
    code_profile = profile_from_env(default_code_profile(snapshot.repo_root))
    config_profile = profile_from_env(default_config_profile(snapshot.repo_root))
    graph_backend = GraphBackendConfig.from_name(args.graph_backend_name)

    execution = args.execution_config(
        tools=tools,
        code_profile=code_profile,
        config_profile=config_profile,
        graph_backend=graph_backend,
    )

    paths = PathsConfig(snapshot=snapshot, execution=execution)
    gateway = _get_gateway(storage_config)

    tool_runner = ToolRunner(tools)
    tool_service = ToolService(runner=tool_runner, tools_config=tools)

    return PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
        tool_service=tool_service,
        change_tracker=None,
    )
```

### 4.2 Single-task wrap

```python
@task
def run_full_pipeline_task(ctx: PipelineContext) -> None:
    # This is just a Prefect wrapper around our pure-Python engine
    run_pipeline(ctx)
```

Then the `flow`:

```python
@flow(name="codeintel-export")
def export_flow(args: ExportArgs) -> None:
    ctx = _build_context(args)
    try:
        run_full_pipeline_task.submit(ctx)
    finally:
        _close_gateways()
```

That’s Phase A:

* `steps.py` is canonical DAG.
* Prefect is just a wrapper around `run_pipeline`.
* No more duplicate wiring in `prefect_flow.py`.

This is already a huge win and is much simpler to implement than per-step tasks.

---

## 5. Phase B: Prefect tasks per step using `PIPELINE_STEPS`

Once Phase A is stable, you can decide if you want **per-step Prefect tasks** (for retries, caching, etc.). This is more advanced, but we can do it without breaking the “steps.py is canonical” invariant.

### 5.1 Make a generic Prefect task for one step

The key idea: Prefect passes *only configs* (which are picklable) into tasks and each task reconstructs its own `PipelineContext` with a cached gateway & tool service.

First, define a small, picklable “config-only” context:

```python
# src/codeintel/orchestration/steps.py or a small shared module

from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineContextConfig:
    snapshot: SnapshotConfig
    execution: ExecutionConfig
    paths: PathsConfig
    storage_config: StorageConfig
```

In Prefect flow:

```python
from codeintel.orchestration.steps import (
    PIPELINE_STEPS,
    PIPELINE_STEPS_BY_NAME,
    PIPELINE_DEPS,
    PipelineContext,
    PipelineContextConfig,
)
```

Global gateway cache (already in your file):

```python
_GATEWAY_CACHE: dict[str, StorageGateway] = {}

def _get_gateway(storage: StorageConfig) -> StorageGateway:
    key = str(storage.db_path) + "|" + str(storage.read_only)
    try:
        return _GATEWAY_CACHE[key]
    except KeyError:
        gw = open_gateway(storage)
        _GATEWAY_CACHE[key] = gw
        return gw

def _close_gateways() -> None:
    for gw in _GATEWAY_CACHE.values():
        gw.con.close()
    _GATEWAY_CACHE.clear()
```

Build tool service per task (cheap):

```python
def _build_tool_service(execution: ExecutionConfig) -> ToolService:
    tools = execution.tools
    runner = ToolRunner(tools)
    return ToolService(runner=runner, tools_config=tools)
```

Generic task:

```python
@task(name="run_pipeline_step")
def run_pipeline_step_task(
    step_name: str,
    ctx_cfg: PipelineContextConfig,
) -> None:
    step = PIPELINE_STEPS_BY_NAME[step_name]

    gateway = _get_gateway(ctx_cfg.storage_config)
    tool_service = _build_tool_service(ctx_cfg.execution)

    ctx = PipelineContext(
        snapshot=ctx_cfg.snapshot,
        execution=ctx_cfg.execution,
        paths=ctx_cfg.paths,
        gateway=gateway,
        tool_service=tool_service,
        change_tracker=None,  # if you later persist ChangeTracker, you can reconstruct it here
    )

    step.run(ctx)
```

### 5.2 Build the Prefect DAG dynamically from `PIPELINE_DEPS`

In `export_flow`:

```python
@flow(name="codeintel-export")
def export_flow(args: ExportArgs) -> None:
    snapshot, execution, storage, paths = _build_configs(args)
    ctx_cfg = PipelineContextConfig(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        storage_config=storage,
    )

    try:
        # Build Prefect task graph from the pipeline DAG definition
        task_results: dict[str, any] = {}

        # Topological order using the same helper as steps.run_pipeline
        step_names = [step.name for step in PIPELINE_STEPS]
        ordered = _topological_order(step_names)  # import from steps or reimplement

        for name in ordered:
            deps = PIPELINE_DEPS[name]
            upstream = [task_results[d] for d in deps]
            if upstream:
                result = run_pipeline_step_task.submit(
                    name,
                    ctx_cfg,
                    wait_for=upstream,
                )
            else:
                result = run_pipeline_step_task.submit(name, ctx_cfg)
            task_results[name] = result

    finally:
        _close_gateways()
```

Notes:

* **Critically**, Prefect no longer knows *what* a step does. It only knows:

  * step name (string)
  * its dependencies (from `PIPELINE_DEPS`)
  * that it should call `step.run(ctx)` inside `run_pipeline_step_task`.

* All changes to pipeline structure now involve only `steps.py`.

### 5.3 What about `ChangeTracker`?

With per-step tasks, you can’t share in-memory mutable state across steps easily. There are two reasonable options:

1. **Simplest**: For the Prefect DAG, recompute changes in each step as needed using the DB. The pure-Python pipeline can still share `ChangeTracker` in memory.

2. **More sophisticated**: Persist `ChangeSet` or `ChangeTracker`-like data to a table in DuckDB (e.g. `core.change_state`), so each step can reconstruct what it needs. This can be added later; it’s independent of the “canonical DAG” change.

For now, I’d recommend:

* Keep `ChangeTracker` for pure-Python / local runs.
* In Prefect mode, let each step that needs incremental info do a fresh `compute_changes` based on `core.file_state` (slower, but simpler and not a blocker to this refactor).

You can add an optional `mode` flag in `PipelineContext`/ExecutionConfig` if you want steps to know whether they’re in “simple incremental” vs “shared tracker” mode.

---

## 6. Clean up duplication and old code

Once the above is in place:

1. **Delete per-step Prefect tasks** that call ingestion/analytics directly:

   * Things like `@task def repo_scan_task(...)`, `@task def graph_metrics_task(...)`, etc., in `prefect_flow.py`.
   * All direct imports of ingestion/analytics modules from `prefect_flow.py` (they should be used only from `steps.py`).

2. Keep Prefect-specific utilities:

   * `_get_gateway`, `_close_gateways`
   * Logging configuration
   * `_snapshot_db_state` (if you have it): you can either:

     * Call it directly from `export_flow` (before/after DAG).
     * Or add a special “snapshot” step in `steps.py` if you want it to be part of the DAG.

3. Confirm that:

   * Adding/removing a pipeline step only requires editing `steps.py` (step class, `PIPELINE_STEPS`, `deps`).
   * `prefect_flow.py` never needs to be touched for pipeline structure changes.

---

## 7. Tests & guardrails

### 7.1 Pipeline consistency test

Add a test to make sure `run_pipeline` and the Prefect wiring use the same DAG:

```python
def test_pipeline_topological_order_matches_prefect_dag() -> None:
    # This mostly ensures _topological_order runs without cycles
    step_names = [s.name for s in PIPELINE_STEPS]
    ordered = _topological_order(step_names)
    assert set(ordered) == set(step_names)
```

You don’t need to spin up Prefect in tests to validate the DAG; the important part is that `PIPELINE_DEPS` is internally consistent.

### 7.2 End-to-end “pure Python” pipeline test

Add a smoke test that:

* Builds a fake `PipelineContext` with:

  * temporary repo_root
  * temporary DuckDB
  * minimal ToolsConfig (maybe stubbed)
  * simple ScanProfiles
* Calls `run_pipeline(ctx)` and asserts that:

  * Key tables exist (e.g. `core.modules`, `core.ast_nodes`, `docs.v_function_summary`).
  * No exceptions are thrown.

This ensures the pipeline can run without Prefect.

### 7.3 End-to-end Prefect flow test (optional)

If you want, add an integration test that calls:

```python
export_flow(ExportArgs(...))
```

with a tiny repo (or fixture) and asserts that:

* It completes successfully.
* DB has the same key tables as in the pure-Python test.

---

If you’d like, I can next zoom in on your current `orchestration/steps.py` and `orchestration/prefect_flow.py` structure (based on the repo) and write a diff-style plan specifically for:

* adding `run_pipeline` and `PIPELINE_STEPS_BY_NAME` in `steps.py`, and
* removing the duplicate direct calls from `prefect_flow.py` and replacing them with the single `run_full_pipeline_task` or the generic `run_pipeline_step_task` pattern.
